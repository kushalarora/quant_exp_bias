from typing import Dict, List, Tuple, Optional, Callable
from overrides import overrides

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.modules import Embedding
from allennlp.nn import util
from allennlp.training.metrics import Average

from quant_exp_bias.modules.decoders.decoder_net import DecoderNet
from quant_exp_bias.oracles.oracle_base import Oracle
from quant_exp_bias.modules.decoders.auto_regressive_decoder import QuantExpAutoRegressiveSeqDecoder
from quant_exp_bias.modules.decoders.seq_decoder import SeqDecoder
from quant_exp_bias.modules.cost_functions.cost_function import CostFunction
from quant_exp_bias.modules.cost_functions.noise_oracle_likelihood_cost_function import NoiseOracleCostFunction

@SeqDecoder.register("quant_exp_reinforce_decoder")
class QuantExpReinforceDecoder(QuantExpAutoRegressiveSeqDecoder):

    def __init__(self, 
                 vocab: Vocabulary,
                 max_decoding_steps: int,
                 generation_batch_size: int,
                 decoder_net: DecoderNet,
                 target_embedder: Embedding,
                 use_in_seq2seq_mode: bool = False,
                 target_namespace: str = "tokens",
                 beam_size: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 scheduled_sampling_k: int = 100,
                 scheduled_sampling_type: str = 'uniform',
                 rollin_mode: str = 'teacher_forcing',
                 rollout_mode: str = 'reference',
                 use_bleu: bool = False,
                 use_hamming: bool = False,
                 dropout: float = None,
                 sample_output: bool = False, 
                 start_token: str =START_SYMBOL,
                 end_token: str = END_SYMBOL,
                 num_decoder_layers: int = 1,
                 mask_pad_and_oov: bool = True,
                 tie_output_embedding: bool = False,
                 label_smoothing_ratio: Optional[float] = None,
                 
                 oracle: Oracle = None,
                 rollout_cost_function: CostFunction = None,
                 rollin_steps: int = 50, 
                 rollin_rollout_combination_mode='rl',
                 rollout_mixing_prob: float = 0.5,
                 num_tokens_to_rollout:int = -1,
                 num_mle_iters: int = 4000,
                 rollin_rollout_mixing_coeff:float = 0.5,
                ) -> None:
        super().__init__(vocab=vocab,
                         max_decoding_steps=max_decoding_steps,
                         generation_batch_size=generation_batch_size,
                         decoder_net=decoder_net,
                         target_embedder=target_embedder,
                         use_in_seq2seq_mode=use_in_seq2seq_mode,
                         target_namespace=target_namespace,
                         beam_size=beam_size,
                         scheduled_sampling_ratio=scheduled_sampling_ratio,
                         scheduled_sampling_k=scheduled_sampling_k,
                         scheduled_sampling_type=scheduled_sampling_type,
                         rollin_mode=rollin_mode,
                         rollout_mode=rollout_mode,
                         use_bleu=use_bleu,
                         use_hamming=use_hamming,
                         dropout=dropout,
                         sample_output=sample_output, 
                         start_token=start_token,
                         end_token=end_token,
                         num_decoder_layers=num_decoder_layers,
                         mask_pad_and_oov=mask_pad_and_oov,
                         tie_output_embedding=tie_output_embedding,
                         label_smoothing_ratio=label_smoothing_ratio,
                         oracle=oracle,
                         rollout_cost_function=rollout_cost_function,
                         rollin_rollout_combination_mode=rollin_rollout_combination_mode,
                         rollout_mixing_prob=rollout_mixing_prob,
                        )

        self._rollin_steps = rollin_steps
        self._rollin_mode = rollin_mode
        self._rollout_mode = rollout_mode
        self._combiner_mode = rollin_rollout_combination_mode

        self._combiner_loss = None
        if self._combiner_mode == 'kl':
            self._combiner_loss = torch.nn.KLDivLoss(reduction='none')

        self._num_mle_iters = num_mle_iters
        self._rollin_rollout_mixing_coeff = 0 * rollin_rollout_mixing_coeff

        self._baseline_regressor = torch.nn.Sequential(torch.nn.Linear(self._decoder_net.get_output_dim(), self._decoder_net.get_output_dim()),
                                                       torch.nn.ReLU(),
                                                      torch.nn.Linear(self._decoder_net.get_output_dim(), 1))

        self._regressor_loss = Average()

    @overrides
    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      start_predictions: torch.LongTensor, 
                      num_decoding_steps,
                      target_tokens: Dict[str, torch.LongTensor] = None,
                     ) -> Dict[str, torch.Tensor]:
        rollin_output_dict:  Dict[str, torch.Tensor] = {}
        rollout_output_dict: Dict[str, torch.Tensor] = {}
        rollin_steps = num_decoding_steps

        rollin_state = {}
        rollin_state.update(state)
        rollout_state = {}
        rollout_state.update(state)

        self._decoder_net._accumulate_hidden_states = False
        rollin_output_dict.update(self.rollin(rollin_state,
                                             start_predictions,
                                             rollin_mode=self._rollin_mode,
                                             rollin_steps=num_decoding_steps,
                                             target_tokens=target_tokens,))


        self._decoder_net._accumulate_hidden_states = True
        rollout_output_dict.update(self.rollout(rollout_state, 
                                                start_predictions, 
                                                rollout_steps=num_decoding_steps,
                                                rollout_mode=self._rollout_mode,
                                                target_tokens=target_tokens,
                                                truncate_at_end_all=False))
        rollout_output_dict['baseline_rewards'] = self._baseline_regressor(rollout_state['decoder_hiddens'].detach()).squeeze(-1)
        return rollin_output_dict, rollout_output_dict

    @overrides
    def _combine_rollin_rollout_losses(self, rollin_output_dict, rollout_output_dict, target_tokens):
        if self._combiner_mode == 'rl':
            output_dict = { 'predictions': rollin_output_dict['predictions']}
            if target_tokens:
                target_mask = util.get_text_field_mask(target_tokens)
                target_mask = target_mask[:, 1:].float()
                non_batch_dims = tuple(range(1, len(target_mask.shape)))

                # rollout_loss_batch : (batch_size,)
                rollout_reward_batch = torch.exp(-1 * rollout_output_dict['loss_batch'])
                rollout_baseline_reward = rollout_output_dict['baseline_rewards']

                baseline_reward_regressor_loss = torch.dist(rollout_reward_batch, 
                                                            rollout_baseline_reward.sum(dim=-1),
                                                            p=2)

                self._regressor_loss(baseline_reward_regressor_loss)
                if self.training_iteration < self._num_mle_iters:
                    loss_batch = rollin_output_dict['loss_batch']
                else:

                    if self.training_iteration % 100 == 0:
                        import pdb; pdb.set_trace()
                        
                    rewards = (rollout_reward_batch - 0 * rollout_baseline_reward.sum(dim=-1)).detach()
                    # rewards = rollout_reward_batch_expanded.detach()
                    rollout_logits = F.softmax(rollout_output_dict["logits"].squeeze(1))
                    predictions = rollout_output_dict["predictions"].squeeze(1)[:, 1:].unsqueeze(2)

                    probs = torch.gather(rollout_logits, -1, predictions).squeeze(2)

                    rollout_rl_loss_batch = (-1 * probs * rewards).sum(-1)
                    # rollout_logits: (batch_size,)

                    loss_batch = self._rollin_rollout_mixing_coeff *  rollin_output_dict['loss_batch']  + \
                                 (1 - self._rollin_rollout_mixing_coeff) * rollout_rl_loss_batch

                # shape : (batch_size,)
                target_mask_sum = target_mask.sum(dim=non_batch_dims)
                num_non_empty_sequences = ((target_mask_sum > 0).float().sum() + 1e-13)
                loss = loss_batch.sum()/num_non_empty_sequences
                # output_dict['loss'] = rollin_output_dict['loss'] if self.training_iteration < 10 else rollin_output_dict['loss'] + loss

                output_dict['loss'] = loss + baseline_reward_regressor_loss
            return output_dict
        return None

    @overrides
    def get_metrics(self, reset: bool = False, get_exposure_bias: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        all_metrics.update(super().get_metrics(reset, get_exposure_bias))
        all_metrics.update({'regressor_loss':  float(self._regressor_loss.get_metric(reset=reset))})

        return all_metrics