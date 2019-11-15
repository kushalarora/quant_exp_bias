from typing import Dict, List, Tuple, Optional, Callable
from overrides import overrides

import torch
import torch.nn.functional as F

import numpy as np

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.modules import Embedding
from allennlp.nn import util

from quant_exp_bias.modules.decoders.decoder_net import DecoderNet
from quant_exp_bias.oracles.oracle_base import Oracle
from quant_exp_bias.modules.decoders.auto_regressive_decoder import QuantExpAutoRegressiveSeqDecoder
from quant_exp_bias.modules.decoders.seq_decoder import SeqDecoder
from quant_exp_bias.modules.cost_functions.cost_function import CostFunction
from quant_exp_bias.modules.cost_functions.noise_oracle_likelihood_cost_function import NoiseOracleCostFunction

@SeqDecoder.register("quant_exp_searnn_decoder")
class QuantExpSEARNNDecoder(QuantExpAutoRegressiveSeqDecoder):

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
                 rollin_mode: str = 'teacher-forcing',
                 rollout_mode: str = 'teacher-forcing',
                 use_bleu: bool = True,
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
                 rollin_rollout_combination_mode='kl',
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
                         dropout=dropout,
                         sample_output=sample_output, 
                         start_token=start_token,
                         end_token=end_token,
                         num_decoder_layers=num_decoder_layers,
                         mask_pad_and_oov=mask_pad_and_oov,
                         tie_output_embedding=tie_output_embedding,
                         label_smoothing_ratio=label_smoothing_ratio,
                         oracle=oracle,
                         rollout_cost_function=NoiseOracleCostFunction(oracle),
                         rollin_rollout_combination_mode=rollin_rollout_combination_mode,
                        )

        self._rollin_steps = rollin_steps
        self._rollin_mode = rollin_mode
        self._rollout_mode = rollout_mode
        self._combiner_mode = rollin_rollout_combination_mode

        self._combiner_loss = None
        if self._combiner_mode == 'kl':
            self._combiner_loss = torch.nn.KLDivLoss(reduction='none')

    @overrides
    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      start_predictions: torch.LongTensor, 
                      num_decoding_steps,
                      computing_exposure_bias = False,
                      target_tokens: Dict[str, torch.LongTensor] = None,
                     ) -> Dict[str, torch.Tensor]:
        output_dict:  Dict[str, torch.Tensor] = {}
        rollin_steps = num_decoding_steps


        if computing_exposure_bias:
            self._decoder_net._accumulate_hidden_states = False
            import pdb;pdb.set_trace()

            return None, self.rollout(state, 
                                      start_predictions, 
                                      rollout_steps=num_decoding_steps,
                                      rollout_mode='learned')

        self._decoder_net._accumulate_hidden_states = True

        rollin_output_dict = self.rollin(state,
                                         start_predictions,
                                         rollin_steps=rollin_steps,
                                         target_tokens=target_tokens,)

        batch_size, beam_size, num_rollin_steps, num_classes = rollin_output_dict['logits'].size()
        
        skippable_tokens = set([])
        # skippable_tokens = set([self._end_index, 
        #                         self._start_index,
        #                         self._padding_index,
        #                         self._oov_index])

        non_skippable_tokens = [x for x in range(0, num_classes) if x not in skippable_tokens]
        non_skippable_num_classes = len(non_skippable_tokens)

        # shape (rollin_predictions) : (batch_size * num_decoding_steps * non_skippable_num_classes)
        rollin_start_predictions = torch.LongTensor(non_skippable_tokens) \
                                         .unsqueeze(0) \
                                         .expand(batch_size, non_skippable_num_classes) \
                                         .reshape(-1).to(torch.cuda.current_device())

        rollin_predictions = rollin_output_dict['predictions'].squeeze(1)

        # decoder_hidden: (batch_size, num_rollin_steps, hidden_state_size)
        rollin_decoder_hiddens = state['decoder_hiddens']
        batch_size, num_rollin_steps, hidden_size = rollin_decoder_hiddens.size()
        
        # decoder_context: (batch_size, num_rollin_steps,  hidden_state_size)
        rollin_decoder_context = state['decoder_contexts']
            
        rollout_logits = []
        rollout_predictions = []

        # targets_plus_1 = None
        # if target_tokens is not None:
        #     # targets Shape: (batch_size, num_decoding_steps + 1)
        #     targets = target_tokens['tokens']

        #     # targets_plus_1 Shape: (batch_size, num_decoding_steps + 2)
        #     targets_plus_1 = torch.cat([targets, targets[:, -1].unsqueeze(1)], dim=-1)
        
        for step in range(1, num_decoding_steps + 1):
            # import pdb;pdb.set_trace()
            rollout_steps = num_decoding_steps + 1 - step 

            target_tokens_truncated = None
            if target_tokens is not None:
                # targets Shape: (batch_size, num_decoding_steps + 1)
                targets = target_tokens['tokens']
                targets_step_onwards = targets[:, step:]

                targets_step_onwards_expanded = targets_step_onwards \
                                                    .unsqueeze(1) \
                                                    .expand(batch_size, non_skippable_num_classes, rollout_steps) \
                                                    .reshape(batch_size * non_skippable_num_classes, rollout_steps)

                target_tokens_truncated = {'tokens': targets_step_onwards_expanded}

            # decoder_hidden_step: (batch_size, hidden_state_size)
            decoder_hidden_step = rollin_decoder_hiddens[:, step - 1, :]

            # decoder_hidden_step_expanded: (batch_size, non_skippable_num_classes, hidden_state_size)
            decoder_hidden_step_expanded = decoder_hidden_step \
                                            .unsqueeze(1) \
                                            .expand(batch_size, non_skippable_num_classes, hidden_size)

            # decoder_context_step: (batch_size, hidden_state_size)
            decoder_context_step = rollin_decoder_context[:, step - 1, :]

            # decoder_hidden_step_expanded: (batch_size, non_skippable_num_classes, hidden_state_size)
            decoder_context_step_expanded = decoder_context_step \
                                                .unsqueeze(1) \
                                                .expand(batch_size, non_skippable_num_classes, hidden_size)

            # decoder_hidden: (batch_size * non_skippable_num_classes, 1, hidden_state_size)
            state['decoder_hiddens'] = decoder_hidden_step_expanded.reshape(-1, 1,  hidden_size)
            
            # decoder_context: (batch_size *  non_skippable_num_classes, 1, hidden_state_size)
            state['decoder_contexts'] = decoder_context_step_expanded.reshape(-1, 1, hidden_size)
            
            prediction_prefixes = rollin_predictions[:, :step] \
                                    .unsqueeze(1) \
                                    .expand(batch_size, non_skippable_num_classes, step) \
                                    .reshape(batch_size * non_skippable_num_classes, step) \
                                        if step > 0 else None

            # prediction_prefixes = targets[:, :step] \
            #                         .unsqueeze(1) \
            #                         .expand(batch_size, non_skippable_num_classes, step) \
            #                         .reshape(batch_size * non_skippable_num_classes, step) \
            #                             if step > 0 else None

            self._decoder_net._accumulate_hidden_states = False

            rollout_output_dict = self.rollout(state, 
                                                rollin_start_predictions, 
                                                rollout_steps=rollout_steps,
                                                rollout_mode=self._rollout_mode,
                                                target_tokens=target_tokens_truncated, 
                                                prediction_prefixes=prediction_prefixes, 
                                                truncate_at_end_all=False)
            
            rollout_output_dict['predictions'] = rollout_output_dict['predictions']\
                                                    .reshape(batch_size, non_skippable_num_classes, -1)

            rollout_predictions.append(rollout_output_dict['predictions'].unsqueeze(1))
            rollout_output_dict['loss_batch'] =  rollout_output_dict['loss_batch'] \
                                                    .reshape(batch_size, 1, non_skippable_num_classes)
                                                    
            rollout_logits.append(rollout_output_dict['loss_batch'])
        rollout_output_dict['loss_batch'] = torch.cat(rollout_logits, dim=1)
        rollout_output_dict['predictions'] = torch.cat(rollout_predictions, dim=1)

        return rollin_output_dict, rollout_output_dict

    @overrides
    def _combine_rollin_rollout_losses(self, rollin_output_dict, rollout_output_dict, target_tokens, compute_exposure_bias):
        if compute_exposure_bias:
            return rollout_output_dict

        if self._combiner_mode == 'kl':
            
            # import pdb;pdb.set_trace()
            output_dict = { 'predictions': rollin_output_dict['predictions']}
            if target_tokens:
                target_mask = util.get_text_field_mask(target_tokens)
                target_mask = target_mask[:, 1:].float()
                non_batch_dims = tuple(range(1, len(target_mask.shape)))

                x = F.log_softmax(rollin_output_dict['logits'].squeeze(1), dim=-1)
                y = F.softmax(-1 * rollout_output_dict['loss_batch'], dim=-1)
                kl_losses = self._combiner_loss(x,y).sum(dim=-1)
                kl_loss_batch = (kl_losses * target_mask).sum(dim=non_batch_dims)
            
                # Generate denominator for normalizing loss across batch.
                # Ideally this will be equal to batch_size, but this is a
                # safer way to do this. Here, we ignore sequences with all
                # pad tokens.
              
                # shape : (batch_size,)
                target_mask_sum = target_mask.sum(dim=non_batch_dims)
                num_non_empty_sequences = ((target_mask_sum > 0).float().sum() + 1e-13)
                loss = kl_loss_batch.sum()/num_non_empty_sequences
                # output_dict['loss'] = rollin_output_dict['loss'] if self.training_iteration < 10 else rollin_output_dict['loss'] + loss
                output_dict['loss'] = loss

            return output_dict
        elif self._combiner_mode == 'mle':
            return rollin_output_dict
        return None