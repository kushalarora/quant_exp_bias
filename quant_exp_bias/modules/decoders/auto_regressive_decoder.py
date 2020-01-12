from typing import Dict, List, Tuple, Optional, Callable
from overrides import overrides
from functools import partial

import copy
import logging
import math
import numpy
import torch
import torch.nn.functional as F
from torch.nn import Linear

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.modules import Embedding
from allennlp.nn import util
from allennlp.training.metrics import BLEU, Perplexity, Average

from quant_exp_bias.metrics.exposure_bias import ExposureBias
from quant_exp_bias.models.sampled_beam_search import SampledBeamSearch
from quant_exp_bias.oracles.oracle_base import Oracle
from quant_exp_bias.modules.decoders.seq_decoder import SeqDecoder
from quant_exp_bias.modules.decoders.decoder_net import DecoderNet
from quant_exp_bias.modules.cost_functions.cost_function import CostFunction
from quant_exp_bias.modules.detokenizers.detokenizer import DeTokenizer, default_tokenizer

from quant_exp_bias.metrics.hamming_loss import HammingLoss



logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

RollinPolicyType = Callable[[int, torch.LongTensor, Optional[torch.LongTensor]], torch.LongTensor]
RolloutPolicyType = Callable[[torch.LongTensor, torch.LongTensor, Optional[torch.LongTensor]], torch.LongTensor]
RolloutMixingProbFuncType = Callable[[], torch.Tensor]
DeTokenizerType = Callable[[List[List[str]]], List[str]]

@SeqDecoder.register("quant_exp_auto_regressive_seq_decoder")
class QuantExpAutoRegressiveSeqDecoder(SeqDecoder):
    """
    An autoregressive decoder for Quantifying Exposure Bias experiments.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    decoder_net : ``DecoderNet``, required
        Module that contains implementation of neural network for decoding output elements
    max_decoding_steps : ``int``
        Maximum length of decoded sequences.
    target_embedder : ``Embedding``
        Embedder for target tokens.
    target_namespace : ``str``, optional (default = 'target_tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    beam_size : ``int``, optional (default = 4)
        Width of the beam for beam search.
    tensor_based_metric : ``Metric``, optional (default = None)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : ``Metric``, optional (default = None)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    scheduled_sampling_ratio : ``float`` optional (default = 0)
        Defines ratio between teacher forced training and real output usage. If its zero
        (teacher forcing only) and `decoder_net`supports parallel decoding, we get the output
        predictions in a single forward pass of the `decoder_net`.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 max_decoding_steps: int,
                 decoder_net: DecoderNet,
                 target_embedder: Embedding,
                 generation_batch_size: int = 200,
                 use_in_seq2seq_mode: bool = False,
                 target_namespace: str = "tokens",
                 beam_size: int = None,
                 scheduled_sampling_ratio: float = 0.2,
                 scheduled_sampling_k: int = 100,
                 scheduled_sampling_type: str = 'uniform',
                 rollin_mode: str = 'teacher_forcing',
                 rollout_mode: str = 'learned',
                 use_bleu: bool = False,
                 use_hamming: bool = False,
                 dropout: float = None,
                 sample_output: bool = False,
                 start_token: str =START_SYMBOL,
                 end_token: str = END_SYMBOL,
                 num_decoder_layers: int = 1,
                 mask_pad_and_oov: bool = False,
                 tie_output_embedding: bool = False,
                 label_smoothing_ratio: Optional[float] = None,

                 oracle: Oracle = None,
                 rollout_cost_function: CostFunction = None,
                 rollin_rollout_combination_mode='mle',
                 rollout_mixing_prob:float = 0.5,
                 detokenizer: DeTokenizer = default_tokenizer,
                 temperature: float = 1,
                ) -> None:
        super().__init__(target_embedder)

        self._vocab = vocab
        self._seq2seq_mode = use_in_seq2seq_mode

        # Decodes the sequence of encoded hidden states into e new sequence of hidden states.
        self._max_decoding_steps = max_decoding_steps
        self._generation_batch_size = generation_batch_size
        self._decoder_net = decoder_net

        self._target_namespace = target_namespace
        self._label_smoothing_ratio = label_smoothing_ratio

        # TODO (Kushal): Maybe make them modules so that we can add more of these later.
        self._rollin_mode = rollin_mode
        self._rollout_mode = rollout_mode

        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._scheduled_sampling_k = scheduled_sampling_k
        self._scheduled_sampling_type = scheduled_sampling_type
        self._sample_output = sample_output
        self._mask_pad_and_oov = mask_pad_and_oov

        self._rollout_mixing_prob = rollout_mixing_prob

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self._vocab.get_token_index(start_token, self._target_namespace)
        self._end_index = self._vocab.get_token_index(end_token, self._target_namespace)

        self._padding_index = self._vocab.get_token_index(DEFAULT_PADDING_TOKEN, self._target_namespace)
        self._oov_index = self._vocab.get_token_index(DEFAULT_OOV_TOKEN, self._target_namespace)

        if self._mask_pad_and_oov:
            self._vocab_mask = torch.ones(self._vocab.get_vocab_size(self._target_namespace),
                                        device=torch.cuda.current_device()) \
                                    .scatter(0, torch.tensor([self._padding_index, self._oov_index, self._start_index],
                                                 device=torch.cuda.current_device()), 0)
        if use_bleu:
            pad_index = self._vocab.get_token_index(self._vocab._padding_token, self._target_namespace)  # pylint: disable=protected-access
            self._bleu = BLEU(exclude_indices={pad_index, self._end_index, self._start_index})
        else:
            self._bleu = None

        if use_hamming:
            self._hamming = HammingLoss()
        else:
            self._hamming = None

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 1

        # TODO(Kushal): Pass in the arguments for sampled. Also, make sure you do not sample in case of Seq2Seq models.
        self._beam_search = SampledBeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size, temperature=temperature)

        self._num_classes = self._vocab.get_vocab_size(self._target_namespace)

        if self.target_embedder.get_output_dim() != self._decoder_net.target_embedding_dim:
            raise ConfigurationError(
                "Target Embedder output_dim doesn't match decoder module's input."
            )

        self._perplexity = Perplexity()

        if oracle is not None:
            self._oracle = oracle
            self._exposure_bias = ExposureBias(self._oracle)

        self._ss_ratio = Average()

        self._rollout_cost_function = rollout_cost_function
        if self._rollout_cost_function:
            self._rollout_cf_avg = Average()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        self.training_iteration = 0
        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self._decoder_net.get_output_dim(), self._num_classes)

        if tie_output_embedding:
            if self._output_projection_layer.weight.shape != self.target_embedder.weight.shape:
                raise ConfigurationError(
                    "Can't tie embeddings with output linear layer, due to shape mismatch"
                )
            self._output_projection_layer.weight = self.target_embedder.weight

        self._combiner_mode = rollin_rollout_combination_mode

        self._detokenizer = detokenizer

    def get_output_dim(self):
        return self._decoder_net.get_output_dim()

    def rollin_policy(self,
                      timestep: int,
                      last_predictions: torch.LongTensor,
                      target_tokens: Dict[str, torch.LongTensor] = None,
                      rollin_mode = 'teacher_forcing') -> torch.LongTensor:
        """ Roll-in policy to use.
            This takes in targets, timestep and last_predictions, and decide
            which to use for taking next step i.e., generating next token.
            What to do is decided by rolling mode. Options are
                - teacher_forcing,
                - learned,
                - mixed,

        Arguments:
            timestep {int} -- Current timestep decides which target token to use.
                              In case of teacher_forcing this is usually {t-1}^{th} timestep
                              for predicting t^{th} token.
            last_predictions {torch.LongTensor} -- {t-1}^th token predicted by the model.

        Keyword Arguments:
            targets {torch.LongTensor} -- Targets value if it is available. This will be
                                           available in training mode but not in inference mode. (default: {None})
            rollin_mode {str} -- Rollin mode. Options are
                                  teacher_forcing, learned, scheduled-sampling (default: {'teacher_forcing'})
        Returns:
            torch.LongTensor -- The method returns input token for predicting next token.
        """
        # For first timestep, you are passing start token, so don't do anything smart.
        if (timestep == 0 or
           # If no targets, no way to do teacher_forcing, so use your own predictions.
           target_tokens is None  or
           rollin_mode == 'learned'):
            # shape: (batch_size,)
            return last_predictions

        targets = target_tokens['tokens']
        if rollin_mode == 'teacher_forcing':
            # shape: (batch_size,)
            input_choices = targets[:, timestep]
        elif rollin_mode == 'mixed':
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]
        else:
            raise ConfigurationError(f"invalid configuration for rollin policy: {rollin_mode}")
        return input_choices

    def rollout_policy(self,
                       timestep: int,
                       logits: torch.LongTensor,
                       target_tokens: Dict[str, torch.LongTensor] = None,
                       rollout_mode: str = 'learned',
                       rollout_mixing_func: RolloutMixingProbFuncType = None,
                      ) -> torch.LongTensor:
        """Rollout policy to use.
           This takes in predicted logits at timestep {t}^{th} and
           depending upon the rollout_mode replaces some of the predictions
           with targets.

           The options for rollout mode are:
               - learned,
               - reference,
               - mixed.

        Arguments:
            timestep {int} -- Current timestep decides which target token to use.
                              In case of reference this is usually {t-1}^{th} timestep
                              for predicting t^{th} token.
            logits {torch.LongTensor} -- Logits generated by the model for {t}^{th} timestep.
                                         (batch_size, num_classes).

        Keyword Arguments:
            targets {torch.LongTensor} -- Targets value if it is available. This will be
                                available in training mode but not in inference mode. (default: {None})
            rollout_mode {str} -- Rollout mode: Options are:
                                    learned, reference, mixed. (default: {'learned'})
            rollout_mixing_func {RolloutMixingProbFuncType} -- Function to get mask to choose predicted logits vs targets in case of mixed
                                    rollouts.  (default: {0.5})

        Returns:
            torch.LongTensor -- The method returns logits with rollout policy applied.
        """
        output_logits = logits
        assert rollout_mode == 'learned' or target_tokens is not None, \
            f"Rollout mode {rollout_mode} needs targets to be specified."

        if rollout_mode == 'learned' or target_tokens is None:
            # For learned rollout policy, just return the same logits.
            return output_logits

        targets = target_tokens['tokens']
        seq_len = targets.size(1)

        # So this must be prediction of last step,
        # where there are no more
        if seq_len > timestep + 1:  # + 1 because timestep is an index, indexed at 0.
            # As we might be overriding  the next/predicted token/
            # We have to use the value corresponding to {t+1}^{th}
            # timestep.
            target_at_timesteps = targets[:, timestep + 1]
        else:
            # We have overshot the seq_len, so just repeat the
            # last token which is either _end_token or _pad_token.
            target_at_timesteps = targets[:, -1]

        # target_logits: (batch_size, num_classes).
        # This tensor has 0 at targets and (near) -inf at other places.
        target_logits = (target_at_timesteps.new_zeros(logits.shape) + 1e-45) \
                            .scatter_(dim=1,
                                      index=target_at_timesteps.unsqueeze(1),
                                      value=1.0).log()
        batch_size = logits.size(0)
        if rollout_mode == 'reference':
             output_logits += target_logits
        elif rollout_mode == 'mixed':
            # Based on the mask (Value=1), copy target values.

            if rollout_mixing_func is not None:
                rollout_mixing_prob_tensor = rollout_mixing_func()
            else:
                # This returns a (batch_size, num_classes) boolean map where the rows are either all zeros or all ones.
                rollout_mixing_prob_tensor = torch.bernoulli(torch.ones(batch_size) * self._rollout_mixing_prob)

            rollout_mixing_mask = rollout_mixing_prob_tensor \
                                                .unsqueeze(1) \
                                                .expand(logits.shape) \
                                                .to(torch.cuda.current_device())

            # The target_logits ranges from (-inf , 0), so, by adding those to logits,
            # we turn the values that are not target token to -inf, hence making the distribution
            # skew towards the target.
            output_logits += rollout_mixing_mask * target_logits
        else:
            raise ConfigurationError(f"Incompatible rollout mode: {rollout_mode}")
        return output_logits

    def take_step(self,
                  timestep: int,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor],
                  target_tokens: Dict[str, torch.LongTensor] = None,
                  rollin_mode: str = 'learned',
                  rollout_mode: str = 'learned',
                  rollout_mixing_func: RolloutMixingProbFuncType = None,
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """

        input_choices = self.rollin_policy(timestep,
                                           last_predictions,
                                            target_tokens,
                                            rollin_mode=rollin_mode)

        # shape: (group_size, num_classes)
        class_logits, state = self._prepare_output_projections(input_choices, state)

        if not self.training and self._mask_pad_and_oov:
            # This implementation is copied from masked_log_softmax from allennlp.nn.util.
            mask = (self._vocab_mask.expand(class_logits.shape) + 1e-45).log()
            # shape: (group_size, num_classes)
            class_logits = class_logits + mask

        class_logits = self.rollout_policy(timestep,
                                            class_logits,
                                            target_tokens,
                                            rollout_mode=rollout_mode, 
                                            rollout_mixing_func=rollout_mixing_func)
        return class_logits, state

    @overrides
    def forward(self,  # type: ignore
                encoder_out: Dict[str, torch.LongTensor] = {},
                target_tokens: Dict[str, torch.LongTensor] = None,
                compute_exposure_bias: bool = False,
                generation_batch_size:int = 1024,
                max_decoding_step: int = None,) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        source_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        output_dict:  Dict[str, torch.Tensor] = {}
        state:  Dict[str, torch.Tensor] = {}
        decoder_init_state: Dict[str, torch.Tensor] = {}

        # In Seq2Seq setting, we will encode the source sequence,
        # and init the state object with encoder output and decoder
        # cell will use these encoder outputs for attention/initing
        # the decoder states.
        if self._seq2seq_mode:
            state = encoder_out
            decoder_init_state = self._decoder_net.init_decoder_state(state)
            state.update(decoder_init_state)

        # In case we have target_tokens, roll-in and roll-out
        # only till those many steps, otherwise we roll-out for
        # `self._max_decoding_steps`.
        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

       # Initialize target predictions with the start index.
        # shape: (batch_size,)
        start_predictions = self._get_start_predictions(state,
                                                        target_tokens,
                                                        generation_batch_size)

        # This is training loop.
        if target_tokens:
            rollin_output_dict, rollout_output_dict = \
                    self._forward_loop(state,
                                        start_predictions,
                                        num_decoding_steps=num_decoding_steps,
                                        target_tokens=target_tokens)

            output_dict.update(self._combine_rollin_rollout_losses(rollin_output_dict,
                                                                    rollout_output_dict,
                                                                    target_tokens,))

            # The rollin loss (w or w/o teacher_forcing is perplexity.)
            self._perplexity(rollin_output_dict['loss'])

        if not self.training:
            # While validating, testing, or computing exposure bias
            # we need to roll out the learned policy and the output
            # of this rollout is used to compute the secondary metrics
            # like BLEU, or exposure bias.
            state = {}
            state.update(decoder_init_state)
            rollout_output_dict = self.rollout(state,
                                                start_predictions,
                                                rollout_steps=num_decoding_steps,
                                                rollout_mode='learned',
                                                sampled=compute_exposure_bias,
                                                truncate_at_end_all=False)

            if target_tokens and self._rollout_cost_function:
                # all beams.
                top_k_predictions = rollout_output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]

                if self._rollout_cost_function.takes_decoded_input():
                    predicted_tokens = self._decode_tokens(best_predictions,
                                                            vocab_namespace=self._target_namespace,
                                                            truncate=True)

                    decoded_targets = self._decode_tokens(target_tokens['tokens'],
                                                          vocab_namespace=self._target_namespace,
                                                          truncate=True)

                    loss_batch = self._rollout_cost_function(predicted_tokens, decoded_targets)

                else:
                    # This is for rollout cost function like hamming loss for OCR.
                    target_mask = util.get_text_field_mask(target_tokens)
                    loss_batch = self._rollout_cost_function(best_predictions, target_tokens['tokens'], target_mask)

                mask = util.get_text_field_mask({'predictions': best_predictions})
                non_batch_dims = tuple(range(1, len(mask.shape)))
                # shape : (batch_size,)
                mask_sum = mask.sum(dim=non_batch_dims)
                num_non_empty_sequences = ((mask_sum > 0).float().sum() + 1e-13)
                loss = loss_batch.sum()/num_non_empty_sequences

                self._rollout_cf_avg(float(loss.cpu()))

            if target_tokens and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = rollout_output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]

                self._bleu(best_predictions, target_tokens["tokens"])


            if target_tokens and self._hamming:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = rollout_output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]

                target_mask = util.get_text_field_mask(target_tokens)

                self._hamming(best_predictions, target_tokens["tokens"], target_mask)

            if compute_exposure_bias and self._exposure_bias:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = rollout_output_dict["predictions"]
                top_k_log_probabilities = rollout_output_dict["class_log_probabilities"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]

                prediction_loss = top_k_log_probabilities[:,0]
                predicted_tokens = self._decode_tokens(best_predictions,
                                                        vocab_namespace=self._target_namespace,
                                                        truncate=True)

                predicted_tokens, model_probs, oracle_probs = self._exposure_bias(prediction_loss.data.cpu(),
                                                                                    predicted_tokens)

                output_dict['predicted_tokens'] = predicted_tokens
                output_dict['model_probs'] =model_probs
                output_dict['oracle_probs'] = oracle_probs
                output_dict['prediction_loss'] = prediction_loss.data.cpu()
                output_dict['predicted_sequences'] = self._detokenizer(predicted_tokens)
        return output_dict

    def _decode_tokens(self,
                       predicted_indices: torch.Tensor,
                       vocab_namespace:str ='tokens',
                       truncate=False) -> List[str]:
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]

            # We add start token to the predictions.
            # In case it is present at position 0, remove it.
            if self._start_index == indices[0]:
                indices = indices[1:]

            indices = list(indices)
            # Collect indices till the first end_symbol
            if truncate and self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self._vocab.get_token_from_index(x, namespace=vocab_namespace)
                                for x in indices]

            all_predicted_tokens.append(predicted_tokens)
        return all_predicted_tokens

    @overrides
    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        all_predicted_tokens = self._decode_tokens(predicted_indices, 
                                                    vocab_namespace=self._target_namespace,
                                                    truncate=True)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _apply_scheduled_sampling(self):

        if not self.training:
            raise RuntimeError("Scheduled Sampling can only be applied during training.")

        k = self._scheduled_sampling_k
        if self._scheduled_sampling_type == 'uniform':
            # This is same scheduled sampling ratio set by config.
            pass
        elif self._scheduled_sampling_type == 'quantized':
            self._scheduled_sampling_ratio =  1 -  k/(k + math.exp(self.training_iteration//k))
        elif self._scheduled_sampling_type == 'linear':
            self.scheduled_sampling_ratio =  1 -  k/(k + math.exp(self.training_iteration/k))
        else:
            raise ConfigurationError(f"{self._scheduled_sampling_type} is not a valid scheduled sampling type.")

        self._ss_ratio(self._scheduled_sampling_ratio)
        self.training_iteration += 1

    def _combine_rollin_rollout_losses(self,
                                       rollin_output_dict: Dict[str, torch.Tensor],
                                       rollout_output_dict: Dict[str, torch.Tensor],
                                       target_tokens) -> Dict[str, torch.LongTensor]:
        """ Given rollin and rollout, how to combine loss from rollin and
            rollout to compute final loss. This will be used to learning local
            loss such that it reflects the global loss as well.

        Arguments:
            rollin_output_dict {Dict[str, torch.Tensor]} -- Dictionary with rollin computations.
            rollout_output_dict {Dict[str, torch.Tensor]} -- Dictionary with rollin computations.
            compute_exposure_bias {bool} -- If we are computing exposure bias.

        Returns:
             output_dict {Dict[str, torch.LongTensor]} -- Updated outptut dict with global and local
                                                          loss combined.
        """

        # Here, we just do rollin for training and rollout for validation, so nothing to compute.
        return {'loss': rollin_output_dict['loss']}

    def rollin(self,
               state: Dict[str, torch.Tensor],
               start_predictions: torch.LongTensor,
               rollin_steps: int,
               rollin_mode: str = 'teacher_forcing',
               target_tokens: Dict[str, torch.LongTensor] = None,
               beam_size:int = 1,
               per_node_beam_size: int = None,
               sampled: bool = False,
               truncate_at_end_all: bool = False,
              ):

        # We cannot make a class variable as default, so making default value
        # as None and in case it is None, setting it to num_classes.
        per_node_beam_size: int = per_node_beam_size or self._num_classes

        if self.training:
            self._apply_scheduled_sampling()

        rolling_policy=partial(self.take_step,
                                target_tokens=target_tokens,
                                rollin_mode=rollin_mode)

        # shape (step_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        # shape (logits): (batch_size, beam_size, num_decoding_steps, num_classes)
        step_predictions, log_probabilities, logits = \
                    self._beam_search.search(start_predictions,
                                                state,
                                                rolling_policy,
                                                max_steps=rollin_steps,
                                                beam_size=beam_size,
                                                per_node_beam_size=per_node_beam_size,
                                                sampled=sampled,
                                                truncate_at_end_all=truncate_at_end_all)

        logits = torch.cat(logits, dim=2)

        batch_size, beam_size, _ = step_predictions.shape
        start_prediction_length = start_predictions.size(0)
        step_predictions = torch.cat([start_predictions.unsqueeze(1) \
                                        .expand(batch_size, beam_size) \
                                        .reshape(batch_size, beam_size, 1),
                                        step_predictions],
                                        dim=-1)

        output_dict = {"predictions": step_predictions,
                        "logits": logits,
                        "class_log_probabilities": log_probabilities,}

        if target_tokens:
            targets = target_tokens['tokens']

            # shape: (batch_size, num_decoding_steps)
            best_logits = logits[:, 0, :, :].squeeze(1)
            target_mask = util.get_text_field_mask(target_tokens)

            rollin_targets = targets[:, :rollin_steps+1]
            rollin_target_masks = target_mask[:, :rollin_steps+1]

            # Compute loss.
            loss_batch = self._get_loss(best_logits, rollin_targets, rollin_target_masks)
            output_dict["loss_batch"] = loss_batch

            # Generate denominator for normalizing loss across batch.
            # Ideally this will be equal to batch_size, but this is a
            # safer way to do this. Here, we ignore sequences with all
            # pad tokens.
            non_batch_dims = tuple(range(1, len(rollin_target_masks.shape)))
            # shape : (batch_size,)
            target_mask_sum = target_mask.sum(dim=non_batch_dims)
            num_non_empty_sequences = ((target_mask_sum > 0).float().sum() + 1e-13)
            loss = loss_batch.sum()/num_non_empty_sequences

            output_dict['loss'] = loss
        return output_dict

    def rollout(self,
                state: Dict[str, torch.Tensor],
                start_predictions: torch.LongTensor,
                rollout_steps: int,
                rollout_mode: str = 'learned',
                beam_size: int = None,
                per_node_beam_size: int = None,
                target_tokens: Dict[str, torch.LongTensor] = None,
                sampled: bool = True,
                truncate_at_end_all: bool = True,
                # shape (prediction_prefixes): (batch_size, prefix_length)
                prediction_prefixes: torch.LongTensor = None,
                target_prefixes: torch.LongTensor = None,
                rollout_mixing_func: RolloutMixingProbFuncType = None,
               ):

        rolling_policy=partial(self.take_step,
                               target_tokens=target_tokens,
                               rollout_mode=rollout_mode,
                               rollout_mixing_func=rollout_mixing_func)

        # shape (step_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        # shape (logits): (batch_size, beam_size, num_decoding_steps, num_classes)
        step_predictions, log_probabilities, logits = \
                    self._beam_search.search(start_predictions,
                                                state,
                                                rolling_policy,
                                                max_steps=rollout_steps,
                                                beam_size=beam_size,
                                                per_node_beam_size=per_node_beam_size,
                                                sampled=sampled,
                                                truncate_at_end_all=truncate_at_end_all)

        logits = torch.cat(logits, dim=2)

        # Concatenate the start tokens to the predictions.They are not
        # added to the predictions by default.
        batch_size, beam_size, _ = step_predictions.shape

        start_prediction_length = start_predictions.size(0)
        step_predictions = torch.cat([start_predictions.unsqueeze(1) \
                                        .expand(batch_size, beam_size) \
                                        .reshape(batch_size, beam_size, 1),
                                        step_predictions],
                                        dim=-1)

        # There might be some predictions which might have been made by
        # rollin policy. If passed, concatenate them here.
        if prediction_prefixes is not None:
            prefixes_length = prediction_prefixes.size(1)
            step_predictions = torch.cat([prediction_prefixes.unsqueeze(1)\
                                            .expand(batch_size, beam_size, prefixes_length), 
                                         step_predictions],
                                         dim=-1)

        step_targets = None
        if target_tokens is not None:
            step_targets = target_tokens['tokens']

            if target_prefixes is not None:
                prefixes_length = target_prefixes.size(1)
                step_targets = torch.cat([target_prefixes, step_targets], dim=-1)

        output_dict = {"predictions": step_predictions,
                       "logits": logits,
                       "class_log_probabilities": log_probabilities,}

        if step_targets is not None and \
           self._rollout_cost_function is not None:
            # TODO (Kushal): Maybe return loss batches for
            # all beams.
            top_k_predictions = output_dict["predictions"]
            # shape: (batch_size, max_predicted_sequence_length)
            best_predictions = top_k_predictions[:, 0, :]

            predicted_tokens = self._decode_tokens(best_predictions,
                                                    vocab_namespace=self._target_namespace,
                                                    truncate=True)

            output_dict['predicted_tokens'] = predicted_tokens

            if self._rollout_cost_function.takes_decoded_input():
                # This is for rollout cost function like BLEU or Noisy Oracle for OCR.
                decoded_targets = None
                if step_targets is not None:
                    decoded_targets = self._decode_tokens(step_targets,
                                            vocab_namespace=self._target_namespace,
                                            truncate=True)

                loss_batch = self._rollout_cost_function(predicted_tokens, decoded_targets)
            else:
                # This is for rollout cost function like hamming loss for OCR.
                target_mask = util.get_text_field_mask({'tokens': step_targets})
                loss_batch = self._rollout_cost_function(best_predictions, step_targets, target_mask)

            output_dict["loss_batch"] = loss_batch
        return output_dict

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      start_predictions: torch.LongTensor,
                      num_decoding_steps,
                      target_tokens: Dict[str, torch.LongTensor] = None,
                     ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        rollin_output_dict = {}
        rollout_output_dict = {}

        rollin_output_dict.update(self.rollin(state,
                                                start_predictions,
                                                rollin_steps=num_decoding_steps,
                                                target_tokens=target_tokens,))

        return (rollin_output_dict, rollout_output_dict)

    def _get_start_predictions(self,
              state: Dict[str, torch.Tensor],
              target_tokens: Dict[str, torch.LongTensor] = None,
              generation_batch_size:int = None) ->  torch.LongTensor:

        if self._seq2seq_mode:
           source_mask = state["source_mask"]
           batch_size = source_mask.size()[0]
        elif target_tokens:
            batch_size = target_tokens["tokens"].size(0)
        else:
            batch_size = generation_batch_size

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        return torch.zeros((batch_size,),
                                       dtype=torch.long,
                                       device=torch.cuda.current_device()).fill_(self._start_index)

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (batch_size, 1, target_embedding_dim)
        last_predictions_embeddings = self.target_embedder(last_predictions)

        last_predictions_embeddings_w_dropout = self._dropout(last_predictions_embeddings)

        decoder_state, decoder_output = self._decoder_net(
            previous_state=state,
            last_predictions_embedding=last_predictions_embeddings_w_dropout
        )

        # Update state with new decoder state, override previous state
        state.update(decoder_state)

        # add dropout
        decoder_hidden_with_dropout = self._dropout(decoder_output)

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_hidden_with_dropout)

        return output_projections, state

    def _get_loss(self,
                  logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask, label_smoothing=self._label_smoothing_ratio, average=None)

    @overrides
    def get_metrics(self, reset: bool = False, get_exposure_bias: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}

        if get_exposure_bias and self._exposure_bias and not self.training:
            all_metrics.update({'exposure_bias': self._exposure_bias.get_metric(reset=reset)})
            return all_metrics

        all_metrics.update({'perplexity': self._perplexity.get_metric(reset=reset),
                    'ss_ratio': self._ss_ratio.get_metric(reset=reset),
                    'training_iter': self.training_iteration})

        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))

        if self._hamming and not self.training:
            all_metrics.update({'hamming': self._hamming.get_metric(reset=reset)})

        if self._rollout_cost_function:
            all_metrics.update({self._rollout_cost_function.name: self._rollout_cf_avg.get_metric(reset=reset)})
        return all_metrics

    def __del__(self):
        del self._oracle

    def __delete__(self):
        del self._oracle
