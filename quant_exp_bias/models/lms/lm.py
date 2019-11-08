from typing import Dict, List, Tuple, Optional, Callable

import logging

import numpy
import math
from overrides import overrides
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell, LSTM
from torch.distributions import Categorical

from allennlp.nn import InitializerApplicator, RegularizerApplicator

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.training.metrics import BLEU, Perplexity, Average

from quant_exp_bias.metrics.exposure_bias import ExposureBias
from quant_exp_bias.models.sampled_beam_search import SampledBeamSearch
from quant_exp_bias.oracles.oracle_base import Oracle

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class LMBase(Model):
    """
    TODO (Kushal): Rewrite function doc to reflect that this is base lm class. 
    This ``LMBase`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : ``int``
        Maximum length of decoded sequences.
    target_namespace : ``str``, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : ``int``, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention : ``Attention``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    attention_function: ``SimilarityFunction``, optional (default = None)
        This is if you want to use the legacy implementation of attention. This will be deprecated
        since it consumes more memory than the specialized attention modules.
    beam_size : ``int``, optional (default = None)
        Width of the beam for beam search. If not specified, greedy decoding is used.
    scheduled_sampling_ratio : ``float``, optional (default = 0.)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        `Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015 <https://arxiv.org/abs/1506.03099>`_.
    use_bleu : ``bool``, optional (default = True)
        If True, the BLEU metric will be calculated during validation.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 use_in_seq2seq_mode: bool,
                 max_decoding_steps: int,
                 generation_batch_size: int,
                 target_embedding_dim: int = None,
                 target_output_dim : int = None,
                 target_namespace: str = "tokens",
                 beam_size: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 scheduled_sampling_k: int = 100,
                 scheduled_sampling_type: str = 'uniform',
                 use_bleu: bool = True,
                 dropout: float = None,
                 sample_output: bool = False, 
#                 start_index: str = '<S>',
#                 end_index: str = '</S>',
                 start_token: str =START_SYMBOL,
                 end_token: str = END_SYMBOL,
                 num_decoder_layers:int = 1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 mask_pad_and_oov: bool = True,

                 oracle: Oracle = None,

                 # This fields will only come into play in Seq2Seq mode.
                 source_embedder: TextFieldEmbedder = None,
                 encoder: Seq2SeqEncoder = None,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None) -> None:
        super(LMBase, self).__init__(vocab)
        
        self._seq2seq_mode = use_in_seq2seq_mode

        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._scheduled_sampling_k = scheduled_sampling_k
        self._scheduled_sampling_type = scheduled_sampling_type
        self._generation_batch_size = generation_batch_size
        self._sample_output = sample_output
        self._mask_pad_and_oov = mask_pad_and_oov

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(start_token, self._target_namespace)
        self._end_index = self.vocab.get_token_index(end_token, self._target_namespace)
        
        padding_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, self._target_namespace)
        oov_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN, self._target_namespace)

        if self._mask_pad_and_oov:
            self._vocab_mask = torch.ones(self.vocab.get_vocab_size(self._target_namespace),
                                        device=torch.cuda.current_device()) \
                                    .scatter(0, torch.tensor([padding_index,oov_index],
                                                 device=torch.cuda.current_device()), 0)

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)  # pylint: disable=protected-access
            self._bleu = BLEU(exclude_indices={pad_index, self._end_index, self._start_index})
        else:
            self._bleu = None

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 1
        self._max_decoding_steps = max_decoding_steps

        # TODO(Kushal): Pass in the arguments for sampled. Also, make sure you do not sample in case of Seq2Seq models.
        self._beam_search = SampledBeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        self._num_classes = self.vocab.get_vocab_size(self._target_namespace)

        # Dense embedding of vocab words in the target space.
        self._target_embedder = Embedding(self._num_classes, target_embedding_dim)

        self._seq2seq_mode = use_in_seq2seq_mode
        if self._seq2seq_mode:
            # Dense embedding of source vocab tokens.
            self._source_embedder = source_embedder

            # Encodes the sequence of source embeddings into a sequence of hidden states.
            self._encoder = encoder

            self._encoder_output_dim = self._encoder.get_output_dim()

        self._perplexity = Perplexity()

        if oracle is not None:
            self._exposure_bias = ExposureBias(oracle)

        self._ss_ratio = Average()
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x


        if self._seq2seq_mode:
            # Decoder output dim needs to be the same as the encoder output dim since we initialize the
            # hidden state of the decoder with the final hidden state of the encoder.
            self._decoder_output_dim = self._encoder_output_dim
        else:
            self._decoder_output_dim = target_output_dim

        # Attention mechanism applied to the encoder output for each step.
        if attention:
            if attention_function:
                raise ConfigurationError("You can only specify an attention module or an "
                                         "attention function, but not both.")
            self._attention = attention
        elif attention_function:
            self._attention = LegacyAttention(attention_function)
        else:
            self._attention = None

        # Ensure that attention is only set during seq2seq setting.
        if not self._seq2seq_mode and self._attention is not None:
            raise ConfigurationError("Attention is only specified in Seq2Seq setting.")

        if self._attention:
            # If using attention, a weighted average over encoder outputs will be concatenated
            # to the previous target embedding to form the input to the decoder at each
            # time step.
            self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim
        else:
            # Otherwise, the input to the decoder is just the previous target embedding.
            self._decoder_input_dim = target_embedding_dim

        self._num_decoder_layers = num_decoder_layers

        if self._num_decoder_layers > 1:
            self._decoder_cell = LSTM(self._decoder_input_dim, self._decoder_output_dim, self._num_decoder_layers)
        else:
            # We'll use an LSTM cell as the recurrent cell that produces a hidden state
            # for the decoder at each time step.
            # TODO (pradeep): Do not hardcode decoder cell type.
            self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        self.training_iteration = 0
        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self._decoder_output_dim, self._num_classes)

        initializer(self)

    def take_step(self,
                  timestep,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor],
                  targets: torch.LongTensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
            # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
            # during training.
            # shape: (batch_size,)
            input_choices = last_predictions
        elif targets is None:
            # shape: (batch_size,)
            input_choices = last_predictions
        else:
            # shape: (batch_size,)
            input_choices = targets[:, timestep]
        
        # shape: (group_size, num_classes)
        class_logits, state = self._prepare_output_projections(input_choices, state)

        if not self.training and self._mask_pad_and_oov:
            # This implementation is copied from masked_log_softmax from allennlp.nn.util.
            mask = (self._vocab_mask.expand(class_logits.shape) + 1e-45).log()
            # shape: (group_size, num_classes)
            class_logits = class_logits + mask

        return class_logits, state

    @overrides
    def forward(self,  # type: ignore
                target_tokens: Dict[str, torch.LongTensor] = None,
                source_tokens: Dict[str, torch.LongTensor] = None,
                compute_exposure_bias: bool = False,
                generation_batch_size:int = 1024,
                max_decoding_step: int = None) -> Dict[str, torch.Tensor]:
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
        
        if self._seq2seq_mode:
            state.update(self._encode(source_tokens))
            state = self._init_decoder_state_from_encoder(state)

        # These are default for validation and compute exposure bias run.
        beam_size: int = None; per_node_beam_size: int = None; 
        sampled: bool = True; truncate_at_end_all: bool = False;
        targets: torch.LongTensor = None
        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        if self.training:
            beam_size: int = 1; per_node_beam_size: int = self._num_classes; 
            sampled: bool = False; truncate_at_end_all: bool = False;            
            
            self._apply_scheduled_sampling()

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        start_predictions = self._get_start_predictions(state,
                                                        target_tokens,
                                                        generation_batch_size)

        rolling_policy=partial(self.take_step, targets=targets)
        output_dict = self._forward_loop(state, 
                                         start_predictions,
                                         rolling_policy,
                                         max_steps=num_decoding_steps,
                                         beam_size=beam_size,
                                         per_node_beam_size=per_node_beam_size,
                                         sampled=sampled,
                                         truncate_at_end_all=truncate_at_end_all)

        if target_tokens:
            targets = target_tokens['tokens']
            logits = output_dict['logits']

            # shape: (batch_size, num_decoding_steps)
            best_logits = logits[:, 0, :, :].squeeze(1)

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(best_logits, targets, target_mask)
            output_dict["loss"] = loss
            self._perplexity(loss)

        if not self.training:
            if target_tokens and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]

                self._bleu(best_predictions, target_tokens["tokens"])

            
            if compute_exposure_bias and self._exposure_bias:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                top_k_log_probabilities = output_dict["class_log_probabilities"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]

                prediction_loss = top_k_log_probabilities[:,0]
                predicted_tokens = self._decode_tokens(best_predictions, 
                                                        vocab_namespace=self._target_namespace,
                                                        truncate=True)
                
                self._exposure_bias(prediction_loss.data, predicted_tokens)

                output_dict['predicted_tokens'] = predicted_tokens
                output_dict['prediction_loss'] = prediction_loss
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
            indices = list(indices)
            # Collect indices till the first end_symbol

            if truncate and self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=vocab_namespace)
                                for x in indices]
            
            all_predicted_tokens.append(predicted_tokens)
        return all_predicted_tokens


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        all_predicted_tokens = self._decode_tokens(predicted_indices, 
                                                    vocab_namespace=self._target_namespace,
                                                    truncate=True)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)

        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input_w_dropout = self._dropout(embedded_input)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input_w_dropout, source_mask)
        
        return {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
        }

    def _init_decoder_state_from_encoder(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
                state["encoder_outputs"],
                state["source_mask"],
                self._encoder.is_bidirectional())

        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: ((batch_size, decoder_output_dim), (batch_size, decoder_output_dim))
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"]\
                                        .new_zeros(batch_size, self._decoder_output_dim)
        return state

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


    def train_batch(self):

        self.roll_in()
        self.roll_out()
        pass

    def eval_batch(self):
        pass

    def rollin(self,
               state: Dict[str, torch.Tensor],
               target_tokens: Dict[str, torch.LongTensor] = None,):
        pass

    def rollout(self,
               state: Dict[str, torch.Tensor],
               generation_batch_size:int = None,):
        pass

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      start_predictions, 
                      rolling_policy,
                      max_steps:int = None,
                      beam_size:int = None, 
                      per_node_beam_size:int = None,
                      sampled:bool = True,
                      truncate_at_end_all: bool = True) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        # shape (logits): (batch_size, beam_size, num_decoding_steps, num_classes)
        step_predictions, log_probabilities, logits = \
                    self._beam_search.search(start_predictions, 
                                                state,
                                                rolling_policy,
                                                max_steps=max_steps,
                                                beam_size=beam_size,
                                                per_node_beam_size=per_node_beam_size,
                                                sampled=sampled,
                                                truncate_at_end_all=truncate_at_end_all)

        output_dict = {"predictions": step_predictions,
                       "logits": logits,
                       "class_log_probabilities": log_probabilities,
                       "scheduled_sampling_ratio": self._scheduled_sampling_ratio}
        return output_dict

    def _get_start_predictions(self, 
              state: Dict[str, torch.Tensor], 
              target_tokens: torch.LongTensor = None,
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
         # shape: ((group_size, decoder_output_dim), (group_size, decoder_output_dim))
        decoder_hidden = state.get("decoder_hidden", None)

        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_context = state.get("decoder_context", None)

        assert decoder_hidden is None and decoder_context is None or \
                decoder_hidden is not None and decoder_context is not None, \
            "Either decoder_hidden and context should be None or both should exist."

        decoder_hidden_and_context = None \
                                        if decoder_hidden is None or \
                                           decoder_context is None else \
                                     (decoder_hidden.transpose(0,1).contiguous(),
                                      decoder_context.transpose(0,1).contiguous()) \
                                         if self._num_decoder_layers > 1 else \
                                     (decoder_hidden, decoder_context)

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        embedded_input_w_dropout = self._dropout(embedded_input)

        if self._attention:

            # shape: (group_size, max_input_sequence_length, encoder_output_dim)
            encoder_outputs = state["encoder_outputs"]

            # shape: (group_size, max_input_sequence_length)
            source_mask = state["source_mask"]

            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)

            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input_w_dropout), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = embedded_input_w_dropout

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        if self._num_decoder_layers > 1:
            _, (decoder_hidden, decoder_context) = self._decoder_cell(decoder_input.unsqueeze(0),
                                                                      decoder_hidden_and_context)
        else:
            decoder_hidden, decoder_context = self._decoder_cell(decoder_input, 
                                                                 decoder_hidden_and_context)
        # add dropout
        decoder_hidden_with_dropout = self._dropout(decoder_hidden)
        
        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_hidden_with_dropout[-1]
                                                            if self._num_decoder_layers > 1 else
                                                                decoder_hidden_with_dropout)
        if self._num_decoder_layers > 1:
            decoder_hidden = decoder_hidden.transpose(0,1).contiguous()
            decoder_context = decoder_context.transpose(0,1).contiguous()

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context
        return output_projections, state

    def _prepare_attended_input(self,
                                decoder_hidden_state: torch.LongTensor = None,
                                encoder_outputs: torch.LongTensor = None,
                                encoder_outputs_mask: torch.LongTensor = None) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        # shape: (batch_size, max_input_sequence_length)
        encoder_outputs_mask = encoder_outputs_mask.float()

        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(decoder_hidden_state, 
                                        encoder_outputs, 
                                        encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
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

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    @overrides
    def get_metrics(self, reset: bool = False, get_exposure_bias: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if get_exposure_bias and self._exposure_bias and not self.training:
            all_metrics.update({'exposure_bias': self._exposure_bias.get_metric(reset=reset)})
            return all_metrics

        if self.training or not self._seq2seq_mode:
            all_metrics.update({'perplexity': self._perplexity.get_metric(reset=reset),
                                'ss_ratio': self._ss_ratio.get_metric(reset=reset),
                                'training_iter': self.training_iteration})
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))

        return all_metrics
