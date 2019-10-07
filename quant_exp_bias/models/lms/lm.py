from typing import Dict, List, Tuple

import logging

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell
from torch.distributions import Categorical


from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.training.metrics import BLEU
from allennlp.training.metrics import Perplexity

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
    """

    def __init__(self,
                 vocab: Vocabulary,
                 oracle: Oracle,
                 use_in_seq2seq_mode: bool,
                 max_decoding_steps: int,
                 generation_batch_size: int,
                 target_embedding_dim: int = None,
                 target_output_dim : int = None,
                 target_namespace: str = "tokens",
                 beam_size: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 use_bleu: bool = True,
                 dropout: float = None,
                 sample_from_categorical: bool = True, 
#                 start_index: str = '<S>',
#                 end_index: str = '</S>',
                 start_token: str = 'S',
                 end_token: str = 'E',
                 
                 # This fields will only come into play in Seq2Seq mode.
                 source_embedder: TextFieldEmbedder = None,
                 encoder: Seq2SeqEncoder = None,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None) -> None:
        super(LMBase, self).__init__(vocab)
                
        self._seq2seq_mode = use_in_seq2seq_mode

        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(start_token, self._target_namespace)
        self._end_index = self.vocab.get_token_index(end_token, self._target_namespace)
        
        # self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        # self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)  # pylint: disable=protected-access
            self._bleu = BLEU(exclude_indices={pad_index, self._end_index, self._start_index})
        else:
            self._bleu = None

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 1
        self._max_decoding_steps = max_decoding_steps
        self._generation_batch_size = generation_batch_size
        # TODO(Kushal): Pass in the arguments for sampled. Also, make sure you do not sample in case of Seq2Seq models.
        self._beam_search = SampledBeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size, sampled=True)

        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        # Dense embedding of vocab words in the target space.
        self._target_embedder = Embedding(num_classes, target_embedding_dim)

        self._sample_from_categorical = sample_from_categorical

        if self._seq2seq_mode:

            self._seq2seq_mode = use_in_seq2seq_mode

            # Dense embedding of source vocab tokens.
            self._source_embedder = source_embedder

            # Encodes the sequence of source embeddings into a sequence of hidden states.
            self._encoder = encoder

            self._encoder_output_dim = self._encoder.get_output_dim()

        self._perplexity = Perplexity()

        self._exposure_bias = ExposureBias(oracle)

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

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)



    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

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

        if target_tokens:
            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict.update(self._forward_loop(state, target_tokens))


        if not self.training:
            if target_tokens and self._bleu:
                output_dict.update(self._forward_beam_search(state, 
                                            target_tokens=target_tokens))

                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                
                self._bleu(best_predictions, target_tokens["tokens"])

            
            if compute_exposure_bias and self._exposure_bias:
                output_dict.update(self._forward_beam_search(state, 
                                            generation_batch_size=generation_batch_size))
 
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                top_k_log_probabilities = output_dict["class_log_probabilities"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]

                prediction_loss = top_k_log_probabilities[:,0]
                predicted_tokens = self._decode_tokens(best_predictions, truncate=True)
                
                self._exposure_bias(prediction_loss.data, predicted_tokens)

                output_dict['predictions'] = predicted_tokens
                output_dict['prediction_loss'] = prediction_loss
        return output_dict

    def _decode_tokens(self, predicted_indices: torch.Tensor, truncate=False) -> List[str]:
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
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
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
        all_predicted_tokens = self._decode_tokens(predicted_indices, True)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
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
        state["decoder_context"] = state["encoder_outputs"]\
                                        .new_zeros(batch_size, self._decoder_output_dim)
        return state

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """

        if self._seq2seq_mode:
           source_mask = state["source_mask"]
           batch_size = source_mask.size()[0]
        elif target_tokens:
            batch_size = target_tokens["tokens"].size(0)
        else:
            batch_size = self._generation_batch_size

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
        last_predictions = torch.zeros((batch_size,), 
                                       dtype=torch.long, 
                                       device=torch.cuda.current_device()).fill_(self._start_index)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        step_prediction_loss: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(input_choices, state)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)

            if self._sample_from_categorical:
                predicted_classes = torch.multinomial(class_probabilities, 1)
                prediction_loss = torch.gather(class_probabilities, 1, predicted_classes)
                predicted_classes, prediction_loss = predicted_classes.squeeze(1), prediction_loss.squeeze(1)
            else:
                prediction_loss, predicted_classes = torch.max(class_probabilities, 1)
            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))
            step_prediction_loss.append(prediction_loss.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)
        prediction_mask = util.get_text_field_mask({"tokens": predictions})

        prediction_loss = torch.cat(step_prediction_loss, 1)
        prediction_loss *= prediction_mask.float()

        output_dict = {"predictions": predictions,
                       "prediction_loss": prediction_loss}
     
        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)
        
            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss
            self._perplexity(loss)
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


    def _forward_beam_search(self,
                             state: Dict[str, torch.Tensor],
                             target_tokens: torch.LongTensor = None,
                             generation_batch_size:int = None) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        start_predictions = self._get_start_predictions(state, 
                                                        target_tokens, 
                                                        generation_batch_size)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
                start_predictions, state, self.take_step)

        output_dict = {
                "class_log_probabilities": log_probabilities,
                "predictions": all_top_k_predictions,
        }
        return output_dict

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
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

        decoder_hidden_and_context = (decoder_hidden, decoder_context) \
                                        if decoder_hidden is not None or \
                                           decoder_context is not None else \
                                     None

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        if self._attention:

            # shape: (group_size, max_input_sequence_length, encoder_output_dim)
            encoder_outputs = state["encoder_outputs"]

            # shape: (group_size, max_input_sequence_length)
            source_mask = state["source_mask"]

            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)

            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = embedded_input
        
        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(decoder_input, 
                                                             decoder_hidden_and_context)

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # add dropout
        decoder_hidden_with_dropout = self._dropout(decoder_hidden)
        
        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_hidden_with_dropout)

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
        input_weights = self._attention(
                decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

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
            all_metrics.update({'perplexity': self._perplexity.get_metric(reset=reset)})
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))

        return all_metrics