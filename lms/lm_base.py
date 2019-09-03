import math
import time
import torch
import torch.nn.functional as F

from allennlp.common.util import START_SYMBOL, END_SYMBOL, gpu_memory_mb, peak_memory_mb
from allennlp.training import util as allennlp_training_util
from allennlp.nn import util as allennlp_nn_util
from typing import Dict, List, Any, Iterable, Optional
from torch import nn
from torch.nn.modules.rnn import LSTMCell


from .data import Dictionary


class LMBase():
    """
        Base Class for LM models to be evaluated using
        Exposure Bias. The LM models are passed as a
        model object and are trained, sampled from and
        evaluated in this base class.

        Parameters:
        ----------
        model_name: ``str``
            Type of the model to use. LSTM or Transformer.
        vocab: ``Dictionary``
            The dictionary object mapping idx2word and word2idx.
        optimizer: ``torch.nn.Optimizer``
            The optimizer used to train language model.
        config: ``Dict[str, Any]``
            This is the arg object from arg parser. This
            is passed to all the subclasses and model specific
            values are extracted from it.
    """

    def __init__(self,
                 model_name: str,
                 vocab: Dictionary,
                 optimizer: torch.optim.Optimizer,
                 config: Dict[str, Any]):
        self.model_name = model_name
        self.vocab = vocab
        self.optimizer = optimizer

        self.generation_sampling_temprature = config.generation_sampling_temprature
        self.gradient_clip = config.gradient_clip
        self.log_interval = config.log_interval
        self.num_epochs = config.epochs
        self.batch_size = config.batch_size
        self.val_batch_size = config.val_batch_size
        self.lr = config.lr
        self.device = torch.device("cuda" if config.cuda else "cpu")
        self.save = config.save
        self.max_len = config.max_len

        self.pad_idx = config.pad_idx

        self.criterion = nn.CrossEntropyLoss()

        # TODO(@karora) Maybe move this out later sometime if
        # you add a Transformer based model.
        self.ntoken = len(vocab)
        self.ninp = config.ninp
        self.nhid = config.nhid
        self.tie_weights = config.tie_weights

        self.encoder = nn.Embedding(ntoken, ninp)

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        self.lstm_cell = LSTMCell(self.ninp, self.nhid)
        self.decoder = nn.Linear(self.nhid, self.ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if self.tie_weights:
            if nhid != ninp:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    @classmethod
    def build_model(cls,
                    model_name: str,
                    vocab_size: int,
                    device: torch.device,
                    config: Dict[str, Any]):
        raise NotImplementedError

    def init_model_epoch(self):
        """ TODO: Add docstring
        """
        return NotImplementedError

    def train(self,
              train_data: Iterable[torch.LongTensor],
              validation_data: Iterable[torch.LongTensor]):
        """ TODO: Add docstring
        """

        allennlp_training_util.enable_gradient_clipping(
            self.model, self._grad_clipping)

        best_val_loss = None
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()

            self.train_epoch(epoch, train_data)

            val_loss, _ = self._forward_epoch(epoch,
                                              self.val_batch_size,
                                              validation_data,
                                              training=False)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            peak_cpu_usage = peak_memory_mb()
            print(f"Peak CPU memory usage MB: {peak_cpu_usage}")
            gpu_usage = []
            for gpu, memory in gpu_memory_mb().items():
                gpu_usage.append((gpu, memory))
                print(f"GPU {gpu} memory usage MB: {memory}")
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(self.save, 'wb') as f:
                    torch.save(self.model, f)
                best_val_loss = val_loss

    def get_batch(self,
                  data: Iterable[torch.LongTensor]):
        """ This formats lm data into source and target. It
            assumes the each sentence is starts with START_SYMBOL and
            ends with END_SYMBOL.
        """
        source = data[:, :-2]
        target = data[:, 1:]
        target_mask = 1 - (target == self.pad_idx)
        return source, target, target_mask

    def init_model_epoch(self):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        weight = next(self.parameters())
        return (weight.new_zeros(self.batch_size, self.nhid),
                weight.new_zeros(self.batch_size, self.nhid))

    def train_epoch(self,
                    epoch: int,
                    train_data: Iterable[torch.LongTensor]):
        """ TODO: Add docstring
        """
        # Turn on training mode which enables dropout.
        self.model.train()

        self._forward_epoch(epoch,
                            self.batch_size,
                            train_data,
                            training=True)

    def _forward_epoch(self,
                       epoch: int,
                       batch_size: int,
                       data_iterable: Iterable[torch.LongTensor],
                       training: Optional[bool] = True,
                       compute_seq_prob: Optional[bool] = False):

        # For LSTM Based model, get initial hidden state.
        hidden, context = self.init_model_epoch(batch_size)

        total_loss = 0.
        start_time = time.time()
        epoch_loss: float = 0.
        batch_seq_probs: List[torch.Tensor] = []
        for batch_idx, batch in enumerate(data_iterable):
            source, targets, target_mask = self.get_batch(batch)
            self.optimizer.zero_grad()

            _, target_sequence_length = targets.size()

            last_predictions = source[:, 0]

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length

            step_logits: List[torch.Tensor] = []
            step_predictions: List[torch.Tensor] = []
            for timestep in range(num_decoding_steps):

                # TODO (karora): Figure out how to handle this and differentiable
                #                scheduled sampling at the same time.
                if training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                    # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                    # during training.
                    # shape: (batch_size,)
                    input_choices = last_predictions
                else:
                    # shape: (batch_size,)
                    input_choices = source[:, timestep]

                # shape: (batch_size, num_classes)
                embedded_input = self.drop(self.encoder(input_choices))

                hidden, context = self.lstm_cell(embedded_input,
                                                 (hidden, context))

                output = self.decoder(hidden)

                # list of tensors, shape: (batch_size, 1, num_classes)
                step_logits.append(output.unsqueeze(1))

                # shape: (batch_size, num_classes)
                class_probabilities = F.softmax(output, dim=-1)

                # shape (predicted_classes): (batch_size,)
                _, predicted_classes = torch.max(class_probabilities, 1)

                # shape (predicted_classes): (batch_size,)
                last_predictions = predicted_classes

                step_predictions.append(last_predictions.unsqueeze(1))

            # shape: (batch_size, num_decoding_steps)
            predictions = torch.cat(step_predictions, 1)

            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            if compute_seq_prob:
                # shape : (batch * sequence_length, num_classes)
                logits_flat = logits.view(-1, logits.size(-1))
                # shape : (batch * sequence_length, num_classes)
                log_probs_flat = F.log_softmax(logits_flat, dim=-1)

                # shape : (batch * max_len, 1)
                targets_flat = targets.view(-1, 1).long()

                # shape : (batch * sequence_length, 1)
                ll_flat = torch.gather(log_probs_flat, dim=1,
                                       index=targets_flat)

                # shape : (batch, sequence_length)
                ll_batch = ll_flat.view(*targets.size())
                # shape : (batch, sequence_length)
                ll_batch = ll_batch * target_mask

                # sum all dim except batch
                non_batch_dims = tuple(range(1, len(target_mask.shape)))

                seq_probs = torch.exp(ll_batch.sum(non_batch_dims))

                batch_seq_probs.append(seq_probs)

            # Compute loss.
            loss = allennlp_nn_util.sequence_cross_entropy_with_logits(logits,
                                                                       targets,
                                                                       target_mask)
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            if training:
                loss.backward()

                self.optimizer.step()

            total_loss += loss.item()
            epoch_loss += loss.item()
            if batch % self.log_interval == 0 and batch > 0:
                cur_loss = total_loss / self.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch_idx, self.batch_size,
                                                          elapsed * 1000 / self.log_interval, cur_loss, math.exp(cur_loss)))

                total_loss = 0
                start_time = time.time()
            return epoch_loss, batch_seq_probs

    def compute_sent_probs(self,
                           batched_sentences: Iterable[torch.LongTensor]):
        """ TODO: Add docstring
        """
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        _, batch_seq_probs = self._forward_epoch(-1,  # epoch
                                                 self.batch_size,
                                                 batched_sentences,
                                                 training=False,
                                                 compute_seq_probs=True)

        return batch_seq_probs

    def sample_test_set(self,
                        corpus_size: int):
        """ TODO: Add docstring
        """
        self.model.eval()

        ntokens = len(self.vocab)
        inp = torch.randint(ntokens, (1, 1), dtype=torch.long).to(self.device)

        with torch.no_grad():  # no tracking history
            for j in range(corpus_size):
                # TODO: Sample n from \mathcal{N}.
                n = 0
                words = []
                for i in range(n):
                    if False:
                        output = self.model(inp, False)
                        word_weights = output[-1].squeeze().div(
                            self.generation_sampling_temprature).exp().cpu()
                        word_idx = torch.multinomial(word_weights, 1)[0]
                        word_tensor = torch.Tensor(
                            [[word_idx]]).long().to(self.device)
                        inp = torch.cat([input, word_tensor], 0)
                    else:
                        output, hidden = self.model(input, hidden)
                        word_weights = output.squeeze().div(
                            self.generation_sampling_temprature).exp().cpu()
                        word_idx = torch.multinomial(word_weights, 1)[0]
                        inp.fill_(word_idx)

                    words.append(self.vocab.idx2word[word_idx])

                if j % self.log_interval == 0:
                    print('| Generated {:d} sequence.  {}'.format(j, words))
