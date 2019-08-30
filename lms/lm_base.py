import math
import time
import torch

from typing import Dict, List, Any
from torch import nn
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
        self.bptt = config.bptt
        self.gradient_clip = config.gradient_clip
        self.log_interval = config.log_interval
        self.num_epochs = config.epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.device = torch.device("cuda" if config.cuda else "cpu")
        self.save = config.save

        self.criterion = nn.CrossEntropyLoss()
        # This will be bound to subclass's build_model
        # method which will be implemented in the concrete
        # subclass
        self.model = LMBase.build_model(model_name,
                                        len(vocab),
                                        self.device,
                                        config)

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
              train_data,
              validation_data):
        """ TODO: Add docstring
        """
        best_val_loss = None
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
            self.train_epoch(epoch, train_data)
            val_loss = torch.sum(self.evaluate(validation_data)) / \
                (len(validation_data) - 1)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(self.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0

    def get_batch(self, source, i):
        """ TODO: Add docstring
        """
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

    def train_epoch(self,
                    epoch: int,
                    train_data):
        """ TODO: Add docstring
        """
        # Turn on training mode which enables dropout.
        self.model.train()

        total_loss = 0.
        start_time = time.time()
        ntokens = len(self.vocab)

        # Figure out how the batch size will be
        # obtained.
        self.init_model_epoch(self.batch_size)

        for batch, i in enumerate(range(0, train_data.size(0) - 1, self.bptt)):
            data, targets = self.get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.model.zero_grad()

            output, hidden = self.model(data, hidden)
            loss = self.criterion(output.view(-1, ntokens), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip)
            for p in self.model.parameters():
                p.data.add_(-self.lr, p.grad.data)

            total_loss += loss.item()

            if batch % self.log_interval == 0 and batch > 0:
                cur_loss = total_loss / self.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                          epoch, batch, len(train_data) // self.bptt, self.lr,
                          elapsed * 1000 / self.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def compute_sent_probs(self,
                           sentences: List):
        """ TODO: Add docstring
        """
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        log_probs = []
        ntokens = len(self.vocab)
        self.init_model_epoch(self.batch_size)

        with torch.no_grad():
            for i in range(0, sentences.size(0) - 1, self.bptt):
                data, targets = self.get_batch(sentences, i)
                output, hidden = self.model(data, hidden)
                output_flat = output.view(-1, ntokens)
                log_probs.append(
                    len(data) * self.criterion(output_flat, targets).item())
        return log_probs

    def sample_test_set(self, corpus_size):
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
