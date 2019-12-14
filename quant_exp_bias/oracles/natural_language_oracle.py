import logging
import math
import torch
import torch.nn.functional as F

from typing import List
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from quant_exp_bias.oracles.oracle_base import Oracle
from multiprocessing import Pool

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Oracle.register('gpt2_oracle')
class NaturalLanguageOracle(Oracle):

    def __init__(self, model_name="gpt2", 
                       parallelize=True,
                       num_threads=128, 
                       cuda_device=-2,
                       batch_size=10):
        super(Oracle, self).__init__()
        # self._parallelize = parallelize

        self._num_threads = num_threads
        # self._pool = Pool(self._num_threads)

        self.device = "cpu"
        if cuda_device > 0:
            self.device = "cuda:{cuda_device}"
        elif cuda_device == -2:
            self.device = torch.cuda.current_device() 

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Load pre-trained model (weights)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)

        self.batch_size = batch_size
        self.model.eval()

    def sample_training_set(self, num_samples: int):
        """
        TODO: sample subset of sentences from the data used for training GPT-2
        """
        pass

    def compute_sent_probs(self, sequences: List[List[str]]):
        # TODO (Kushal): Try to figure out how to do this efficiently
        # by batching the inputs.
        seq_batch_size = len(sequences)
        probs = []

        for i in range(0, seq_batch_size, self.batch_size):
            batch = sequences[i:i + self.batch_size] if i + self.batch_size < seq_batch_size else sequences[i:seq_batch_size]
            bsize = self.batch_size if i + self.batch_size < len(sequences) else seq_batch_size - i

            max_len = max([len(sequence) for sequence in batch])
            ids = [self.tokenizer.convert_tokens_to_ids(sequence) + [self.tokenizer.eos_token_id] * (max_len - len(sequence)) for sequence in batch]
            tensor_input = torch.tensor(ids).to(self.device)
            attention_mask = (tensor_input != self.tokenizer.eos_token_id).float().to(self.device)

            with torch.no_grad():
                results =  self.model(tensor_input, labels=tensor_input, attention_mask=attention_mask)
                logits = results[1]
                labels = tensor_input

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_batch_seq = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                                    shift_labels.view(-1),
                                                    ignore_index = -1, reduction='none').view(bsize, -1)

                loss_batch_seq *=attention_mask[:, 1:]
                seq_sizes = attention_mask.sum(dim=-1)

                loss_batch = loss_batch_seq.sum(dim=-1)/seq_sizes

                for j in range(self.batch_size):
                    probs.append(math.exp(-1 * loss_batch[j].item()))

        # for i, sequence in enumerate(sequences):
        #     tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(sequence)])
        #     with torch.no_grad():
        #         loss = -1 * self.model(tensor_input, labels=tensor_input)[0]
        #     probs.append(math.exp(loss.item()))  # perplexity
        return probs