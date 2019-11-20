import math
import torch
import logging

from typing import List
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

from quant_exp_bias.oracles.oracle_base import Oracle
from multiprocessing import Pool

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Oracle.register('natural_lang_oracle')
class NaturalLanguageOracle(Oracle):

    def __init__(self, parallelize=True, num_threads=128):
        super(Oracle, self).__init__()

        self._parallelize = parallelize

        self._num_threads = num_threads
        self._pool = Pool(self._num_threads)

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Load pre-trained model (weights)
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

        self.model.eval()

    def sample_training_set(self, num_samples: int):
        """
        TODO: sample subset of sentences from the data used for training GPT-2
        """
        pass

    def compute_sent_probs(self, sequences: List[List[str]]):

        return self._pool.starmap(self.model._compute_one_sent_prob, sequences)

    def _compute_one_sent_prob(self, sequence):

        tokenize_input = self.tokenizer.tokenize(sequence)
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])

        with torch.no_grad():
            loss = self.model(tensor_input, labels=tensor_input)

        return math.exp(loss.item())  # perplexity
