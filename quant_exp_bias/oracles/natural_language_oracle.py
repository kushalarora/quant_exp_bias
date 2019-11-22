import math
import torch
import logging

from typing import List
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

from quant_exp_bias.oracles.oracle_base import Oracle
from multiprocessing import Pool

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Oracle.register('gpt2_oracle')
class NaturalLanguageOracle(Oracle):

    def __init__(self, model_name="gpt2", 
                       parallelize=True,
                       num_threads=128):
        super(Oracle, self).__init__()
        self._parallelize = parallelize

        self._num_threads = num_threads
        self._pool = Pool(self._num_threads)

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Load pre-trained model (weights)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        self.model.eval()

    def sample_training_set(self, num_samples: int):
        """
        TODO: sample subset of sentences from the data used for training GPT-2
        """
        pass

    def compute_sent_probs(self, sequences: List[List[str]]):
        # TODO (Kushal): Try to figure out how to do this efficiently
        # by batching the inputs.
        probs = []
        for i, sequence in enumerate(sequences):
            tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(sequence)])
            with torch.no_grad():
                loss = -1 * self.model(tensor_input, labels=tensor_input)[0]
            probs.append(math.exp(loss.item()))  # perplexity
        return probs