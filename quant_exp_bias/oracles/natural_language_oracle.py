import math
import torch
import logging

from pytorch_transformers import OpenAIGPTTokenizer, GPT2Tokenizer
from pytorch_transformers import OpenAIGPTLMHeadModel, GPT2LMHeadModel

from quant_exp_bias.oracles.oracle_base import Oracle

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Oracle.register('natural_lang_oracle')
class NaturalLanguageOracle(Oracle):

    def __init__(self, lm):
        super(Oracle, self).__init__()
        assert lm in ['gpt', 'gpt2']

        if lm == 'gpt':
            # Load pre-trained model tokenizer (vocabulary)
            self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
            # Load pre-trained model (weights)
            self.model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')

        else:
            # Load pre-trained model (weights)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

            # Load pre-trained model tokenizer (vocabulary)
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')

        self.model.eval()

    def sample_training_set(self, num_samples: int):
        """
        TODO: sample subset of sentences from the data used for training GPT-2
        """
        pass

    def _compute_one_sent_prob(self, sequence):

        tokenize_input = self.tokenizer.tokenize(sequence)
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])

        with torch.no_grad():
            loss = self.model(tensor_input, labels=tensor_input)

        return math.exp(loss.item())  # perplexity
