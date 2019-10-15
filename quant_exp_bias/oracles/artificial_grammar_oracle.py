from typing import List

from quant_exp_bias.oracles.oracle_base import Oracle
from nltk import PCFG
from nltk.grammar import Nonterminal
from nltk.parse.pchart import InsideChartParser
from functools import reduce
import itertools
import random
import ray
import subprocess
import time

@Oracle.register('artificial_lang_oracle')
class ArtificialLanguageOracle(Oracle):
    """
    TODO (Kushal): Expand class doc.
    SO: https://stackoverflow.com/questions/15009656/how-to-use-nltk-to-generate-sentences-from-an-induced-grammar
    """
    FSA_GRAMMAR_STRING = """
                            q0 -> 'S' q1 [0.9900] | 'a' q1 [0.0025] | 'b' q1 [0.0025] | 'c' q1 [0.0025] | 'E' q1 [0.0025]
                            q1 -> 'S' q1 [0.0025] | 'a' q1 [0.3000] | 'b' q1 [0.3000] | 'c' q1 [0.3000] | 'E' q1 [0.0025]
                            q1 -> 'S' q2 [0.0025] | 'a' q2 [0.0300] | 'b' q2 [0.0300] | 'c' q2 [0.0300] | 'E' q2 [0.0025]
                            q2 -> 'S' [0.0025] | 'a' [0.0025] | 'b' [0.0025] | 'c' [0.0025] | 'E' [0.9900]

                         """

    def __init__(self,
                 grammar_string: str,
                 use_weighted_choice: bool = True,
                 parallelize=True, 
                 num_threads=32):
        """ TODO (Kushal): Add function doc.
        """
        super(Oracle, self).__init__()
        self._use_weighted_choice = use_weighted_choice
        self._grammar_string = grammar_string
        self._parallelize = parallelize

        if ray.is_initialized():
            # Just to avoid weird ray behaviors
            # we will shut down the server and will
            # restart it.
            ray.shutdown()
            print("$$$$ Shutting Down Ray $$$$$")

            # Sleep for 5 secs to make sure server
            # is properly shutdown.
            time.sleep(2)

        ray.init(num_cpus=num_threads)

        # Sleep for 5 secs to make sure server
        # is properly up.
        time.sleep(2)
        print("$$$$ Ray Initialized $$$$$")




    @staticmethod
    def _weighted_choice(productions):
        """ TODO (Kushal): Add function doc.
        """
        prods_with_probs = [(prod, prod.prob()) for prod in productions]
        total = sum(prob for prod, prob in prods_with_probs)
        r = random.uniform(0, total)
        upto = 0
        for prod, prob in prods_with_probs:
            if upto + prob >= r:
                return prod
            upto += prob
        assert False, "Shouldn't get here"

    @staticmethod
    def _rewrite_at(index, replacements, the_list):
        """ TODO (Kushal): Add function doc.
        """
        del the_list[index]
        the_list[index:index] = replacements

    @staticmethod
    @ray.remote
    def generate_sequence(grammar_string, use_weighted_choice):
        """ TODO (Kushal): Add function doc.
        """
        grammar = PCFG.fromstring(grammar_string)
        sentence_list = [grammar.start()]
        all_terminals = False
        choice = ArtificialLanguageOracle._weighted_choice if use_weighted_choice else random.choice
        while not all_terminals:
            all_terminals = True
            for position, symbol in enumerate(sentence_list):
                if symbol in grammar._lhs_index:
                    all_terminals = False
                    derivations = grammar._lhs_index[symbol]
                    derivation = choice(derivations)
                    ArtificialLanguageOracle._rewrite_at(position, derivation.rhs(), sentence_list)
        return ' '.join(sentence_list)

    def sample_training_set(self, num_samples: int):
        """ TODO (Kushal): Add function doc.
        """
        # TODO (Kushal): Reformat the code to move generator to the base class and derived class only overloads generate_sequence method.
        samples = ray.get([ArtificialLanguageOracle.generate_sequence.remote(self._grammar_string, self._use_weighted_choice)  for _ in range(num_samples)])
        return samples

    def compute_sent_probs(self, sequences: List[List[str]]):
        """ TODO (Kushal): Add function doc.
        """
        # TODO (Kushal): Reformat the code to move the for loop in the base class.
        probs = ray.get([ArtificialLanguageOracle._compute_one_sent_prob.remote(self._grammar_string, sequence) for sequence in sequences])
        return probs 

    @staticmethod
    @ray.remote
    def _compute_one_sent_prob(grammar_string, sequence: List[str]):
            parser = InsideChartParser(PCFG.fromstring(grammar_string))
            probs = 1e-30
            try:
                parses = list(parser.parse(sequence))
                if parses and len(parses) > 0:
                    probs += reduce(lambda a, b: a + b.prob(), parses, 0) / len(parses)
            except Exception as e:
                pass

            return probs