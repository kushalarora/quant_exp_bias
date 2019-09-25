from quant_exp_bias.oracles.oracle_base import Oracle
from nltk import PCFG
from nltk.grammar import Nonterminal
from nltk.parse.pchart import InsideChartParser
from functools import reduce
import itertools
import random


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
                 num_samples : int,
                 grammar_string: str=FSA_GRAMMAR_STRING,
                 use_weighted_choice: bool = True):
        """ TODO (Kushal): Add function doc.
        """
        super(Oracle, self).__init__()
        self._grammar = PCFG.fromstring(grammar_string)
        self._parser = InsideChartParser(self._grammar)
        self._use_weighted_choice = use_weighted_choice
        self._num_samples = num_samples

    @classmethod
    def _weighted_choice(cls, productions):
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

    @classmethod
    def _rewrite_at(cls, index, replacements, the_list):
        """ TODO (Kushal): Add function doc.
        """
        del the_list[index]
        the_list[index:index] = replacements

    def _generate_sequence(self):
        """ TODO (Kushal): Add function doc.
        """
        sentence_list = [self._grammar.start()]
        all_terminals = False
        choice = self._weighted_choice if self._use_weighted_choice else random.choice
        while not all_terminals:
            all_terminals = True
            for position, symbol in enumerate(sentence_list):
                if symbol in self._grammar._lhs_index:
                    all_terminals = False
                    derivations = self._grammar._lhs_index[symbol]
                    derivation = choice(derivations)
                    self._rewrite_at(position, derivation.rhs(), sentence_list)
        return ' '.join(sentence_list)

    def sample_training_set(self):
        """ TODO (Kushal): Add function doc.
        """
        # TODO (Kushal): Reformat the code to move generator to the base class and derived class only overloads generate_sequence method.
        return [self._generate_sequence() for _ in range(self._num_samples)]

    def compute_sent_probs(self, sequences):
        """ TODO (Kushal): Add function doc.
        """
        # TODO (Kushal): Reformat the code to move the for loop in the base class.
        sent_probs = []
        for sequence in sequences:
            sequence = ['S'] + sequence + ['E']
            probs = 1e-10
            try:
                parses = list(self._parser.parse(sequence))
                if parses and len(parses) > 0:
                    probs += reduce(lambda a, b: a + b.prob(), parses, 0) / len(parses)/ len(sequence)
            except Exception as e:
                pass
            
            sent_probs.append(probs)
        return sent_probs
