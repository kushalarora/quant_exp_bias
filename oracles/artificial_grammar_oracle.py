from .oracle_base import OracleBase
from nltk import PCFG
from nltk.grammar import Nonterminal
from nltk.parse.pchart import InsideChartParser
from functools import reduce
import itertools
import random

class ArtificialLanguageOracle(OracleBase):
    """ 
    SO: https://stackoverflow.com/questions/15009656/how-to-use-nltk-to-generate-sentences-from-an-induced-grammar
    """
    FSA_GRAMMAR_STRING = """
                            q0 -> '<s>' q1 [1]
                            q1 -> 'a' q1 [0.3]
                            q1 -> 'b' q1 [0.3]
                            q1 -> 'c' q1 [0.3]
                            q1 -> '</s>' [0.1]
                         """
    def __init__(self, 
                 grammar_string:str=FSA_GRAMMAR_STRING):
        """ TODO: Add function doc.
        """
        OracleBase.__init__(self)
        self.grammar = PCFG.fromstring(grammar_string)
        self.parser = InsideChartParser(self.grammar)
    
    @classmethod
    def _weighted_choice(cls, productions):
               """ TODO: Add function doc.
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
        """ TODO: Add function doc.
        """
        del the_list[index]
        the_list[index:index] = replacements

    @classmethod
    def _generate_sequence(cls, grammar, use_weighted_choice=True):
        """ TODO: Add function doc.
        """
        sentence_list = [grammar.start()]
        all_terminals = False
        choice = cls._weighted_choice if use_weighted_choice else random.choice
        while not all_terminals:
            all_terminals = True
            for position, symbol in enumerate(sentence_list):
                if symbol in grammar._lhs_index:
                    all_terminals = False
                    derivations = grammar._lhs_index[symbol]
                    derivation = choice(derivations)  
                    cls._rewrite_at(position, derivation.rhs(), sentence_list)
        return sentence_list

    def sample_training_set(self, num_samples, use_weighted_choice=True):
        """ TODO: Add function doc.
        """
        # TODO: Reformat the code to move generator to the base class and derived class only overloads generate_sequence method.

        return [self._generate_sequence(self.grammar, use_weighted_choice) \
                    for _ in range(num_samples)]
    
    def compute_sent_probs(self, sequences):
        """ TODO: Add function doc.
        """
        # TODO: Reformat the code to move the for loop in the base class.
        for sequence in sequences:
            parses = list(self.parser.parse(sequence.split()))
            return reduce(lambda a,b: a + b.prob(), parses, 0)/ len(parses) \
                                    if parses else 0

