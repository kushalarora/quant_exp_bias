from typing import List

from quant_exp_bias.oracles.oracle_base import Oracle
from nltk import PCFG
from nltk.grammar import Nonterminal
from nltk.parse.pchart import InsideChartParser

from scipy.stats import zipf

from functools import reduce
import itertools
import random
import ray
import re
import subprocess
import time
import string
import numpy as np

@Oracle.register('artificial_lang_oracle')
class ArtificialLanguageOracle(Oracle):
    """
    TODO (Kushal): Expand class doc.
    SO: https://stackoverflow.com/questions/15009656/how-to-use-nltk-to-generate-sentences-from-an-induced-grammar
    """

    def __init__(self,
                 grammar_file:str,
                 use_weighted_choice: bool = True,
                 parallelize=True, 
                 num_threads=32):
        """ TODO (Kushal): Add function doc.
        """
        super(Oracle, self).__init__()

        self._grammar_string = open(grammar_file).read()
        self._use_weighted_choice = use_weighted_choice

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
    def generate_grammar_string(grammar_template_file: str, 
                                 vocabulary_size: int,
                                 vocabulary_distribution: str,):
        epsilon = 10**-10

        def _get_vocab_prob(vsize, offset=0):
            if vocabulary_distribution == 'zipf':
                dist = zipf(1.2)
                p_vocabs = [dist.pmf(x + 1) - offset/vsize for x in range(vsize)]
                p_vocabs /= sum(p_vocabs)
            elif vocabulary_distribution == 'uniform':
                p_vocab = [1.0/vsize - offset/vsize] * vsize
            return p_vocab

        printables = [f"'{x}'" for x in string.printable[:-7] if x not in set(["'", '"'])]
        assert vocabulary_size <= len(printables)
        vocab = [printables[i] for i in range(vocabulary_size)]
        extended_vocab = ["'SOS'", "'EOS'"] + vocab
        
        grammar_template = open(grammar_template_file)
        grammar_rules = []
        
        group_set = set([])
        for template in grammar_template:
            states_and_inputs = template.strip().split()
            current_state, arrow, next_states = states_and_inputs[0], states_and_inputs[1], states_and_inputs[2:]
            if len(next_states) == 2:
                inp, next_state = next_states
            elif len(next_states) == 3:
                inp, next_state, prob = next_states

            if re.match("'<G[0-9]+>'", inp):
                group_set.add(inp)

        group2idx = {}
        for i, g in enumerate(group_set):
            group2idx[g] = i
        
        num_groups = len(group_set)  
        group_vocab_size = vocabulary_size//num_groups

        current_state_offset = {}
        grammar_template.seek(0)
        for template in grammar_template:   
            token2p = {} 
            states_and_inputs = template.strip().split()
            current_state, arrow, next_states = states_and_inputs[0], states_and_inputs[1], states_and_inputs[2:]
            
            if current_state not in current_state_offset:
                current_state_offset[current_state] = 0
            
            offset = current_state_offset[current_state]
            prob = None
            if len(next_states) == 2:
                inp, next_state = next_states
            elif len(next_states) == 3:
                inp, next_state, prob = next_states
                prob = float(prob)

            if inp == "'EOS'":
                token2p[inp] = (prob or 1.0 - offset)  - epsilon * (vocabulary_size + 2) 

                for token in extended_vocab:
                    if token in token2p:
                        p = token2p[token]
                    else:
                        p = epsilon

                    current_state_offset[current_state] += p
                    grammar_rules.append(f"{current_state} {arrow} {token} [{p:.10f}]")
            else:
                if re.match("'<G[0-9]+>'", inp):
                    group_num = group2idx[inp]
                    group_vocab = vocab[group_num * group_vocab_size : (group_num + 1) * group_vocab_size]
                    group_p_vocab = _get_vocab_prob(len(group_vocab), offset)
                    if prob: 
                        group_p_vocab =  [prob * x for x in group_p_vocab]

                    for token, p in zip(group_vocab, group_p_vocab):
                        token2p[token] = p - epsilon * (vocabulary_size + 2)/len(group_p_vocab)
                else: 
                   token2p[inp] = (prob or 1.0  - offset) - epsilon * (vocabulary_size + 2)

                for token in extended_vocab:
                    if token in token2p:
                        p = token2p[token]
                    else:
                        p = epsilon

                    current_state_offset[current_state] += p
                    grammar_rules.append(f"{current_state} {arrow} {token} {next_state} [{p:.10f}]")

        grammar_string = ""
        for rule in grammar_rules:
            grammar_string += f"{rule}\n"
        return grammar_string
        
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
            probs = 1e-10
            try:
                parses = list(parser.parse(sequence))
                if parses and len(parses) > 0:
                    probs += np.exp(np.log(reduce(lambda a, b: a + b.prob(), parses, 0)/len(parses))/len(sequence))
            except Exception as e:
                pass

            return probs