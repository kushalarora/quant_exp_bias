
# coding: utf-8

import os
import sys
import wandb

from matplotlib import pyplot as plt
from datetime import datetime

from datetime import datetime
from typing import Dict, List

from allennlp.common import Params
from allennlp.common.util import import_submodules

import_submodules("quant_exp_bias")
from quant_exp_bias.utils import (get_args, quantify_exposure_bias_runner, 
                  sample_oracle_runner, train_runner)
from quant_exp_bias.oracles.artificial_grammar_oracle import ArtificialLanguageOracle
from experiments.util import initialize_experiments, one_exp_run

import itertools
import json

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id = initialize_experiments('vocabulary_size_experiments')
vocabulary_sizes  = [6, 12, 24, 48, 90]
vocab_distributions = ['zipf', 'uniform']
grammar_templates = ['grammar_templates/grammar_2.template', 'grammar_templates/grammar_1.template']

num_samples_and_runs = [(1000, 8), (10000,4), (100000,2)]

# # Validation Experiments
grammar_vocab_size_and_dist = [x for x in itertools.product(grammar_templates, vocabulary_sizes, vocab_distributions)]

def vocabulary_size_experiments(grammar_vocab_size_and_dist, 
                                num_samples_and_runs, 
                                main_args, 
                                serialization_dir, 
                                param_path):
    # Setup variables needed later.
    orig_serialization_dir = serialization_dir
    for grammar_template, size, dist in grammar_vocab_size_and_dist:
        vsexp_serialization_dir = os.path.join(orig_serialization_dir, f'{grammar_template}_{dist}_{size}')
        grammar_string = ArtificialLanguageOracle.generate_grammar_string(grammar_template_file=grammar_template,
                                                                            vocabulary_size=size, 
                                                                            vocabulary_distribution=dist)
        os.makedirs(vsexp_serialization_dir, exist_ok=True)
        grammar_filename = os.path.join(vsexp_serialization_dir, 'grammar.txt')
        with open(grammar_filename, 'w') as f:
            f.write(grammar_string)
        
        overrides = json.dumps({'model':{
                                    'oracle': {
                                        'grammar_file': grammar_filename}}})
        for num_samples, num_runs in num_samples_and_runs:
            for num_run in range(num_runs):
                run_metrics = one_exp_run(serialization_dir=vsexp_serialization_dir, 
                                        num_samples=num_samples,
                                        run=num_run, 
                                        param_path=param_path,
                                        overides_func=lambda: overrides)
        
                assert len(run_metrics) == 1, \
                    'For this experiment, there should only be one final metric object for a run.'
                run_metrics = run_metrics[0]
                for exp_bias_idx, exp_bias in enumerate(run_metrics['exp_biases']):
                    result= {
                            'exp_bias': exp_bias, 
                            'exp_bias_idx': exp_bias_idx,
                            'num_run': num_run,
                            'num_samples': num_samples,
                            'distribution': dist,
                            'vocab_size': size,
                            'val_ppl': run_metrics['best_validation_perplexity'],
                            'best_val_epoch': run_metrics['best_epoch'],
                            'grammar': grammar_template
                        }
                wandb.log(result)

vocabulary_size_experiments(grammar_vocab_size_and_dist, 
                            num_samples_and_runs, 
                            main_args, 
                            serialization_dir, 
                            param_path)
