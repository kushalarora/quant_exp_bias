
# coding: utf-8

import os
import sys
import wandb

from matplotlib import pyplot as plt
from datetime import datetime

from datetime import datetime

from random import randint
from time import sleep

from typing import Dict, List

from allennlp.common import Params
from allennlp.common.util import import_submodules

import_submodules("quant_exp_bias")
from quant_exp_bias.utils import (get_args, quantify_exposure_bias_runner,
                  sample_oracle_runner, train_runner)
from quant_exp_bias.oracles.artificial_grammar_oracle import ArtificialLanguageOracle
from experiments.util import initialize_experiments, generate_grammar_file, one_exp_run

import itertools
import json

import argparse
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--num_samples', type=int, default=10000,
                    help='Number of dataset samples to run this iteration for.')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs for the given dataset size.')
parser.add_argument('--all', action='store_true', help='Run All configurations mentioned below..')
parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
args = parser.parse_args()

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id, experiment = initialize_experiments('artificial_lang/vocabulary_size_experiments', debug=args.debug)
vocabulary_sizes  = [6, 12, 24, 48]
vocab_distributions = ['zipf', 'uniform']
grammar_templates = ['grammar_templates/grammar_2.template', 'grammar_templates/grammar_1.template']

num_samples_and_runs = [(1000, 4), (10000,3), (100000,2)]
# num_samples_and_runs = [(1000, 1), (10000,1), (100000,1)]

# # Validation Experiments
grammar_vocab_size_and_dist = [x for x in itertools.product(grammar_templates, vocabulary_sizes, vocab_distributions)]

experiment.log_parameters({'serialization_dir': serialization_dir,
                          'main_args': main_args,
                          'param_path': param_path,
                          'experiment_id': experiment_id})

def vocabulary_size_experiments(grammar_vocab_size_and_dist,
                                main_args,
                                serialization_dir,
                                param_path,
                                num_samples,
                                num_runs,
                               ):
    # Setup variables needed later.
    step = 0
    orig_serialization_dir = serialization_dir
    for grammar_template, size, dist in grammar_vocab_size_and_dist:
        vsexp_serialization_dir = os.path.join(orig_serialization_dir,  f'{grammar_template}_{dist}_{size}')
        grammar_filename = generate_grammar_file(vsexp_serialization_dir,
                                                 grammar_template=grammar_template,
                                                 vocabulary_size=size,
                                                 vocabulary_distribution=dist)


        overrides = json.dumps({'model':{
                                    'decoder': {
                                        'oracle': {
                                            'grammar_file': grammar_filename}}}})
        for num_run in range(num_runs):
            run_metrics = one_exp_run(serialization_dir=vsexp_serialization_dir,
                                    num_samples=num_samples,
                                    run=num_run,
                                    param_path=param_path,
                                    overides_func=lambda: overrides)

            assert len(run_metrics) == 1, \
                'For this experiment, there should only be one final metric object for a run.'
            run_metrics = run_metrics[0]
            for exp_bias_idx, (exp_bias, df_p_q, df_q_p) in enumerate(zip(run_metrics['exp_biases'],
                                                                        run_metrics['df_p_qs'],
                                                                        run_metrics['df_q_ps'])):                
                result= {
                        'exp_bias': exp_bias,
                        'Df_p_q': df_p_q,
                        'Df_q_p': df_q_p,
                        'exp_bias_idx': exp_bias_idx,
                        'num_run': num_run,
                        'num_samples': num_samples,
                        'distribution': dist,
                        'vocab_size': size,
                        'val_ppl': run_metrics['best_validation_perplexity'],
                        'best_val_epoch': run_metrics['best_epoch'],
                        'grammar': grammar_template
                    }
                experiment.log_metrics(result, step=step)
                step += 1
                sleep(randint(1,10))

            experiment.log_metric('exp_bias_mean', run_metrics['exp_bias_mean'], step=step)
            experiment.log_metric('df_p_q_mean', run_metrics['df_p_q_mean'], step=step)
            experiment.log_metric('df_q_p_mean', run_metrics['df_q_p_mean'], step=step)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:

        vocabulary_size_experiments(grammar_vocab_size_and_dist,
                                main_args,
                                serialization_dir,
                                param_path,
                                num_samples,
                                num_runs)
else:
    vocabulary_size_experiments(grammar_vocab_size_and_dist,
                                main_args,
                                serialization_dir,
                                param_path,
                                args.num_samples,
                                args.num_runs)