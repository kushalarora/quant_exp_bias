
# coding: utf-8
import itertools
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
from experiments.util import initialize_experiments, generate_grammar_file, one_exp_run

import json

import argparse
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--num_samples', type=int, default=1000,
                    help='Number of dataset samples to run this iteration for.')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs for the given dataset size.')
parser.add_argument('--all', action='store_true', help='Run All configurations mentioned below..')
args = parser.parse_args()

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id = initialize_experiments('artificial_lang/searnn_experiments', 
                                                                                 param_path='training_configs/artificial_grammar/artificial_grammar_searnn.jsonnet')
generate_grammar_file(serialization_dir)

rollin_configs = ['learned', 'mixed', 'teacher_forcing']
rollout_configs = ['learned', 'mixed', 'reference']
rollin_rollout_configs = [x for x in itertools.product(rollin_configs, rollout_configs)]

num_samples_and_runs = [(1000, 4), (10000,2), (100000,2)]

def searnn_experiments(rollin_rollout_configs,
                            num_samples_and_runs,
                            main_args,
                            serialization_dir,
                            param_path,
                            num_samples,
                            num_runs,
                      ):
    # Setup variables needed later.
    orig_serialization_dir = serialization_dir
    for rollin_policy, rollout_policy in rollin_rollout_configs:
        os.environ['rollin_mode'] = rollin_policy
        os.environ['rollout_mode'] = rollout_policy

        serialization_dir = os.path.join(orig_serialization_dir, f'{rollin_policy}_{rollout_policy}')
        for num_run in range(num_runs):
            run_metrics = one_exp_run(serialization_dir=serialization_dir, 
                                        num_samples=num_samples,
                                        run=num_run, 
                                        param_path=param_path)

            assert len(run_metrics) == 1, \
                'For this experiment, there should only be one final metric object for a run.'
            run_metrics = run_metrics[0]
            for exp_bias_idx, exp_bias in enumerate(run_metrics['exp_biases']):
                result = {
                            'exp_bias': exp_bias, 
                            'exp_bias_idx': exp_bias_idx,
                            'num_run': num_run,
                            'num_samples': num_samples,
                            'rollin_policy': rollin_policy,
                            'rollout_policy': rollout_policy,
                            'val_ppl': run_metrics['best_validation_perplexity'],
                            'best_val_epoch': run_metrics['best_epoch']
                        }
                wandb.log(result)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        searnn_experiments(rollin_rollout_configs, num_samples_and_runs, main_args, serialization_dir, param_path, num_samples, num_runs)
else:
    searnn_experiments(rollin_rollout_configs, num_samples_and_runs, main_args, serialization_dir, param_path, args.num_samples, args.num_runs)
