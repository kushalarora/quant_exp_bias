
# coding: utf-8

import os
import sys
import wandb

from matplotlib import pyplot as plt
from datetime import datetime

from datetime import datetime
from typing import Dict, List, Callable

from allennlp.common import Params
from allennlp.common.util import import_submodules

import_submodules("quant_exp_bias")
from quant_exp_bias.utils import (get_args, quantify_exposure_bias_runner,
                  sample_oracle_runner, train_runner)
from experiments.util import initialize_experiments, one_exp_run

import json

import argparse
parser = argparse.ArgumentParser(description='Dataset Size Experiments.')
parser.add_argument('--num_samples', type=int, default=10000,
                    help='Number of dataset samples to run this iteration for.')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs for the given dataset size.')
parser.add_argument('--all', action='store_true', help='Run All configurations mentioned below..')
args = parser.parse_args()

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id, experiment = initialize_experiments('natural_lang/dataset_experiments',
                                                                                 param_path = 'training_configs/natural_lang/emnlp_news_gpt2.jsonnet',
                                                                                )

dataset_experiments_params = [(10000, 8), (50000, 6) , (500000, 4), (2000000, 2), (5000000, 1)]

experiment.log_parameters({'serialization_dir': serialization_dir,
                          'main_args': main_args,
                          'param_path': param_path,
                          'experiment_id': experiment_id})

def dataset_experiments(main_args,
                        serialization_dir,
                        param_path,
                        num_samples,
                        num_runs,
                       ):
    step = 0
    for num_run in range(num_runs):
        run_metrics = one_exp_run(serialization_dir=serialization_dir,
                                    num_samples=num_samples,
                                    run=num_run,
                                    param_path=param_path,
                                    sample_from_file=True,
                                    dataset_filename='data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered')

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
                'num_samples': num_samples,
                'num_run': num_run,
                'val_ppl': run_metrics['best_validation_perplexity'],
                'best_val_epoch': run_metrics['best_epoch']
            }
            experiment.log_metrics(result, step=step)
            step += 1

        experiment.log_metric('exp_bias_mean', run_metrics['exp_bias_mean'], step=step)
        experiment.log_metric('df_p_q_mean', run_metrics['df_p_q_mean'], step=step)
        experiment.log_metric('df_q_p_mean', run_metrics['df_q_p_mean'], step=step)

if args.all:
    for num_samples, num_runs in dataset_experiments_params:
        dataset_experiments(main_args, serialization_dir, param_path, num_samples, num_runs)
else:
    dataset_experiments(main_args, serialization_dir, param_path, args.num_samples, args.num_runs)