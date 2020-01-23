
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
from experiments.util import initialize_experiments, generate_grammar_file, one_exp_run

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

main_args, serialization_dir, param_path, experiment_id, experiment = initialize_experiments('artificial_lang/beam_size_experiments')
generate_grammar_file(serialization_dir, vocabulary_size=24)

experiment.log_parameters({'serialization_dir': serialization_dir,
                          'main_args': main_args,
                          'param_path': param_path,
                          'experiment_id', experiment_id})

beam_sizes = [2,4,6]
num_samples_and_runs = [(1000, 8), (10000,4), (100000,2)]
def beam_size_experiments(beam_sizes,
                            main_args,
                            serialization_dir,
                            param_path,
                            num_samples,
                            num_runs,
                          ):
    step = 0
    # Setup variables needed later.
    orig_serialization_dir = serialization_dir
    def qeb_beam_size_overrides_func():
        for beam_size in beam_sizes:
            overrides = json.dumps({'model':{
                                        'decoder': {
                                            'beam_size': beam_size,
                                         }
                                       }
                                   })
            yield ('beam_size', beam_size, overrides)

    serialization_dir = os.path.join(orig_serialization_dir, f'beam_size_experiments')
    for num_run in range(num_runs):
        run_metrics_list = one_exp_run(serialization_dir=serialization_dir, 
                                        num_samples=num_samples,
                                        run=num_run, 
                                        param_path=param_path,
                                        exp_bias_inference_funcs=qeb_beam_size_overrides_func,)

        for run_metrics in run_metrics_list:
            beam_size = run_metrics.get('beam_size', 1)

            for exp_bias_idx, (exp_bias, df_p_q, df_q_p) in enumerate(zip(run_metrics['exp_biases'],
                                                                        run_metrics['df_p_qs'],
                                                                        run_metrics['df_q_ps'])):
                result = {
                            'exp_bias': exp_bias,
                            'Df_p_q': df_p_q,
                            'Df_q_p': df_q_p,
                            'exp_bias_idx': exp_bias_idx,
                            'num_run': num_run,
                            'num_samples': num_samples,
                            'beam_size': beam_size,
                            'val_ppl': run_metrics['best_validation_perplexity'],
                            'best_val_epoch': run_metrics['best_epoch'],
                            }
                experiment.log_metrics(result, step=step)
                step += 1

            experiment.log_metric('exp_bias_mean', run_metrics['exp_bias_mean'], step=step)
            experiment.log_metric('df_p_q_mean', run_metrics['df_p_q_mean'], step=step)
            experiment.log_metric('df_q_p_mean', run_metrics['df_q_p_mean'], step=step)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        beam_size_experiments(beam_sizes, main_args, serialization_dir, param_path, num_samples, num_runs)
else:
    beam_size_experiments(beam_sizes, main_args, serialization_dir, param_path, args.num_samples, args.num_runs)
