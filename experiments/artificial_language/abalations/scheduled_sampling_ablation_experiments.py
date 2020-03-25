
# coding: utf-8

import os
import sys

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
from experiments.util import initialize_experiments, generate_grammar_file, one_exp_run

import json

scheduled_sampling_ratios  = [
        ('uniform', 0.0, -1), ('uniform', 0.1, -1), ('uniform', 0.25, -1), ('uniform', 0.5, -1), ('uniform', 1.0, -1),  # Fixed SS ratio
        ('quantized', 1.0, 50), ('quantized', 1.0, 100), ('quantized', 1.0, 250), ('quantized', 1.0, 500), ('quantized', 1.0, 1000),  # Linearly increase ss ratio.
        ('linear', 1.0, 50), ('linear', 1.0, 100), ('linear', 1.0, 500), ('linear', 1.0, 1000),  # Linearly increase ss ratio.
]

import argparse
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--num_samples', type=int, default=10000,
                    help='Number of dataset samples to run this iteration for.')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs for the given dataset size.')
parser.add_argument('--all', action='store_true', help='Run All configurations mentioned below..')
parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
parser.add_argument('--exp_msg', type=str, default=None, help='Debug(maybe) experiment message.')

args = parser.parse_args()

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id, experiment = initialize_experiments('artificial_lang/scheduled_sampling_experiments',
                                                                                             debug=args.debug,
                                                                                             experiment_text=args.exp_msg,
                                                                                             )

num_samples_and_runs = [(10000,4)]

experiment.log_parameters({'serialization_dir': serialization_dir,
                          'main_args': main_args,
                          'param_path': param_path,
                          'experiment_id': experiment_id})

def scheduled_sampling_experiments(scheduled_sampling_ratios,
                                    main_args,
                                    serialization_dir,
                                    param_path,
                                    num_samples,
                                    num_runs,
                                  ):
    step = 0
    orig_serialization_dir = serialization_dir
    for ss_type, ss_ratio, ss_k in scheduled_sampling_ratios:
        serialization_dir = os.path.join(orig_serialization_dir, f'{ss_type}_{ss_ratio}_{ss_k}')
        ss_k = ss_k * (num_samples/1000.0)
        overrides = json.dumps({'model': {
                                    'decoder': {
                                        'scheduled_sampling_ratio': ss_ratio,
                                        'scheduled_sampling_k': ss_k,
                                        'scheduled_sampling_type': ss_type},
                                    }
                                })
        for run in range(num_runs):
            run_metrics = one_exp_run(serialization_dir=serialization_dir,
                                        num_samples=num_samples,
                                        run=run,
                                        param_path=param_path,
                                        overides_func=lambda:overrides,
                                        shall_generate_grammar_file=True,
                                        num_trials=5,
                                        num_length_samples=5,
                                        num_samples_per_length=100,
                                     )

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
                        'scheduled_sampling_ratio': ss_ratio,
                        'scheduled_sampling_k': ss_k,
                        'scheduled_sampling_type': ss_type,
                        'num_run': run,
                        'num_samples': num_samples,
                        'val_ppl': run_metrics['best_validation_perplexity'],
                        'best_val_epoch': run_metrics['best_epoch'],
                        'final_ss_ratio': run_metrics['validation_ss_ratio'],
                        'best_val_ss_ratio': run_metrics['best_validation_ss_ratio']
                    }
                experiment.log_metrics(result, step=step)
                step += 1
                sleep(randint(1,10)/10.0)

            experiment.log_metric('exp_bias_mean', run_metrics['exp_bias_mean'], step=step)
            experiment.log_metric('df_p_q_mean', run_metrics['df_p_q_mean'], step=step)
            experiment.log_metric('df_q_p_mean', run_metrics['df_q_p_mean'], step=step)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        scheduled_sampling_experiments(scheduled_sampling_ratios,
                                        main_args,
                                        serialization_dir,
                                        param_path,
                                        num_samples,
                                        num_runs)
else:
    scheduled_sampling_experiments(scheduled_sampling_ratios,
                                main_args,
                                serialization_dir,
                                param_path,
                                args.num_samples,
                                args.num_runs)
