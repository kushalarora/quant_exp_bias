
# coding: utf-8
import itertools
import os
import sys
from functools import partial

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

import argparse
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--num_samples', type=int, default=50000,
                    help='Number of dataset samples to run this iteration for.')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs for the given dataset size.')
parser.add_argument('--all', action='store_true', help='Run All configurations mentioned below..')
parser.add_argument('--rollins', nargs='+', help='Rollins to use', type=str, default=['teacher_forcing', 'learned'])
parser.add_argument('--rollouts', nargs='+', help='Rollouts to use', type=str, default=['reference', 'mixed', 'learned'])
parser.add_argument('--rollout_cost_funcs', nargs='+', help='Type of Oracle to use', type=str, default=['noisy_oracle', 'bleu'])
parser.add_argument('--mixing_coeff', nargs='+', help='Mixing coefficients for rollin and rollouts', type=float, default=[0.25, 0.5, 0,])

parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
parser.add_argument('--exp_msg', type=str, default=None, help='Debug(maybe) experiment message.')

args = parser.parse_args()

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id, experiment = initialize_experiments('natural_lang/searnn_ablation_experiments', 
                                                                                param_path='training_configs/natural_lang/emnlp_news_gpt2_searnn.jsonnet',
                                                                                 debug=args.debug,
                                                                                 experiment_text=args.exp_msg,
                                                                                )

rollin_rollout_cost_func_configs = [x for x in itertools.product(args.rollins, args.rollouts, args.rollout_cost_funcs, args.mixing_coeff)]

num_samples_and_runs = [(50000,2)]

def rollout_cost_function_configs(cost_func, mixing_coeff):
    if cost_func == 'bleu':
        overrides_dict = { "type": "bleu",}
    elif cost_func == 'noisy_oracle':
        overrides_dict = {
          "type": "noisy_oracle",
          "oracle": {
            "type": "gpt2_oracle",
            "model_name": "gpt2",
            "batch_size": 4,
            "cuda_device": -2,
          },
        }
    overrides_dict = {
        "model": {
            "decoder": {
                "rollout_cost_function": overrides_dict,
                "rollin_rollout_mixing_coeff": mixing_coeff,
            }
        }
    }
    experiment.log_parameters(overrides_dict)
    return json.dumps(overrides_dict)

experiment.log_parameters({'serialization_dir': serialization_dir,
                          'main_args': main_args,
                          'param_path': param_path,
                          'experiment_id': experiment_id})

def searnn_experiments(rollin_rollout_configs,
                            main_args,
                            serialization_dir,
                            param_path,
                            num_samples,
                            num_runs,
                      ):
    # Setup variables needed later.
    step = 0
    orig_serialization_dir = serialization_dir
    for rollin_policy, rollout_policy, cost_func, mixing_coeff in rollin_rollout_cost_func_configs:
        os.environ['rollin_mode'] = rollin_policy
        os.environ['rollout_mode'] = rollout_policy

        serialization_dir = os.path.join(orig_serialization_dir, f'{rollin_policy}_{rollout_policy}_{cost_func}_{mixing_coeff}')
        for num_run in range(num_runs):
            # Using partial here as we want the os.environ for noisy oracle to be
            # instantiated with overrides method is called.
            run_metrics = one_exp_run(serialization_dir=serialization_dir, 
                                        num_samples=num_samples,
                                        run=num_run,
                                        param_path=param_path,
                                        overides_func=partial(rollout_cost_function_configs, cost_func, mixing_coeff), 
                                        sample_from_file=True,
                                        dataset_filename='data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered',
                                     )

            assert len(run_metrics) == 1, \
                'For this experiment, there should only be one final metric object for a run.'
            run_metrics = run_metrics[0]
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
                            'rollin_policy': rollin_policy,
                            'rollout_policy': rollout_policy,
                            'cost_func': cost_func,
                            'mixing_coeff': mixing_coeff,
                            'val_ppl': run_metrics['best_validation_perplexity'],
                            'best_val_epoch': run_metrics['best_epoch']
                        }
                experiment.log_metrics(result, step=step)
                step += 1
            experiment.log_metric('exp_bias_mean', run_metrics['exp_bias_mean'], step=step)
            experiment.log_metric('df_p_q_mean', run_metrics['df_p_q_mean'], step=step)
            experiment.log_metric('df_q_p_mean', run_metrics['df_q_p_mean'], step=step)
            sleep(randint(1,10)/10.0)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        searnn_experiments(rollin_rollout_cost_func_configs, main_args,
                           serialization_dir, param_path, num_samples, num_runs)
else:
    searnn_experiments(rollin_rollout_cost_func_configs, main_args, serialization_dir, param_path, args.num_samples, args.num_runs)
