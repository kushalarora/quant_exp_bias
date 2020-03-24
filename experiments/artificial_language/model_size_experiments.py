
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


import argparse
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--num_samples', type=int, default=10000,
                    help='Number of dataset samples to run this iteration for.')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs for the given dataset size.')
parser.add_argument('--all', action='store_true', help='Run All configurations mentioned below..')
parser.add_argument('--model_sizes', nargs='+', help='Model Sizes to consider', type=str, default=['xsmall', 'small', 'medium', 'large', 'xlarge'])
parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
parser.add_argument('--exp_msg', type=str, default=None, help='Debug(maybe) experiment message.')


args = parser.parse_args()

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id, experiment = initialize_experiments('artificial_lang/model_size_experiments', 
                                                                                             debug=args.debug,
                                                                                             experiment_text=args.exp_msg,
                                                                                             )

model_size_configs  = {
    'xsmall' : (10, 10, 1),
    'small': (50, 50, 1),
    'medium': (100, 100, 1),
    'large': (300, 300, 1),
    'xlarge': (300, 1200, 1)
    }
num_samples_and_runs = [(1000, 8), (10000,4), (100000,2)]
# num_samples_and_runs = [(1000, 1), (10000,1), (100000,1)]

experiment.log_parameters({'serialization_dir': serialization_dir,
                          'main_args': main_args,
                          'param_path': param_path,
                          'experiment_id': experiment_id})

def model_size_experiments(model_sizes,
                            main_args, 
                            serialization_dir, 
                            param_path,
                            num_samples,
                            num_runs,
                           ):
    step = 0
    # Setup variables needed later.
    orig_serialization_dir = serialization_dir
    for model_size in model_sizes:
        model_tuple = model_size_configs[model_size]
        overrides = json.dumps({'model':{
                                    'decoder': {
                                        'decoder_net': {
                                            'decoding_dim': model_tuple[0],
                                            'target_embedding_dim': model_tuple[1],
                                            'num_decoder_layers': model_tuple[2]
                                        },
                                        'target_embedder': {
                                            'embedding_dim': model_tuple[1],
                                        }
                                    }
                                }})
        serialization_dir = os.path.join(orig_serialization_dir, model_size)
        for num_run in range(num_runs):
            run_metrics = one_exp_run(serialization_dir=serialization_dir, 
                                        num_samples=num_samples,
                                        run=num_run,
                                        param_path=param_path,
                                        overides_func=lambda:overrides,
                                        shall_generate_grammar_file=True,
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
                            'model_size': model_size,
                            'val_ppl': run_metrics['best_validation_perplexity'],
                            'best_val_epoch': run_metrics['best_epoch'],
                        }
                experiment.log_parameter('model_size', model_size, step=step)
                experiment.log_metrics(result, step=step)
                step += 1
            sleep(randint(1,10)/10.0)

            experiment.log_metric('exp_bias_mean', run_metrics['exp_bias_mean'], step=step)
            experiment.log_metric('df_p_q_mean', run_metrics['df_p_q_mean'], step=step)
            experiment.log_metric('df_q_p_mean', run_metrics['df_q_p_mean'], step=step)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        model_size_experiments(args.model_sizes, main_args, serialization_dir, param_path, num_samples, num_runs)
else:
    model_size_experiments(args.model_sizes, main_args, serialization_dir, param_path, args.num_samples, args.num_runs)
