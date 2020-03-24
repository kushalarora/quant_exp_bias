
# coding: utf-8

import os
import sys

from matplotlib import pyplot as plt
from datetime import datetime

from datetime import datetime
from typing import Dict, List

from allennlp.common import Params
from allennlp.common.util import import_submodules

import_submodules("quant_exp_bias")
from quant_exp_bias.utils import (get_args, quantify_exposure_bias_runner, 
                  sample_oracle_runner, train_runner)
from experiments.util import initialize_experiments, one_exp_run

import json


import argparse
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--num_samples', type=int, default=10000,
                    help='Number of dataset samples to run this iteration for.')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs for the given dataset size.')
parser.add_argument('--all', action='store_true', help='Run All configurations mentioned below..')
parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
parser.add_argument('--exp_msg', type=str, default=None, help='Debug(maybe) experiment message.')
parser.add_argument('--model_sizes', nargs='+', help='Model sizes to use',
                        type=str, default=None)
args = parser.parse_args()

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id, experiment = initialize_experiments('natural_lang/model_size_experiments',
                                                                                             param_path = 'training_configs/natural_lang/emnlp_news_gpt2.jsonnet',
                                                                                             debug=args.debug,
                                                                                             experiment_text=args.exp_msg,
                                                                                            )

msz2config  = {
    'xsmall' : (100, 100, 1),
    'small': (300, 300, 1),
    'medium': (800, 300, 1),
    'large': (2400, 300, 1),
    'xlarge': (2400, 300, 2)
    }

# num_samples_and_runs = [(1000, 8), (10000,4), (100000,2)]
num_samples_and_runs = [(50000, 4), (500000,2), (2000000,2)]

experiment.log_parameters({'serialization_dir': serialization_dir,
                          'main_args': main_args,
                          'param_path': param_path,
                          'experiment_id': experiment_id})

experiment.log_parameters(msz2config)

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
        model_tuple = msz2config[model_size]
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
        experiment.log_parameters(json.loads(overrides), step=step)
        for num_run in range(num_runs):
            run_metrics = one_exp_run(serialization_dir=serialization_dir, 
                                        num_samples=num_samples,
                                        run=num_run, 
                                        param_path=param_path,
                                        overides_func=lambda:overrides,
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
                            'model_size': model_size,
                            'val_ppl': run_metrics['best_validation_perplexity'],
                            'best_val_epoch': run_metrics['best_epoch']
                        }
                experiment.log_metrics(result, step=step)
                step += 1

            experiment.log_metric('exp_bias_mean', run_metrics['exp_bias_mean'], step=step)
            experiment.log_metric('df_p_q_mean', run_metrics['df_p_q_mean'], step=step)
            experiment.log_metric('df_q_p_mean', run_metrics['df_q_p_mean'], step=step)
    experiment.log_parameters({})
    experiment.log_asset_folder(serialization_dir, log_file_name=True, recursive=True)


if args.all:
    model_sizes = msz2configs.keys()
    for num_samples, num_runs in num_samples_and_runs:
        model_size_experiments(model_sizes, main_args,
                                serialization_dir, param_path, num_samples, num_runs)
else:
    # You can specify the model sizes you want to run, else it runs everything.
    model_sizes = args.model_sizes or msz2config.keys()
    model_size_experiments(model_sizes, main_args, serialization_dir, 
                            param_path, args.num_samples, args.num_runs)
