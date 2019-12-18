# coding: utf-8

import os
import sys
import wandb

from matplotlib import pyplot as plt


from typing import Dict, List

from allennlp.common import Params
from allennlp.common.util import import_submodules

import_submodules("quant_exp_bias")
from quant_exp_bias.utils import (get_args, quantify_exposure_bias_runner, 
                  sample_oracle_runner, train_runner)
from experiments.util import initialize_experiments, one_exp_run

import glob
import json
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--num_samples', type=int, default=10000,
                    help='Number of dataset samples to run this iteration for.')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs for the given dataset size.')
parser.add_argument('--all', action='store_true', help='Run All configurations mentioned below..')
args = parser.parse_args()

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id = initialize_experiments('natural_lang/validation_experiments',
                                                                                 is_natural_lang_exp=True)
num_samples_and_runs = [(50000, 6), (500000, 4), (2000000, 2)]

# # Dataset Experiments

def validation_experiments(main_args,
                            serialization_dir,
                            param_path,
                            num_samples,
                            num_runs,
                           ):

    overrides = json.dumps({'trainer': {'num_epochs': 20, 'patience': None}})

    def validation_exp_bias_epochs_func(train_model_serialization_dir):
        for epoch in range(len(glob.glob(os.path.join(train_model_serialization_dir + '/model_state_epoch_*.th')))):
            qeb_suffix = f'epoch_{epoch}'
            metrics_filename = f'metrics_epoch_{epoch}.json'
            yield (epoch, qeb_suffix, metrics_filename)

    for run in range(num_runs):
        run_metrics_list = one_exp_run(serialization_dir=serialization_dir, 
                                        num_samples=num_samples,
                                        run=run, 
                                        param_path=param_path,
                                        overides_func=lambda:overrides,
                                        exp_bias_epochs_func=validation_exp_bias_epochs_func,
                                        sample_from_file=True,
                                        dataset_filename='data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered')
        for run_metrics in run_metrics_list:
            epoch = run_metrics['epoch']

            for exp_bias_idx, exp_bias in enumerate(run_metrics['exp_biases']):
                results = {
                            'exp_bias': exp_bias, 
                            'exp_bias_idx': exp_bias_idx,
                            'val_ppl': run_metrics['validation_perplexity'],
                            'epoch': epoch,
                            'num_run': run,
                            'num_samples': num_samples
                        }

                wandb.log(results)
if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        validation_experiments(main_args, serialization_dir, param_path, num_samples, num_runs)
else:
    validation_experiments(main_args, serialization_dir, param_path, args.num_samples, args.num_runs)
