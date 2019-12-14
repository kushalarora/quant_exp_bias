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

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id = initialize_experiments('validation_experiments')
# num_samples_and_runs = [(1000, 8), (10000,4), (100000,2)]
num_samples_and_runs = [(1000, 1), (10000,1), (100000,1)]


# # Dataset Experiments

def validation_experiments(num_samples_and_runs, 
                            main_args, 
                            serialization_dir, 
                            param_path):

    overrides = json.dumps({'trainer': {'num_epochs': 50, 'patience': None}})

    def validation_exp_bias_epochs_func(train_model_serialization_dir):
        for epoch in range(len(glob.glob(os.path.join(train_model_serialization_dir + '/model_state_epoch_*.th')))):
            qeb_suffix = f'epoch_{epoch}'
            metrics_filename = f'metrics_epoch_{epoch}.json'
            yield (epoch, qeb_suffix, metrics_filename)

    for num_samples, num_runs in num_samples_and_runs:
        for run in range(num_runs):
            run_metrics_list = one_exp_run(serialization_dir=serialization_dir, 
                                            num_samples=num_samples,
                                            run=run, 
                                            param_path=param_path,
                                            overides_func=lambda:overrides,
                                            exp_bias_epochs_func=validation_exp_bias_epochs_func)
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

validation_experiments(num_samples_and_runs, main_args, serialization_dir, param_path)
