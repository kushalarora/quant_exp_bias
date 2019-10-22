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
from util import run_on_cluster, initialize_experiments

import glob
import json
import numpy as np

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id = initialize_experiments('validation_experiments')
num_samples_and_runs = [(1000, 8), (10000,4), (100000,2)]
num_trials = 10
num_samples_per_length=2000

# # Dataset Experiments

@run_on_cluster(job_name='validation_experiments', 
                job_id=experiment_id,
                conda_env='quant_exp', gpu=1)
def validation_experiments(num_sample_oracles, num_trials, num_samples_per_length, serialization_dir, param_path):
    validation_exp_results = {}
    for num_samples, num_runs in num_samples_and_runs:
        for run in range(num_runs):
            num_run_serialization_dir = os.path.join(serialization_dir, 'validation_experiments', str(num_samples), str(run))
            sample_oracle_args = get_args(args=['sample-oracle', param_path, '-s', num_run_serialization_dir, f'-n {num_samples}'])
            oracle_train_filename, oracle_dev_filename = sample_oracle_runner(sample_oracle_args, 
                                                                            num_run_serialization_dir);

            os.environ['ARTIFICIAL_GRAMMAR_TRAIN'] = oracle_train_filename
            os.environ['ARTIFICIAL_GRAMMAR_DEV'] = oracle_dev_filename

            overrides = json.dumps({'trainer': {'num_epochs': 20, 'patience': None}})
            train_args = get_args(args=['train' , param_path, '-s', num_run_serialization_dir, '-o', overrides])
            trainer_params = Params.from_file(train_args.param_path, train_args.overrides)
            cuda_device = trainer_params['trainer']['cuda_device']
            train_model_serialization_dir = train_runner(train_args, 
                                                        num_run_serialization_dir);

            archive_file = os.path.join(train_model_serialization_dir, 'model.tar.gz')
            for epoch in range(len(glob.glob(os.path.join(train_model_serialization_dir + '/model_state_epoch_*.th')))):
                qeb_output_dir = os.path.join(num_run_serialization_dir, 'exp_bias', 'epoch_' + str(epoch))
                
                metrics = json.load(open(os.path.join(train_model_serialization_dir, f'metrics_epoch_{epoch}.json')))

                weights_file = os.path.join(train_model_serialization_dir, f'model_state_epoch_{epoch}.th')
                qeb_args = get_args(args=['quantify-exposure-bias', archive_file, '--output-dir', qeb_output_dir, '--weights-file', weights_file])
                exp_biases, exp_bias_mean, exp_bias_std = quantify_exposure_bias_runner(qeb_args, 
                                                                                        archive_file,
                                                                                        qeb_output_dir,
                                                                                        cuda_device=cuda_device,
                                                                                        weights_file=weights_file,
                                                                                        num_trials=num_trials,
                                                                                        num_samples_per_length=num_samples_per_length);
                if epoch not in validation_exp_results:
                    validation_exp_results[epoch] = []
            
                results = {
                            'exp_biases': exp_biases,
                            'val_ppl': metrics['validation_perplexity'],
                            'epoch': epoch,
                            'exp_mean': exp_bias_mean,
                            'exp_std': exp_bias_std,
                            'num_run': run,
                            'num_samples': num_samples
                        }

                validation_exp_results[epoch].append(results)
                wandb.log(results)
    return validation_exp_results

validation_exp_results = validation_experiments(num_samples_and_runs, num_trials, num_samples_per_length, serialization_dir, param_path)
result_path = os.path.join(serialization_dir, 'validation_experiments.json')

with open(result_path, 'w') as f:
    json.dump(validation_exp_results, f, indent=4, sort_keys=True)
print(result_path)
