
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
from util import run_on_cluster, initialize_experiments

import json

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id = initialize_experiments('model_size_experiments')
model_sizes  = ['xsmall', 'small', 'medium', 'large', 'xlarge']

num_samples_and_runs = [(1000, 8), (10000,4), (100000,2)]
num_trials = 10
num_samples_per_length=2000

# # Validation Experiments

@run_on_cluster(job_name='model_size_experiments', 
                job_id=experiment_id,
                conda_env='quant_exp', gpu=1)
def model_size_experiments(main_args, serialization_dir, param_path):
    # Setup variables needed later.
    model_size_exp_results = {}
    for size in model_sizes:
        model_size_exp_results[size] = []
        
    orig_serialization_dir = serialization_dir
    for model_size in reversed(model_sizes):
        param_path = f'training_configs/model_size_experiments/artificial_grammar_{model_size}.jsonnet'
        msexp_serialization_dir = os.path.join(orig_serialization_dir, model_size)
        for num_samples, num_runs in num_samples_and_runs:
            for run in range(num_runs):
                serialization_dir = os.path.join(msexp_serialization_dir, str(num_samples), str(run))
                sample_oracle_args = get_args(args=['sample-oracle', param_path, '-s', serialization_dir, f'-n {num_samples}'])
                oracle_train_filename, oracle_dev_filename = sample_oracle_runner(sample_oracle_args, 
                                                                                serialization_dir);

                os.environ['ARTIFICIAL_GRAMMAR_TRAIN'] = oracle_train_filename
                os.environ['ARTIFICIAL_GRAMMAR_DEV'] = oracle_dev_filename

                train_args = get_args(args=['train' , param_path, '-s', serialization_dir])
                trainer_params = Params.from_file(train_args.param_path, train_args.overrides)
                cuda_device = trainer_params['trainer']['cuda_device']
                train_model_serialization_dir = train_runner(train_args, 
                                                            serialization_dir);

                archive_file = os.path.join(train_model_serialization_dir, 'model.tar.gz')
                qeb_output_dir = os.path.join(serialization_dir, 'exp_bias')

                metrics = json.load(open(os.path.join(train_model_serialization_dir, f'metrics.json')))

                qeb_args = get_args(args=['quantify-exposure-bias', archive_file, '--output-dir', qeb_output_dir])
                exp_biases, exp_bias_mean, exp_bias_std = quantify_exposure_bias_runner(qeb_args, 
                                                                                        archive_file,
                                                                                        qeb_output_dir,
                                                                                        cuda_device=cuda_device)

                result = {
                            'exp_biases': exp_biases,
                            'exp_bias_mean': exp_bias_mean,
                            'exp_bias_std': exp_bias_std,
                            'num_run': run,
                            'num_samples': num_samples,
                            'model_size': model_size,
                            'val_ppl': metrics['best_validation_perplexity'],
                            'best_val_epoch': metrics['best_epoch']
                        }
                model_size_exp_results[model_size].append(result)
                wandb.log(result)
    return model_size_exp_results

dataset_exp_results = model_size_experiments(main_args, serialization_dir, param_path)

result_path = os.path.join(serialization_dir, 'model_size_experiments.json')
with open(result_path, 'w') as f:
    json.dump(dataset_exp_results, f, indent=4, sort_keys=True)
print(result_path)
#
