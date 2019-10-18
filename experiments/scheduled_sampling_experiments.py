
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
from util import run_on_cluster, initialize_experiments

import json

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path = initialize_experiments()
scheduled_sampling_ratios  = [(1.0, 300), (0.5, 300), (0.25, 300), (0.1, 300), (0.05, 300), (0,0)]

num_sample_oracles = 8
num_trials = 10
num_samples_per_length=2000

# # Validation Experiments

@run_on_cluster(job_name='scheduled_sampling_experiments', 
                conda_env='quant_exp', gpu=1,
                walltime="14:00:00")
def scheduled_sampling_experiments(main_args, serialization_dir, param_path):
    # Setup variables needed later.
    model_size_exp_results = {}
    for ss_ratio, ss_k in scheduled_sampling_ratios:
        model_size_exp_results[f'{ss_ratio}_{ss_k}'] = []
        
    orig_serialization_dir = serialization_dir
    for ss_ratio, ss_k in scheduled_sampling_ratios:
        msexp_serialization_dir = os.path.join(orig_serialization_dir, 'scheduled_sampling_experiments', f'{ss_ratio}_{ss_k}')
        for num_run in range(num_sample_oracles):
            serialization_dir = os.path.join(msexp_serialization_dir, str(num_run))
            sample_oracle_args = get_args(args=['sample-oracle', param_path, '-s', serialization_dir])
            oracle_train_filename, oracle_dev_filename = sample_oracle_runner(sample_oracle_args, 
                                                                              serialization_dir);

            os.environ['ARTIFICIAL_GRAMMAR_TRAIN'] = oracle_train_filename
            os.environ['ARTIFICIAL_GRAMMAR_DEV'] = oracle_dev_filename
            overrides = json.dumps({'model':{'scheduled_sampling_ratio': ss_ratio, 
                                             'scheduled_sampling_k': ss_k},
                                    'trainer': {'patience': None}})

            train_args = get_args(args=['train' , param_path, '-s', serialization_dir, '-o', overrides])
            trainer_params = Params.from_file(train_args.param_path, train_args.overrides)
            cuda_device = trainer_params['trainer']['cuda_device']
            train_model_serialization_dir = train_runner(train_args, 
                                                        serialization_dir);

            archive_file = os.path.join(train_model_serialization_dir, 'model.tar.gz')
            qeb_output_dir = os.path.join(serialization_dir, 'exp_bias')

            qeb_args = get_args(args=['quantify-exposure-bias', archive_file, '--output-dir', qeb_output_dir])
            exp_biases, exp_bias_mean, exp_bias_std = quantify_exposure_bias_runner(qeb_args, 
                                                                                    archive_file,
                                                                                    qeb_output_dir,
                                                                                    cuda_device=cuda_device, 
                                                                                    num_trials=num_trials,
                                                                                    num_samples_per_length=num_samples_per_length)
            model_size_exp_results[f'{ss_ratio}_{ss_k}'].extend(exp_biases)
    return model_size_exp_results

dataset_exp_results = scheduled_sampling_experiments(main_args, serialization_dir, param_path)

result_path = os.path.join(serialization_dir, 'scheduled_sampling_experiments.json')
with open(result_path, 'w') as f:
    json.dump(dataset_exp_results, f, indent=4, sort_keys=True)
print(result_path)
