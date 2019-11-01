
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
from experiments.util import run_on_cluster, initialize_experiments, one_exp_run

import json

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id = initialize_experiments('model_size_experiments')
model_sizes  = ['xsmall', 'small', 'medium', 'large', 'xlarge']
num_samples_and_runs = [(1000, 8), (10000,4), (100000,2)]

# # Validation Experiments

@run_on_cluster(job_name='model_size_experiments', 
                job_id=experiment_id,
                conda_env='quant_exp', gpu=1)
def model_size_experiments(model_sizes,
                            num_samples_and_runs, 
                            main_args, 
                            serialization_dir, 
                            param_path):
    
    # Setup variables needed later.
    orig_serialization_dir = serialization_dir
    for model_size in reversed(model_sizes):
        param_path = f'training_configs/model_size_experiments/artificial_grammar_{model_size}.jsonnet'
        serialization_dir = os.path.join(orig_serialization_dir, model_size)
        for num_samples, num_runs in num_samples_and_runs:
            for num_run in range(num_runs):
                run_metrics = one_exp_run(serialization_dir=serialization_dir, 
                                            num_samples=num_samples,
                                            run=num_run, 
                                            param_path=param_path)
            
                assert len(run_metrics) == 1, \
                    'For this experiment, there should only be one final metric object for a run.'
                run_metrics = run_metrics[0]
                for exp_bias_idx, exp_bias in enumerate(run_metrics['exp_biases']):
                    result = {
                                'exp_bias': exp_bias, 
                                'exp_bias_idx': exp_bias_idx,
                                'num_run': num_run,
                                'num_samples': num_samples,
                                'model_size': model_size,
                                'val_ppl': run_metrics['best_validation_perplexity'],
                                'best_val_epoch': run_metrics['best_epoch']
                            }
                wandb.log(result)

model_size_experiments(model_sizes, num_samples_and_runs, main_args, serialization_dir, param_path)
