
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
from util import run_on_cluster, initialize_experiments, one_exp_run

import json

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id = initialize_experiments('scheduled_sampling_experiments')
scheduled_sampling_ratios  = [
        ('uniform', 0.0, -1), ('uniform', 0.1, -1), ('uniform', 0.25, -1), ('uniform', 0.5, -1), ('uniform', 1.0, -1),  # Fixed SS ratio
        ('quantized', 1.0, 50), ('quantized', 1.0, 100), ('quantized', 1.0, 250), ('quantized', 1.0, 500), ('quantized', 1.0, 1000),  # Linearly increase ss ratio.
        ('linear', 1.0, 50), ('linear', 1.0, 100), ('linear', 1.0, 250), ('linear', 1.0, 500), ('linear', 1.0, 1000),  # Linearly increase ss ratio.
]

num_samples_and_runs = [(1000, 8), (10000,4), (100000,2)]


# # Validation Experiments

@run_on_cluster(job_name='scheduled_sampling_experiments', 
                job_id=experiment_id,
                conda_env='quant_exp', gpu=1,
                walltime="18:00:00")
def scheduled_sampling_experiments(scheduled_sampling_ratios, 
                                    num_samples_and_runs, 
                                    main_args, 
                                    serialization_dir, 
                                    param_path):

    model_size_exp_results = {}
    for  ss_type, ss_ratio, ss_k in scheduled_sampling_ratios:
        model_size_exp_results[f'{ss_type}_{ss_ratio}_{ss_k}'] = []
        
    orig_serialization_dir = serialization_dir
    for ss_type, ss_ratio, ss_k in scheduled_sampling_ratios:
        serialization_dir = os.path.join(orig_serialization_dir, f'{ss_type}_{ss_ratio}_{ss_k}')
        overrides = json.dumps({'model':{'scheduled_sampling_ratio': ss_ratio, 
                                    'scheduled_sampling_k': ss_k,
                                    'scheduled_sampling_type': ss_type},
                               })
        for num_samples, num_runs in num_samples_and_runs:
            for run in range(num_runs):
                run_metrics = one_exp_run(serialization_dir=serialization_dir, 
                                            num_samples=num_samples,
                                            run=run, 
                                            param_path=param_path, 
                                            overides_func=lambda:overrides)
            
                assert len(run_metrics) == 1, \
                    'For this experiment, there should only be one final metric object for a run.'
                run_metrics = run_metrics[0]
                result = {
                            'exp_biases': run_metrics['exp_biases'],
                            'exp_bias_mean': run_metrics['exp_bias_mean'],
                            'exp_bias_std': run_metrics['exp_bias_std'],
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
                model_size_exp_results[f'{ss_type}_{ss_ratio}_{ss_k}'].append(result)
                wandb.log(result)
    return model_size_exp_results

dataset_exp_results = scheduled_sampling_experiments(scheduled_sampling_ratios, 
                                                        num_samples_and_runs,
                                                        main_args, 
                                                        serialization_dir, 
                                                        param_path)

result_path = os.path.join(serialization_dir, 'scheduled_sampling_experiments.json')
with open(result_path, 'w') as f:
    json.dump(dataset_exp_results, f, indent=4, sort_keys=True)
print(result_path)
