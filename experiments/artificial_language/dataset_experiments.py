
# coding: utf-8

import os
import sys
import wandb

from matplotlib import pyplot as plt
from datetime import datetime

from datetime import datetime
from typing import Dict, List, Callable

from allennlp.common import Params
from allennlp.common.util import import_submodules

import_submodules("quant_exp_bias")
from quant_exp_bias.utils import (get_args, quantify_exposure_bias_runner,
                  sample_oracle_runner, train_runner)
from experiments.util import initialize_experiments, generate_grammar_file, one_exp_run

import json

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id, experiment = initialize_experiments('artificial_lang/dataset_experiments')
generate_grammar_file(serialization_dir)

# dataset_experiments_params = [(100, 8), (1000,6) , (10000, 4), (100000, 2), (1000000, 1)]
dataset_experiments_params = [(100, 1), (1000, 1) , (10000, 1), (100000, 1), (1000000, 1)]

# # Validation Experiments

def dataset_experiments(dataset_experiments_params,
                        main_args,
                        serialization_dir,
                        param_path):
    step = 0
    for num_samples, num_runs in dataset_experiments_params:
        for num_run in range(num_runs):
            run_metrics = one_exp_run(serialization_dir=serialization_dir,
                                        num_samples=num_samples,
                                        run=num_run,
                                        param_path=param_path)

            assert len(run_metrics) == 1, \
                'For this experiment, there should only be one final metric object for a run.'
            run_metrics = run_metrics[0]

            for exp_bias_idx, (exp_bias, df_p_q, df_q_p) in enumerate(zip(run_metrics['exp_biases'],
                                                                        run_metrics['df_p_qs'],
                                                                        run_metrics['df_q_ps'])):
                result= {
                    'exp_bias': exp_bias,
                    'Df_p_q': df_p_q,
                    'Df_q_p': df_q_p,
                    'exp_bias_idx': exp_bias_idx,
                    'num_samples': num_samples,
                    'num_run': num_run,
                    'val_ppl': run_metrics['best_validation_perplexity'],
                    'best_val_epoch': run_metrics['best_epoch'],
                }
                experiment.log_metrics(result, step=step)
                step += 1

dataset_experiments(dataset_experiments_params, main_args, serialization_dir, param_path)
