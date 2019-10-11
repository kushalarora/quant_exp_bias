# coding: utf-8

import os
import sys
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


from datetime import datetime
from typing import Dict, List

from allennlp.common import Params
from allennlp.common.util import import_submodules

import_submodules("quant_exp_bias")

from utils import (get_args, quantify_exposure_bias_runner, run,
                  sample_oracle_runner, train_runner)
from pprint import pprint


# ## Basic Setup of grammar and global variables like serialization directory and training config file

# In[ ]:


FSA_GRAMMAR_STRING = """
                        q0 -> 'S' q1 [0.9900] | 'a' q1 [0.0025] | 'b' q1 [0.0025] | 'c' q1 [0.0025] | 'E' q1 [0.0025]
                        q1 -> 'S' q1 [0.0025] | 'a' q1 [0.3000] | 'b' q1 [0.3000] | 'c' q1 [0.3000] | 'E' q1 [0.0025]
                        q1 -> 'S' q2 [0.0025] | 'a' q2 [0.0300] | 'b' q2 [0.0300] | 'c' q2 [0.0300] | 'E' q2 [0.0025]
                        q2 -> 'S' [0.0025] | 'a' [0.0025] | 'b' [0.0025] | 'c' [0.0025] | 'E' [0.9900]
                    """
    
os.environ["FSA_GRAMMAR_STRING"] = FSA_GRAMMAR_STRING
os.environ['ARTIFICIAL_GRAMMAR_TRAIN'] = ""
os.environ['ARTIFICIAL_GRAMMAR_DEV'] = ""

# dataset_experiments_params = [(1000, 16), (10000,8) , (25000, 4), (50000,2), (100000,1)]
dataset_experiments_params = [(10000,2)]

num_sample_oracles = 10
num_trials = 10
num_samples_per_length=2000

# Ipython by default adds some arguments to sys.argv.
#  We don't want those arguments, hence we pass [] here.
#
# The deafult argument get_args is args=None. 
# This translates to parsing sys.argv. This is useful
# in case we run the method from a python file but not here.
# Hence, we keep the default argument as None but pass [] for 
# ipython notebook.
main_args = get_args(args=[])

serialization_dir = os.path.join(main_args.output_dir, datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
param_path = main_args.config


# # Dataset Experiments

# In[ ]:
import glob
import json
import numpy as np

validation_exp_results = {}
for num_run in range(num_sample_oracles):
    num_run_serialization_dir = os.path.join(serialization_dir, 'validation_experiments', str(num_run))
    sample_oracle_args = get_args(args=['sample-oracle', param_path, '-s', num_run_serialization_dir])
    oracle_train_filename, oracle_dev_filename = sample_oracle_runner(sample_oracle_args, 
                                                                      num_run_serialization_dir);

    os.environ['ARTIFICIAL_GRAMMAR_TRAIN'] = oracle_train_filename
    os.environ['ARTIFICIAL_GRAMMAR_DEV'] = oracle_dev_filename

    train_args = get_args(args=['train' , param_path, '-s', num_run_serialization_dir])
    trainer_params = Params.from_file(train_args.param_path, train_args.overrides)
    cuda_device = trainer_params['trainer']['cuda_device']
    train_model_serialization_dir = train_runner(train_args, 
                                                 num_run_serialization_dir);

    archive_file = os.path.join(train_model_serialization_dir, 'model.tar.gz')
    for epoch in range(len(glob.glob(os.path.join(train_model_serialization_dir + '/model_state_epoch_*.th')))):
        qeb_output_dir = os.path.join(num_run_serialization_dir, 'exp_bias', 'epoch_' + str(epoch))
        
        # TODO(Kushal): Clean this up.
        validation_ppl = json.load(open(os.path.join(train_model_serialization_dir, f'metrics_epoch_{epoch}.json')))['validation_perplexity']

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
            validation_exp_results[epoch] = {
                                                'exp_biases': exp_biases,
                                                'val_ppl': [validation_ppl],
                                                'exp_mean': [exp_bias_mean],
                                                'exp_std': [exp_bias_std]
                                            }
        else:
            validation_exp_results[epoch]['exp_biases'].extend(exp_biases)
            validation_exp_results[epoch]['exp_mean'].append(exp_bias_mean)
            validation_exp_results[epoch]['exp_biases'].append(exp_bias_std)
            validation_exp_results[epoch]['val_ppl'].append(validation_ppl)

pd_validation_exp_results = pd.DataFrame(validation_exp_results)
result_path = os.path.join(serialization_dir, 'validation_experiments.json')
pd_validation_exp_results.to_json(result_path)
print(result_path)
