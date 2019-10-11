
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
from utils import (get_args, quantify_exposure_bias_runner, 
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

dataset_experiments_params = [(1000, 16), (10000,8) , (25000, 4), (50000,2), (100000,1)]
# dataset_experiments_params = [(10000,2)]

num_sample_oracles = 1
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

def dataset_experiments(main_args, serialization_dir, param_path):
    # Setup variables needed later.
    dataset_exp_results = {}
    for num_samples, _ in dataset_experiments_params:
        dataset_exp_results[num_samples] = []
        
    orig_serialization_dir = serialization_dir
    for num_samples, num_runs in dataset_experiments_params:
        dexp_serialization_dir = os.path.join(orig_serialization_dir, 'dataset_experiments', str(num_samples))
        for num_run in range(num_runs):
            serialization_dir = os.path.join(dexp_serialization_dir, str(num_run))
            sample_oracle_args = get_args(args=['sample-oracle', param_path, '-s', serialization_dir, '-n', str(num_samples)])
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

            qeb_args = get_args(args=['quantify-exposure-bias', archive_file, '--output-dir', qeb_output_dir])
            exp_biases, exp_bias_mean, exp_bias_std = quantify_exposure_bias_runner(qeb_args, 
                                                                                    archive_file,
                                                                                    qeb_output_dir,
                                                                                    cuda_device=cuda_device, 
                                                                                    num_trials=num_trials,
                                                                                    num_samples_per_length=num_samples_per_length)
            dataset_exp_results[num_samples].extend(exp_biases)
    pprint(dataset_exp_results)
    return dataset_exp_results

dataset_exp_results = dataset_experiments(main_args, serialization_dir, param_path)
for num_samples in dataset_exp_results:
    dataset_exp_results[num_samples]['mean'] = np.mean(dataset_exp_results[num_samples])
    dataset_exp_results[num_samples]['std'] = np.std(dataset_exp_results[num_samples])

pd_dataset_exp = pd.DataFrame(dataset_exp_results)
result_path = os.path.join(serialization_dir, 'dataset_experiments.json')
pd_dataset_exp.to_json(result_path)
#
