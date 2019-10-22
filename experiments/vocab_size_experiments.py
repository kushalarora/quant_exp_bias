
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
from quant_exp_bias.oracles.artificial_grammar_oracle import ArtificialLanguageOracle
from util import run_on_cluster, initialize_experiments

import itertools
import json

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id = initialize_experiments('vocabulary_size_experiments')
vocabulary_sizes  = [6, 12, 24, 48, 90]
vocab_distributions = ['zipf', 'uniform']
grammar_templates = ['grammar_templates/grammar_2.template', 'grammar_templates/grammar_1.template']

num_samples_and_runs = [(1000, 8), (10000,4), (100000,2)]
num_trials = 10
num_samples_per_length=2000

# # Validation Experiments
grammar_vocab_size_and_dist = [x for x in itertools.product(grammar_templates, vocabulary_sizes, vocab_distributions)]

@run_on_cluster(job_name='vocabulary_size_experiments', 
                job_id=experiment_id,
                conda_env='quant_exp', gpu=1,
                walltime="18:00:00")
def vocabulary_size_experiments(main_args, serialization_dir, param_path):
    # Setup variables needed later.
    vocab_size_exp_results = {}
    for grammar_template, size, dist in grammar_vocab_size_and_dist:
        vocab_size_exp_results[f'{grammar_template}_{dist}_{size}'] = []
        
    orig_serialization_dir = serialization_dir
    for grammar_template, size, dist in grammar_vocab_size_and_dist:
        vsexp_serialization_dir = os.path.join(orig_serialization_dir, 'vocabulary_size_experiments', f'{dist}_{size}')
        grammar_string = ArtificialLanguageOracle.generate_grammar_string(grammar_template_file=grammar_template,
                                                                            vocabulary_size=size, 
                                                                            vocabulary_distribution=dist)
        os.makedirs(vsexp_serialization_dir, exist_ok=True)
        grammar_filename = os.path.join(vsexp_serialization_dir, 'grammar.txt')
        with open(grammar_filename, 'w') as f:
            f.write(grammar_string)
        
        overrides = json.dumps({'model':{
                                    'oracle': {
                                        'grammar_file': grammar_filename}}})
        for num_samples, num_runs in num_samples_and_runs:
            for run in range(num_runs):
                serialization_dir = os.path.join(vsexp_serialization_dir, str(num_samples), str(run))
                sample_oracle_args = get_args(args=['sample-oracle', param_path, '-s', serialization_dir, '-o', overrides, '-n', f'{num_samples}'])
                oracle_train_filename, oracle_dev_filename = sample_oracle_runner(sample_oracle_args, 
                                                                                serialization_dir);

                os.environ['ARTIFICIAL_GRAMMAR_TRAIN'] = oracle_train_filename
                os.environ['ARTIFICIAL_GRAMMAR_DEV'] = oracle_dev_filename

                train_args = get_args(args=['train' , param_path, '-s', serialization_dir, '-o', overrides])
                trainer_params = Params.from_file(train_args.param_path, train_args.overrides)
                cuda_device = trainer_params['trainer']['cuda_device']
                train_vocab_serialization_dir = train_runner(train_args, 
                                                            serialization_dir);

                archive_file = os.path.join(train_vocab_serialization_dir, 'model.tar.gz')
                qeb_output_dir = os.path.join(serialization_dir, 'exp_bias')
                
                metrics = json.load(open(os.path.join(train_vocab_serialization_dir, f'metrics.json')))
                qeb_args = get_args(args=['quantify-exposure-bias', archive_file, '--output-dir', qeb_output_dir, '-o', overrides])
                exp_biases, exp_bias_mean, exp_bias_std = quantify_exposure_bias_runner(qeb_args, 
                                                                                        archive_file,
                                                                                        qeb_output_dir,
                                                                                        cuda_device=cuda_device, 
                                                                                        num_trials=num_trials,
                                                                                        num_samples_per_length=num_samples_per_length)
                result = {
                            'exp_biases': exp_biases,
                            'exp_bias_mean': exp_bias_mean,
                            'exp_bias_std': exp_bias_std,
                            'num_run': run,
                            'num_samples': num_samples,
                            'distribution': dist,
                            'vocab_size': size,
                            'val_ppl': metrics['best_validation_perplexity'],
                            'best_val_epoch': metrics['best_epoch']
                        }
                vocab_size_exp_results[f'{dist}_{size}'].append(result)
                wandb.log(result)
    return vocab_size_exp_results

dataset_exp_results = vocabulary_size_experiments(main_args, serialization_dir, param_path)

result_path = os.path.join(serialization_dir, 'vocabulary_size_experiments.json')
with open(result_path, 'w') as f:
    json.dump(dataset_exp_results, f, indent=4, sort_keys=True)
print(result_path)
