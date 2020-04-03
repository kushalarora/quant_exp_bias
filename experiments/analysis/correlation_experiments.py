# coding: utf-8

import os
import sys

from matplotlib import pyplot as plt

from scipy.stats import spearmanr

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

import argparse
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
parser.add_argument('--exp_msg', type=str, default=None, help='Debug(maybe) experiment message.')
parser.add_argument('--exp_dir', type=str, default=None, required=True, help='Experiment directory')
args = parser.parse_args()

# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id, experiment = initialize_experiments('correlation_experiments',
                                                                                 debug=args.debug,
                                                                                 experiment_text=args.exp_msg,
                                                                                )

experiment.log_parameters({'serialization_dir': serialization_dir,
                          'main_args': main_args,
                          'experiment_id': experiment_id})

def correlation_experiments(main_args,
                            serialization_dir,
                            param_path,
                           ):
    step = 0

    exp_bias_means = []
    H_m_m_means = []
    H_m_o_means = []
    H_o_m_means = []
    H_o_o_means = []
    ppls = []
    run_serialization_dir = args.exp_dir

    def validation_exp_bias_epochs_func(train_model_serialization_dir):
        for epoch in range(len(glob.glob(os.path.join(train_model_serialization_dir + '/model_state_epoch_*.th')))):
            qeb_suffix = f'epoch_{epoch}'
            metrics_filename = f'metrics_epoch_{epoch}.json'
            yield (epoch, qeb_suffix, metrics_filename)

    run_metrics_list = one_exp_run(only_quantify=True,
                                    run_serialization_dir=run_serialization_dir,
                                    train_model_serialization_dir=os.path.join(run_serialization_dir, 'training'),
                                    oracle_test_filename=os.path.join(run_serialization_dir, 'data/oracle_samples-test.txt'),
                                    exp_bias_epochs_func=validation_exp_bias_epochs_func,
                                    num_trials=5,
                                    num_length_samples=1,
                                    num_samples_per_length=32,
                                    )

    for run_metrics in run_metrics_list:
        epoch = run_metrics['epoch']

        experiment.log_metric('epoch', epoch, step=step)
        experiment.log_metric('exp_bias_mean', run_metrics['exp_bias_mean'], step=step)
        exp_bias_means.append(run_metrics['exp_bias_mean'])

        ppls.append(run_metrics['validation_perplexity'])


        experiment.log_metric('H_m_m_mean', run_metrics['H_m_m_mean'], step=step)
        H_m_m_means.append(run_metrics['H_m_m_mean'])

        experiment.log_metric('H_m_o_mean', run_metrics['H_m_o_mean'], step=step)
        H_m_o_means.append(run_metrics['H_m_o_mean'])

        experiment.log_metric('H_o_m_mean', run_metrics['H_o_m_mean'], step=step)
        H_o_m_means.append(run_metrics['H_o_m_mean'])

        experiment.log_metric('H_o_o_mean', run_metrics['H_o_o_mean'], step=step)
        H_o_o_means.append(run_metrics['H_o_o_mean'])

        experiment.log_metric('df_p_q_mean', run_metrics['df_p_q_mean'], step=step)
        experiment.log_metric('df_q_p_mean', run_metrics['df_q_p_mean'], step=step)

        step += 1

    import pdb;pdb.set_trace()
    exp_bias_hmm_corr = spearmanr(exp_bias_means, H_m_m_means)
    experiment.log_metric('exp_bias_hmm_corr', exp_bias_hmm_corr.correlation)
    print(f'exp_bias_hmm_corr: {exp_bias_hmm_corr}')

    exp_bias_hmo_corr = spearmanr(exp_bias_means, H_m_o_means)
    experiment.log_metric('exp_bias_hmo_corr', exp_bias_hmo_corr.correlation)
    print(f'exp_bias_hmo_corr: {exp_bias_hmo_corr}')

    exp_bias_hom_corr = spearmanr(exp_bias_means, H_o_m_means)
    experiment.log_metric('exp_bias_hom_corr', exp_bias_hom_corr.correlation)
    print(f'exp_bias_hom_corr: {exp_bias_hom_corr}')

    exp_bias_hoo_corr = spearmanr(exp_bias_means, H_o_o_means)
    experiment.log_metric('ppl_hoo_corr', exp_bias_hoo_corr.correlation)
    print(f'exp_bias_hoo_corr: {exp_bias_hoo_corr}')

    ppl_hmm_corr = spearmanr(ppls, H_m_m_means)
    experiment.log_metric('ppl_hmm_corr', ppl_hmm_corr.correlation)
    print(f'ppl_hmm_corr: {ppl_hmm_corr}')

    ppl_hmo_corr = spearmanr(ppls, H_m_o_means)
    experiment.log_metric('ppl_hmo_corr', ppl_hmo_corr.correlation)
    print(f'ppl_hmo_corr: {ppl_hmo_corr}')

    ppl_hom_corr = spearmanr(ppls, H_o_m_means)
    experiment.log_metric('ppl_hom_corr', ppl_hom_corr.correlation)
    print(f'ppl_hom_corr: {ppl_hom_corr}')

    ppl_hoo_corr = spearmanr(ppls, H_o_o_means)
    experiment.log_metric('ppl_hoo_corr', ppl_hoo_corr.correlation)
    print(f'ppl_hoo_corr: {ppl_hoo_corr}')

correlation_experiments(main_args, serialization_dir, param_path)
