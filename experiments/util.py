import itertools
import json
import math
import numpy as np
import os
import sys

from datetime import datetime
from random import randint
from time import sleep

from allennlp.common.util import import_submodules
import_submodules("quant_exp_bias")

from comet_ml import Experiment, OfflineExperiment
from typing import Dict, List, Callable, Tuple, Union, Any

from allennlp.common import Params

from quant_exp_bias.utils import (get_args, quantify_exposure_bias_runner,
                                  sample_oracle_runner, train_runner)

from quant_exp_bias.oracles.artificial_grammar_oracle import ArtificialLanguageOracle
from quant_exp_bias.utils import get_args


OverrideFuncType = Callable[[], Dict[str, Union[float, str, int]]]
ExpBiasEpochsFuncType = Callable[[str], List[Tuple[int, str, str]]]


def generate_grammar_file(serialization_dir: str,
                          grammar_template: str='grammar_templates/grammar_2.template',
                          vocabulary_size: int=6,
                          vocabulary_distribution: str='zipf',
                          epsilon:float = 0,
                          cost_func_grammar:bool = False,
                          ):
    grammar_string = ArtificialLanguageOracle.generate_grammar_string(grammar_template_file=grammar_template,
                                                                      vocabulary_size=vocabulary_size,
                                                                      vocabulary_distribution=vocabulary_distribution,
                                                                      epsilon=epsilon,
                                                                      )
    os.makedirs(serialization_dir, exist_ok=True)
    grammar_filename = os.path.join(serialization_dir, f'epsilon_{epsilon}_grammar.txt')
    with open(grammar_filename, 'w') as f:
        f.write(grammar_string)
    
    os.environ["FSA_GRAMMAR_FILENAME"] = grammar_filename
    if cost_func_grammar:
        os.environ["FSA_GRAMMAR_FILENAME_COST_FUNC"] = grammar_filename
    return grammar_filename


def initialize_experiments(experiment_name: str,
                           output_dir: str = None,
                           param_path: str = None,
                           debug: bool = False,
                           offline: bool = False,
                           experiment_text: str = None,
                           ):
    # Ipython by default adds some arguments to sys.argv.
    #  We don't want those arguments, hence we pass [] here.
    #
    # The deafult argument get_args is args=None.
    # This translates to parsing sys.argv. This is useful
    # in case we run the method from a python file but not here.
    # Hence, we keep the default argument as None but pass [] for
    # ipython notebook.
    main_args = get_args(args=[])

    experiment_id = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    serialization_dir = os.path.join(output_dir or main_args.output_dir, experiment_name, experiment_id)
    param_path = param_path or main_args.config

    os.makedirs(serialization_dir, exist_ok=True)

    os.environ['TRAIN_FILE'] = ""
    os.environ['DEV_FILE'] = ""
    
    if debug:
        experiment_name += '_debug'
    
    workspace_name = 'quantifying_exposure_bias'
    try:
        if offline:
            raise ValueError
        experiment = Experiment(api_key='2UIhYs7jRdE2DbJDAB5OysNqM',
                                workspace=workspace_name,
                                project_name=experiment_name,
                                auto_metric_logging=False,
                                auto_param_logging=False,
                                )
    except:
        experiment = OfflineExperiment(
            workspace=workspace_name,
            project_name=experiment_name,
            auto_metric_logging=False,
            auto_param_logging=False,
            offline_directory="./comet_exp/",
        )
    
    if experiment_text:
        experiment.log_text(experiment_text)

    experiment.log_parameters({'serialization_dir': serialization_dir,
                                'main_args': main_args,
                                'param_path': param_path,
                                'experiment_id': experiment_id})
    return main_args, serialization_dir, param_path, experiment_id, experiment


def default_overides_func():
    return '{}'


def default_exp_bias_epochs_func(train_model_serialization_dir):
    epoch = -1
    qeb_suffix = ''
    metrics_filename = 'metrics.json'
    return [(epoch, qeb_suffix, metrics_filename)]


def one_exp_run(serialization_dir: str = None,
                num_samples: int = 10000,
                run: int = 0,
                param_path: str = None,
                cuda_device:int = 0,
                overides_func: OverrideFuncType = default_overides_func,
                exp_bias_epochs_func: ExpBiasEpochsFuncType = default_exp_bias_epochs_func,
                sample_from_file=False,
                dataset_filename=None,
                exp_bias_inference_funcs: List[Tuple[str, Any, OverrideFuncType]] = lambda: [],
                shall_generate_grammar_file: str = False,
                grammar_template: str='grammar_templates/grammar_2.template',
                vocabulary_size: int=6,
                vocabulary_distribution: str='zipf',
                num_trials: int = None,
                num_length_samples: int = None,
                num_samples_per_length: int = None,
                grammar_file_epsilon_0: str = None,
                grammar_file_epsilon: str = None,
                only_quantify: bool = False,
                run_serialization_dir:str = None,
                train_model_serialization_dir: str = None,
                oracle_test_filename:str = None,
                ):
    
    overrides = default_overides_func()
    run_serialization_dir = run_serialization_dir or \
                                os.path.join(serialization_dir, 
                                            str(num_samples), 
                                            str(run))
    if not only_quantify:
        # This is grammar with epsilon 0 to sample correct sequence.
        if shall_generate_grammar_file:
            generate_grammar_file(run_serialization_dir, grammar_template,
                                    vocabulary_size, vocabulary_distribution, 
                                    epsilon=0, cost_func_grammar=True)
        elif grammar_file_epsilon_0:
            os.environ["FSA_GRAMMAR_FILENAME"] = grammar_file_epsilon_0
            os.environ["FSA_GRAMMAR_FILENAME_COST_FUNC"] = grammar_file_epsilon_0

        overrides = overides_func()
        sample_oracle_args = ['sample-oracle',
                            param_path,
                            '-s', run_serialization_dir,
                            '-n', str(num_samples),
                            '-o',  overrides]

        # We might want to sample from file, for example, in cases,
        # where dataset is fixed. This is the case with natural language
        # experiments.
        if sample_from_file:
            sample_oracle_args += ['-f', dataset_filename]

        sample_oracle_args = get_args(args=sample_oracle_args)
        oracle_train_filename, oracle_dev_filename, oracle_test_filename = \
            sample_oracle_runner(sample_oracle_args,
                                run_serialization_dir)

        os.environ['TRAIN_FILE'] = oracle_train_filename
        os.environ['DEV_FILE'] = oracle_dev_filename

        # This is grammar with epsilon 1e-4 to smoothened probability distribution
        # so that we can assign some prob. to incorrect sequences.
        if shall_generate_grammar_file:
            generate_grammar_file(run_serialization_dir, grammar_template,
                                vocabulary_size, vocabulary_distribution, epsilon=1e-4)
        elif grammar_file_epsilon_0 or grammar_file_epsilon:
            os.environ["FSA_GRAMMAR_FILENAME"] = grammar_file_epsilon or grammar_file_epsilon_0

        train_args = get_args(args=['train',
                                    param_path,
                                    '-s', run_serialization_dir,
                                    '-o',  overrides])
        trainer_params = Params.from_file(train_args.param_path, train_args.overrides)

        train_model_serialization_dir = train_runner(train_args,
                                                    run_serialization_dir)

    archive_file = os.path.join(train_model_serialization_dir, 'model.tar.gz')
    metric_list = []

    for epoch, qeb_suffix, metric_filename in exp_bias_epochs_func(train_model_serialization_dir):
        qeb_output_dir = os.path.join(run_serialization_dir, 
                                      'exp_bias', 
                                      qeb_suffix)

        metrics = json.load(open(os.path.join(train_model_serialization_dir, 
                                              metric_filename)))

        # This is only needed when doing validation experiments.
        weights_file = None
        if epoch != -1:
            weights_file = os.path.join(train_model_serialization_dir, 
                                        f'model_state_epoch_{epoch}.th')

        qeb_args = get_args(args=['quantify-exposure-bias',
                                  archive_file,
                                  oracle_test_filename,
                                  '--output-dir', qeb_output_dir,
                                  '--weights-file', weights_file,
                                  '-o',  overrides])

        exp_biases, exp_bias_mean, exp_bias_std, \
            df_p_qs, df_p_q_mean, df_p_q_std, \
            df_q_ps, df_q_p_mean, df_q_p_std, \
            h_m_m_mean, h_m_m_std, h_m_o_mean, h_m_o_std, \
            h_o_m_mean, h_o_m_std, h_o_o_mean, h_o_o_std  = \
                quantify_exposure_bias_runner(qeb_args,
                                                archive_file,
                                                oracle_test_filename,
                                                qeb_output_dir,
                                                cuda_device=cuda_device,
                                                weights_file=weights_file,
                                                num_trials=num_trials,
                                                num_length_samples=num_length_samples,
                                                num_samples_per_length=num_samples_per_length)

        metrics['exp_biases'] = exp_biases
        metrics['exp_bias_mean'] = exp_bias_mean
        metrics['exp_bias_std'] = exp_bias_std

        metrics['df_p_qs'] = df_p_qs
        metrics['df_p_q_mean'] = df_p_q_mean
        metrics['df_p_q_std'] = df_p_q_std

        metrics['df_q_ps'] = df_q_ps
        metrics['df_q_p_mean'] = df_q_p_mean
        metrics['df_q_p_std'] = df_q_p_std

        metrics['H_m_m_mean'] = h_m_m_mean
        metrics['H_m_m_std'] =  h_m_m_std

        metrics['H_m_o_mean'] = h_m_o_mean
        metrics['H_m_o_std'] = h_m_o_std

        metrics['H_o_m_mean'] = h_o_m_mean
        metrics['H_o_m_std'] = h_o_m_std

        metrics['H_o_o_mean'] = h_o_o_mean
        metrics['H_o_o_std'] = h_o_o_std

        metric_list.append(metrics)

    for key, value, qeb_overides in exp_bias_inference_funcs():
        metrics = json.load(open(os.path.join(train_model_serialization_dir, 
                                              metric_filename)))

        qeb_suffix = f"{key}_{value}"
        qeb_output_dir = os.path.join(run_serialization_dir, 
                                      'exp_bias', 
                                      qeb_suffix)

        qeb_args = get_args(args=['quantify-exposure-bias',
                                  archive_file,
                                  oracle_test_filename,
                                  '--output-dir', qeb_output_dir,
                                  '-o', qeb_overides])

        exp_biases, exp_bias_mean, exp_bias_std, \
            df_p_qs, df_p_q_mean, df_p_q_std, \
            df_q_ps, df_q_p_mean, df_q_p_std, \
            h_m_m_mean, h_m_m_std, h_m_o_mean, h_m_o_std, \
            h_o_m_mean, h_o_m_std, h_o_o_mean, h_o_o_std  = \
                quantify_exposure_bias_runner(qeb_args,
                                                archive_file,
                                                oracle_test_filename,
                                                qeb_output_dir,
                                                cuda_device=cuda_device,
                                                num_trials=num_trials,
                                                num_length_samples=num_length_samples,
                                                num_samples_per_length=num_samples_per_length)

        metrics['exp_biases'] = exp_biases
        metrics['exp_bias_mean'] = exp_bias_mean
        metrics['exp_bias_std'] = exp_bias_std

        metrics['df_p_qs'] = df_p_qs
        metrics['df_p_q_mean'] = df_p_q_mean
        metrics['df_p_q_std'] = df_p_q_std

        metrics['df_q_ps'] = df_q_ps
        metrics['df_q_p_mean'] = df_q_p_mean
        metrics['df_q_p_std'] = df_q_p_std

        metrics['H_m_m_mean'] = h_m_m_mean
        metrics['H_m_m_std'] =  h_m_m_std

        metrics['H_m_o_mean'] = h_m_o_mean
        metrics['H_m_o_std'] = h_m_o_std

        metrics['H_o_m_mean'] = h_o_m_mean
        metrics['H_o_m_std'] = h_o_m_std
        
        metrics['H_o_o_mean'] = h_o_o_mean
        metrics['H_o_o_std'] = h_o_o_std

        metrics[key] = value
        metric_list.append(metrics)
    return metric_list

def get_experiment_args(experiment_type: str = 'artificial_language', 
             experiment_name: str = 'dataset_experiments'):
    
    import argparse
    parser = argparse.ArgumentParser(description=f'{experiment_type}/{experiment_name}.')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of dataset samples to run this iteration for.')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs for the given dataset size.')
    parser.add_argument('--all', action='store_true', help='Run All configurations mentioned below..')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
    parser.add_argument('--offline', action='store_true', help='Run in offline mode.')
    parser.add_argument('--exp_msg', type=str, default=None, help='Debug(maybe) experiment message.')
    parser.add_argument('--output_dir', '-o', type=str, default=os.path.expanduser('~/scratch/quant_exp_bias/'), help='Output directory.')
    
    if experiment_type == 'artificial_language':
        parser.add_argument('--vocab_distributions', nargs='+', type=str, default=['zipf', 'uniform'], 
                                help='Distributions to use.')
        parser.add_argument('--grammar_templates', nargs='+',type=str, 
                                default=['grammar_1', 'grammar_2', 'grammar_3'], 
                                help='Grammar templates to use in this experiment')

    if experiment_name == 'model_size_experiments':
        parser.add_argument('--model_sizes', nargs='+', type=str, 
                                 default=['xsmall', 'small', 'medium', 'large', 'xlarge'],
                                 help='Model sizes to consider')

    if experiment_name == 'scheduled_sampling_experiments':
        default_num_epochs = 50; default_batch_size = 128
        if experiment_type == 'natural_language':
            default_num_epochs = 20; default_batch_size = 4
        parser.add_argument('--num_epochs', type=int, default=default_num_epochs,
                                help='Number of batches for the experiment.')

        parser.add_argument('--batch_size', type=int, default=default_batch_size,
                                help='Batch size for this experiment.')
    
    if experiment_name == 'searnn_experiments' or \
            experiment_name == 'searnn_ablation_experiments':
        parser.add_argument('--rollins', nargs='+', type=str,
                                 default=['teacher_forcing', 'mixed', 'learned'],
                                help='Rollins to use')

        parser.add_argument('--rollouts', nargs='+', type=str, 
                                default=['reference', 'mixed', 'learned'], 
                                help='Rollouts to use')
        parser.add_argument('--temperature', type=float, default=1.0,
                            help='temperature for SEARNN experiments')

    if experiment_name == 'vocabulary_experiments':
        parser.add_argument('--vocabulary_sizes', nargs='+', type=int, default=[6, 12, 24, 48],
                                help='Vocabulary Sizes to run.')

    if experiment_name == 'searnn_ablation_experiments' or \
        experiment_name == 'reinforce_ablation_experiments':
        parser.add_argument('--rollout_cost_funcs', nargs='+', type=str, 
                                default=['noisy_oracle', 'bleu'], 
                                help='Type of Oracle to use')

        parser.add_argument('--mixing_coeff', nargs='+', type=float, default=[0, 0.25, 0.5,],
                                help='Mixing coefficients for rollin and rollouts')

    if experiment_name == 'scheduled_sampling_ablation_experiments':
        parser.add_argument('--batch_size', type=int, 
                                default=96, help='Scheduled Sampling experiment batch size for natural language experiments.')

        parser.add_argument('--num_epochs', type=int, 
                                default=20, help='Scheduled Sampling experiment batch size for natural language experiments.')
    return parser.parse_args()

def calculate_ss_k(num_samples, batch_size, num_epochs, ratio_level=0.5):
    high = 20000; low = 1
    num_iteration_per_batch = num_samples/batch_size
    iteration_to_ratio = num_iteration_per_batch * int((num_epochs + 1)*ratio_level)
    k_func = lambda k: k/(k + math.exp(iteration_to_ratio/k))
    while low <= high:
        mid = (high + low)//2
        k_mid = k_func(mid)
        if k_mid > 0.505:
            high = mid
        elif k_mid < 0.495:
            low = mid
        else:
            return mid
    return -1

def get_grammar_template_path(grammar_template: str):
    return {
        'grammar_1': ('grammar_templates/grammar_1.template', True),
        'grammar_2': ('grammar_templates/grammar_2.template', True),
        'grammar_3': ('grammar_templates/grammar_3.template', True),
    }[grammar_template]

def get_grammar_iterator(experiment: Union[Experiment, OfflineExperiment], 
                            grammar_templates: List[str], 
                            vocab_distributions: List[str], 
                            num_runs: int):
    grammars_and_vocabularies = [x for x in itertools.product(grammar_templates, vocab_distributions)]
    num_grammars = len(grammars_and_vocabularies)
    for num_run in range(num_runs):
        for i, (grammar_template, vocab_dist) in enumerate(grammars_and_vocabularies):
            grammar_template_file, shall_generate_grammar_file = \
                                            get_grammar_template_path(grammar_template)
            
            params = {
                    'grammar_template': grammar_template,
                    'vocab_distribution': vocab_dist,
                    'grammar_template_file': grammar_template_file,
                    'shall_generate_grammar_file': shall_generate_grammar_file,
                    }
            yield (num_run * num_grammars + i, 
                    grammar_template_file, 
                    vocab_dist, 
                    shall_generate_grammar_file, 
                    params)

def get_result_iterator(run_metrics: Dict[str, Any]):
    for exp_bias_idx, (exp_bias, df_p_q, df_q_p) in enumerate(zip(run_metrics['exp_biases'],
                                                                      run_metrics['df_p_qs'],
                                                                      run_metrics['df_q_ps'])):
            sleep(randint(1, 10)/100.0)
            yield {
                'exp_bias': exp_bias,
                'Df_p_q': df_p_q,
                'Df_q_p': df_q_p,
                'exp_bias_idx': exp_bias_idx,
            }

def get_mean_std_results(num_run:int,
                         num_samples:int,
                         run_metrics: Dict[str, Any]):
    return {
        'num_run': num_run,
        'num_samples': num_samples,
        'val_ppl': run_metrics['best_validation_perplexity'],
        'best_val_epoch': run_metrics['best_epoch'],
        'exp_bias_mean': run_metrics['exp_bias_mean'],
        'exp_bias_std': run_metrics['exp_bias_std'],
        'df_p_q_mean': run_metrics['df_p_q_mean'],
        'df_p_q_std': run_metrics['df_p_q_std'],
        'df_q_p_mean': run_metrics['df_q_p_mean'],
        'df_q_p_std': run_metrics['df_q_p_std'],
        'H_m_m_mean': run_metrics['H_m_m_mean'],
        'H_m_m_std': run_metrics['H_m_m_std'],
        'H_m_o_mean': run_metrics['H_m_o_mean'],
        'H_m_o_std': run_metrics['H_m_o_std'],
        'H_o_m_mean': run_metrics['H_o_m_mean'],
        'H_o_m_std': run_metrics['H_o_m_std'],
        'H_o_o_mean': run_metrics['H_o_o_mean'],
        'H_o_o_std': run_metrics['H_o_o_std'],
    }

def get_model_overrides_func(embed_dim: int, hidden_dim: int, num_layers: int):
    return lambda: json.dumps({
        'model':{
            'decoder': {
                'decoder_net': {
                    'target_embedding_dim': embed_dim,
                    'decoding_dim': hidden_dim,
                    'num_decoder_layers': num_layers,
                },
                'target_embedder': {
                    'embedding_dim': embed_dim,
                }
            }
        }
    })

def get_scheduled_sampling_overrides_func(ss_type:str, ss_ratio:float, ss_k:int):
    return lambda: json.dumps({'model': {
                                    'decoder': {
                                        'scheduled_sampling_type': ss_type,
                                        'scheduled_sampling_ratio': ss_ratio,
                                        'scheduled_sampling_k': ss_k,
                                    },
                                }
                            })

def get_rollout_cost_function_configs(experiment_type, cost_func, mixing_coeff, temperature):
    if cost_func == 'bleu':
        rollout_cost_func_dict = { "type": "bleu",}
        temperature = 100
    elif cost_func == 'noisy_oracle':
        if experiment_type == 'artificial_language':
            oracle = {
                    "type": "artificial_lang_oracle",
                    "grammar_file":  os.environ["FSA_GRAMMAR_FILENAME_COST_FUNC"],
                }
            temperature = 1
        elif experiment_type == 'natural_language':
            oracle = {
                    "type": "gpt2_oracle",
                    "model_name": "distilgpt2",
                    "batch_size": 4,
                    "cuda_device": -2,
                }
            temperature = temperature
        rollout_cost_func_dict = {
          "type": "noisy_oracle",
          "oracle": oracle,
        }
    overrides_dict = {
        "model": {
            "decoder": {
                "rollout_cost_function": rollout_cost_func_dict,
                "rollin_rollout_mixing_coeff": mixing_coeff,
                "temperature": temperature
            }, 
        }
    }
    return lambda: json.dumps(overrides_dict)
