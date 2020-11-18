import glob
import itertools
import json
import math
import numpy as np
import os
import random
import re
import sys
import uuid
import subprocess
from comet_ml import Experiment, OfflineExperiment

from datetime import datetime
from random import randint
from time import sleep

from allennlp.common.util import import_module_and_submodules

from typing import Dict, List, Callable, Tuple, Union, Any

from allennlp.common import Params

from quant_exp_bias.utils import (get_args, quantify_exposure_bias_runner,
                                  sample_oracle_runner, train_runner)

from quant_exp_bias.oracles.artificial_grammar_oracle import ArtificialLanguageOracle


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
    # Import LMPL library as plugin. 
    import_module_and_submodules("quant_exp_bias")
    import_module_and_submodules("lmpl")

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
        random.seed(220488)

    workspace_name = 'qeb'

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

def last_exp_bias_epoch_func(train_model_serialization_dir):
    epoch_files = glob.glob(os.path.join(train_model_serialization_dir + '/model_state_epoch_*.th'))
    epochs = sorted([int(re.search('epoch_([0-9]+).th', fname).group(1)) for fname in epoch_files])
    epoch = epochs[-1]
    metrics_filename = f'metrics_epoch_{epoch}.json'
    return [(-1, '', metrics_filename)]


def shard_file(filename: str):
    head, tail = os.path.split(filename)
    tail_prefix, tail_suffix = tail.split('.')

    shards = []
    for i in range(1,5):
        shard_filename =os.path.join(head, f'{tail_prefix}_{i}.{tail_suffix}')
        shards.append(open(shard_filename, 'w'))

    with open(filename) as f:
        for i,line in enumerate(f):
            shard_idx = 3
            if  (i + 3) % 4 == 0:
                shard_idx = 0
            elif (i + 2) % 4 == 0:
                shard_idx = 1
            elif (i + 1) % 4 == 0:
                shard_idx = 2
            shards[shard_idx].write(line)
    for shard_file in shards:
        shard_file.close()
    return os.path.join(head, f'{tail_prefix}_*.{tail_suffix}')

def generate_dataset(run_serialization_dir: str,
                     oracle_config: str = 'experiments/artificial_language/training_configs/artificial_grammar_oracle.jsonnet',
                     shall_generate_grammar_file: str = False,
                     grammar_template: str='grammar_templates/grammar_2.template',
                     vocabulary_size: int=6,
                     vocabulary_distribution: str='zipf',
                     num_samples: int = 10000,
                     grammar_file_epsilon_0: str = None,
                     grammar_file_epsilon: str = None,
                     sample_from_file: bool = False,
                     dataset_filename: str = None,
                    ):
    # This is grammar with epsilon 0 to sample correct sequence.
    if shall_generate_grammar_file:
        generate_grammar_file(run_serialization_dir, grammar_template,
                                vocabulary_size, vocabulary_distribution, 
                                epsilon=0, cost_func_grammar=True)
    elif grammar_file_epsilon_0:
        os.environ["FSA_GRAMMAR_FILENAME"] = grammar_file_epsilon_0
        os.environ["FSA_GRAMMAR_FILENAME_COST_FUNC"] = grammar_file_epsilon_0

    # We might want to sample from file, for example, in cases,
    # where dataset is fixed. This is the case with natural language
    # experiments.
    sample_oracle_arg_list = ['sample-oracle',
                            oracle_config,
                            '-s', run_serialization_dir,
                            '-n', str(num_samples)]

    if sample_from_file:
        sample_oracle_arg_list += ['-f', dataset_filename]

    sample_oracle_args = get_args(args=sample_oracle_arg_list)
    oracle_train_filename, oracle_dev_filename, _ = \
        sample_oracle_runner(sample_oracle_args,
                            run_serialization_dir 
                            )

    oracle_train_filename = shard_file(oracle_train_filename)
    oracle_dev_filename = shard_file(oracle_dev_filename)

    os.environ['TRAIN_FILE'] = oracle_train_filename
    os.environ['DEV_FILE'] = oracle_dev_filename

    # This is grammar with epsilon 1e-4 to smoothened probability distribution
    # so that we can assign some prob. to incorrect sequences.
    if shall_generate_grammar_file:
        generate_grammar_file(run_serialization_dir, grammar_template,
                            vocabulary_size, vocabulary_distribution, epsilon=1e-4)
    elif grammar_file_epsilon_0 or grammar_file_epsilon:
        os.environ["FSA_GRAMMAR_FILENAME"] = grammar_file_epsilon or grammar_file_epsilon_0

    return oracle_train_filename, oracle_dev_filename

def quantify_exposure_bias(metric_filepath: str,
                            archive_file: str,
                            oracle_config: str,
                            output_dir: str,
                            cuda_device: int = 0,
                            weights_file: str = None,
                            opt_level: str = "O2",
                            num_trials: int = None,
                            num_length_samples: int = None,
                            num_samples_per_length: int = None,
                            overrides: str = None,
                          ):
    overrides = overrides or default_overides_func()

    metrics = json.load(open(metric_filepath))
    args=['quantify-exposure-bias',
                                archive_file,
                                '--oracle-config', oracle_config,
                                '--output-dir', output_dir]
    if weights_file:
        args += ['--weights-file', weights_file]
    
    if overrides:
        args += ['-o', overrides]

    qeb_args = get_args(args=args)

    exp_biases, exp_bias_mean, exp_bias_std, \
        df_p_qs, df_p_q_mean, df_p_q_std, \
        h_m_m_mean, h_m_m_std, h_m_o_mean, h_m_o_std = \
            quantify_exposure_bias_runner(qeb_args,
                                    archive_file,
                                    oracle_config,
                                    output_dir,
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

    # metrics['df_q_ps'] = df_q_ps
    # metrics['df_q_p_mean'] = df_q_p_mean
    # metrics['df_q_p_std'] = df_q_p_std

    metrics['H_m_m_mean'] = h_m_m_mean
    metrics['H_m_m_std'] =  h_m_m_std

    metrics['H_m_o_mean'] = h_m_o_mean
    metrics['H_m_o_std'] = h_m_o_std

    # metrics['H_o_m_mean'] = h_o_m_mean
    # metrics['H_o_m_std'] = h_o_m_std
    
    # metrics['H_o_o_mean'] = h_o_o_mean
    # metrics['H_o_o_std'] = h_o_o_std

    return metrics

def one_exp_run(serialization_dir: str = None,
                num_samples: int = 10000,
                run: int = 0,
                param_path: str = None,
                oracle_config: str = None,
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
                run_serialization_dir: str = None,
                train_model_serialization_dir: str = None,
                oracle_train_filename: str = None, 
                oracle_dev_filename: str = None,
                recover: bool = False,
                donot_quantify: bool = False,
            ):
    overrides = default_overides_func()

    if only_quantify:
        train_model_serialization_dir = os.path.join(run_serialization_dir, 'training')
        # Doing this as the command might not have completed and metrics.json might not exist.
        exp_bias_epochs_func: ExpBiasEpochsFuncType = last_exp_bias_epoch_func
        archive_file = train_model_serialization_dir
    else:
        if recover:
            param_path = os.path.join(run_serialization_dir, 'training/config.json')
        else:
            # UUID adds a random id at the end in case two or more runs start at the same time.
            run_serialization_dir = os.path.join(serialization_dir, 
                                                    str(num_samples), 
                                                    str(run), 
                                                    str(uuid.uuid4().fields[0])) 
            if oracle_dev_filename is None and oracle_dev_filename is None:
                oracle_train_filename, oracle_dev_filename = \
                    generate_dataset(run_serialization_dir=run_serialization_dir,
                                        oracle_config=oracle_config,
                                        shall_generate_grammar_file=shall_generate_grammar_file,
                                        grammar_template=grammar_template,
                                        vocabulary_size=vocabulary_size,
                                        vocabulary_distribution=vocabulary_distribution,
                                        num_samples=num_samples,
                                        grammar_file_epsilon_0=grammar_file_epsilon_0,
                                        grammar_file_epsilon=grammar_file_epsilon,
                                        sample_from_file=sample_from_file,
                                        dataset_filename=dataset_filename,
                                    )
        overrides = overides_func()
        if os.environ["DISTRIBUTED"]== "true":
            from multiprocessing import freeze_support
            freeze_support()

            train_model_serialization_dir =  os.path.join(run_serialization_dir, 'training')
            train_cmd = ["allennlp", "train", 
                                param_path, '-s', train_model_serialization_dir,
                                '-o',  overrides,
                                '--include-package', 'lmpl',
                                '--include-package', 'quant_exp_bias']
            if recover:
                train_cmd += ['--recover']

            subprocess.run(train_cmd)
        else:
            train_args = get_args(args=['train',
                                        param_path,
                                        '-s', run_serialization_dir,
                                        '-o',  overrides])
            trainer_params = Params.from_file(train_args.param_path, train_args.overrides)

            train_model_serialization_dir = train_runner(train_args,
                                                        run_serialization_dir, 
                                                        recover=recover)

        archive_file = os.path.join(train_model_serialization_dir, 'model.tar.gz')

    metric_list = []
    if not donot_quantify:
        # This is only needed when doing validation experiments.
        for epoch, qeb_suffix, metric_filename in exp_bias_epochs_func(train_model_serialization_dir):
            qeb_output_dir = os.path.join(run_serialization_dir, 
                                        'exp_bias', 
                                        qeb_suffix)

            metrics_filepath = os.path.join(train_model_serialization_dir, 
                                                metric_filename)
            weights_file = None
            if epoch != -1:
                weights_file = os.path.join(train_model_serialization_dir, 
                                            f'model_state_epoch_{epoch}.th')

            metrics = quantify_exposure_bias(metric_filepath=metrics_filepath,
                                            archive_file=archive_file, 
                                            oracle_config=oracle_config,
                                            output_dir=qeb_output_dir,
                                            cuda_device=cuda_device,
                                            weights_file=weights_file,
                                            num_trials=num_trials,
                                            num_length_samples=num_length_samples,
                                            num_samples_per_length=num_samples_per_length)
            metrics['run_serialization_dir'] = run_serialization_dir
            metric_list.append(metrics)

        for key, value, qeb_overides in exp_bias_inference_funcs():
            qeb_suffix = f"{key}_{value}"
            qeb_output_dir = os.path.join(run_serialization_dir, 
                                        'exp_bias', 
                                        qeb_suffix)
            metric_filepath = os.path.join(train_model_serialization_dir, 
                                                    metric_filename)

            metrics = quantify_exposure_bias(metric_filepath=metrics_filepath,
                                            archive_file=archive_file, 
                                            oracle_config=oracle_config,
                                            output_dir=qeb_output_dir,
                                            cuda_device=cuda_device,
                                            weights_file=weights_file,
                                            num_trials=num_trials,
                                            num_length_samples=num_length_samples,
                                            num_samples_per_length=num_samples_per_length, 
                                            overrides=qeb_overides)
            metrics[key] = value
            metrics['run_serialization_dir'] = run_serialization_dir
            metric_list.append(metrics)
    return metric_list, run_serialization_dir

def get_experiment_args(experiment_type: str = 'artificial_language', 
                        experiment_name: str = 'dataset_experiments', 
                        args: List[str] = None):
    
    import argparse
    parser = argparse.ArgumentParser(description=f'{experiment_type}/{experiment_name}.')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of dataset samples to run this iteration for.')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs for the given dataset size.')
    parser.add_argument('--all', action='store_true', help='Run All configurations mentioned below..')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
    parser.add_argument('--offline', action='store_true', help='Run in offline mode.')
    parser.add_argument('--exp_msg', type=str, default=None, help='Debug(maybe) experiment message.')
    parser.add_argument('--only_quantify', action='store_true', help='Run in debug mode.')
    parser.add_argument('--recover', action='store_true', help='Recover the run.')
    parser.add_argument('--error_acc', action='store_true', help='Error Accumulation Experiments.')
    parser.add_argument('--output_dir', '-o', type=str, default=os.path.expanduser('~/scratch/quant_exp_bias/'), help='Output directory.')
    parser.add_argument('--run_serialization_dir', type=str, default=None, help='Specify run serialization dir if only quantifying.')

    if experiment_type == 'artificial_language':
        parser.add_argument('--oracle_config', type=str, 
                                default='experiments/artificial_language/training_configs/artificial_grammar_oracle.jsonnet',
                                help='Number of batches for the experiment.')
    elif experiment_type == 'natural_language':
            parser.add_argument('--oracle_config', type=str, 
                                default='experiments/natural_language/training_configs/gpt2_oracle.jsonnet',
                                help='Number of batches for the experiment.')
    else:
        raise ValueError(f"experiment_type can either be ['artificial_language', 'natural_language'], fount {experiment_type}")
    
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

    if experiment_name == 'scheduled_sampling_experiments' or \
            experiment_name == 'scheduled_sampling_ablation_experiments':
        default_num_epochs = 50; default_batch_size = 128
        if experiment_type == 'natural_language':
            default_num_epochs = 20; default_batch_size = 96
        parser.add_argument('--num_epochs', type=int, default=default_num_epochs,
                                help='Number of batches for the experiment.')

        parser.add_argument('--batch_size', type=int, default=default_batch_size,
                                help='Batch size for this experiment.')
            
        parser.add_argument('--ss_configs', nargs='+', type=str, 
                                default=['u_0.05', 'u_0.10', 'u_0.25',
                                         'linear', 'exponential', 'inverse_sigmoid'],
                                help='Scheduled Sampling configs to try.')

    if experiment_name == 'searnn_experiments' or \
            experiment_name == 'searnn_ablation_experiments':
        parser.add_argument('--rollins', nargs='+', type=str,
                                 default=['teacher_forcing', 'mixed', 'learned'],
                                help='Rollins to use')

        parser.add_argument('--rollouts', nargs='+', type=str, 
                                default=['reference', 'mixed', 'learned'], 
                                help='Rollouts to use')

    if experiment_name == 'vocabulary_experiments':
        parser.add_argument('--vocabulary_sizes', nargs='+', type=int, default=[6, 12, 24, 48],
                                help='Vocabulary Sizes to run.')

    if experiment_name == 'searnn_ablation_experiments' or \
        experiment_name == 'reinforce_ablation_experiments':
        parser.add_argument('--rollout_cost_funcs', nargs='+', type=str, 
                                default=['noisy_oracle'], 
                                help='Type of Oracle to use')
        parser.add_argument('--mixing_coeffs', nargs='+', type=float, default=[0, 0.25, 0.5,],
                                help='Mixing coefficients for rollin and rollouts')

    if experiment_name == 'searnn_ablation_experiments' or \
        experiment_name == 'searnn':
        parser.add_argument('--temperature', type=float, default=10.0,
                            help='temperature for SEARNN experiments')
        parser.add_argument('--neighbors', type=int, default=6,
                            help='Number of neighbors to add for SEARNN experiments')
    return parser.parse_args(args)

def calculate_ss_k(num_samples, batch_size, num_epochs, ss_type='exponential'):
    num_iteration_per_batch = num_samples/batch_size
    iteration_to_ratio = int(num_iteration_per_batch * num_epochs)
    if ss_type == 'exponential':
        k_func, high, low = (lambda k: 1 - (k/10_000_000)**(iteration_to_ratio//100), 10_000_000, 8_00_0000)
    elif ss_type == 'inverse_sigmoid':
        k_func, high, low = (lambda k: 1- k/(k + math.exp(iteration_to_ratio//(100 * k))), 10000, 1)
    elif ss_type == 'linear':
        return iteration_to_ratio
    else: raise ConfigurationError(f"SS Type not supported: {type}")

    while low < high:
        mid = (high + low)//2
        if mid == high or mid == low:
            return mid
            
        k_mid = k_func(mid)
        if k_mid > 0.01:
            low = mid
        elif k_mid <= 0.001:
            high = mid
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
    for exp_bias_idx, (exp_bias, df_p_q) in enumerate(zip(run_metrics['exp_biases'],
                                                          run_metrics['df_p_qs'])):
            sleep(randint(1, 10)/100.0)
            yield {
                'exp_bias': exp_bias,
                'Df_p_q': df_p_q,
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
        # 'df_q_p_mean': run_metrics['df_q_p_mean'],
        # 'df_q_p_std': run_metrics['df_q_p_std'],
        'H_m_m_mean': run_metrics['H_m_m_mean'],
        'H_m_m_std': run_metrics['H_m_m_std'],
        'H_m_o_mean': run_metrics['H_m_o_mean'],
        'H_m_o_std': run_metrics['H_m_o_std'],
        # 'H_o_m_mean': run_metrics['H_o_m_mean'],
        # 'H_o_m_std': run_metrics['H_o_m_std'],
        # 'H_o_o_mean': run_metrics['H_o_o_mean'],
        # 'H_o_o_std': run_metrics['H_o_o_std'],
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
                                        "rollin_mode": "mixed",
                                    },
                                }
                            })

def get_rollout_cost_function_configs(experiment_type, cost_func, mixing_coeff, 
                                        temperature=1, num_neighbors_to_add=-1):
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
                    "model_name": "gpt2",
                    "batch_size": 16,
                    "cuda_device": -2,
                }
            temperature = temperature
        rollout_cost_func_dict = {
          "type": "noisy_oracle",
          "add_brevity_penalty": true,
          "oracle": oracle,
        }
    overrides_dict = {
        "model": {
            "decoder": {
                "rollout_cost_function": rollout_cost_func_dict,
                "rollin_rollout_mixing_coeff": mixing_coeff,
                "temperature": temperature,
            }, 
        }
    }
    if num_neighbors_to_add > -1:
        overrides_dict["model"]["decoder"]["num_neighbors_to_add"] = num_neighbors_to_add
    return lambda: json.dumps(overrides_dict)

def get_scheduled_sampling_configs(num_samples, batch_size, num_epochs):
    k = lambda ss_type: calculate_ss_k(num_samples, batch_size, 
                                            num_epochs, ss_type=ss_type)
    return {
            'u_0': ('uniform', 0.0, -1),
            'u_0.05':  ('uniform', 0.05, -1),
            'u_0.10':  ('uniform', 0.1, -1),
            'u_0.25':  ('uniform', 0.25, -1),
            'linear': ('linear', 1.0, int(k('linear'))),
            'exponential': ('exponential', 1.0, int(k('exponential'))),
            'inverse_sigmoid': ('inverse_sigmoid', 1.0, int(k('inverse_sigmoid'))),
           }