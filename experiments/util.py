import json
import os
import sys
import wandb

from comet_ml import Experiment
from datetime import datetime
from typing import Dict, List, Callable, Tuple, Union, Any

from allennlp.common import Params
from allennlp.common.util import import_submodules
import_submodules("quant_exp_bias")
from quant_exp_bias.utils import (get_args, quantify_exposure_bias_runner,
                  sample_oracle_runner, train_runner)

from quant_exp_bias.oracles.artificial_grammar_oracle import ArtificialLanguageOracle
from quant_exp_bias.utils import get_args


OverrideFuncType = Callable[[], Dict[str, Union[float, str, int]]]
ExpBiasEpochsFuncType = Callable[[str], List[Tuple[int, str, str]]]

def generate_grammar_file(serialization_dir:str,
                            grammar_template: str='grammar_templates/grammar_2.template',
                            vocabulary_size: int=6,
                            vocabulary_distribution: str='uniform'):
    grammar_string = ArtificialLanguageOracle.generate_grammar_string(grammar_template_file=grammar_template,
                                                                        vocabulary_size=vocabulary_size,
                                                                        vocabulary_distribution=vocabulary_distribution)
    grammar_filename = os.path.join(serialization_dir, 'grammar.txt')
    with open(grammar_filename, 'w') as f:
        f.write(grammar_string)
    os.environ["FSA_GRAMMAR_FILENAME"]  = grammar_filename
    return grammar_filename

def initialize_experiments(experiment_name: str, 
                            param_path: str = None):
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
    serialization_dir = os.path.join(main_args.output_dir, experiment_name, experiment_id)
    param_path = param_path or main_args.config

    os.makedirs(serialization_dir, exist_ok=True)

    os.environ['TRAIN_FILE'] = ""
    os.environ['DEV_FILE'] = ""

    experiment = Experiment(workspace='quantifying_exposure_bias', 
                            project_name=experiment_name,
                            auto_metric_logging=False,
                            auto_param_logging=False,
                           )

    return main_args, serialization_dir, param_path, experiment_id, experiment

def default_overides_func():
    return '{}'

def default_exp_bias_epochs_func(train_model_serialization_dir):
    epoch = -1; qeb_suffix = ''; metrics_filename='metrics.json'
    return [(epoch, qeb_suffix, metrics_filename)]

def one_exp_run(serialization_dir:str, 
                num_samples:int, 
                run:int, 
                param_path:str,
                overides_func:OverrideFuncType = default_overides_func,
                exp_bias_epochs_func:ExpBiasEpochsFuncType = default_exp_bias_epochs_func,
                sample_from_file=False,
                dataset_filename=None,
                exp_bias_inference_funcs:List[Tuple[str, Any, OverrideFuncType]] = lambda:[],
               ):
    run_serialization_dir = os.path.join(serialization_dir, str(num_samples), str(run))
    overrides = overides_func()
    sample_oracle_args=['sample-oracle', 
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

    train_args = get_args(args=['train', 
                                    param_path, 
                                    '-s', run_serialization_dir, 
                                    '-o',  overrides])
    trainer_params = Params.from_file(train_args.param_path, train_args.overrides)
    cuda_device = trainer_params['trainer']['cuda_device']

    train_model_serialization_dir = train_runner(train_args, 
                                                run_serialization_dir);

    archive_file = os.path.join(train_model_serialization_dir, 'model.tar.gz')
    metric_list = []

    for epoch, qeb_suffix, metric_filename in exp_bias_epochs_func(train_model_serialization_dir):
        qeb_output_dir = os.path.join(run_serialization_dir, 'exp_bias', qeb_suffix)
        metrics = json.load(open(os.path.join(train_model_serialization_dir, metric_filename)))

        # This is only needed when doing validation experiments.
        weights_file = None
        if epoch != -1:
            weights_file = os.path.join(train_model_serialization_dir, f'model_state_epoch_{epoch}.th')

        qeb_args = get_args(args=['quantify-exposure-bias',
                                    archive_file,
                                    oracle_test_filename,
                                    '--output-dir', qeb_output_dir,
                                    '--weights-file', weights_file,
                                    '-o',  overrides])

        exp_biases, exp_bias_mean, exp_bias_std, \
            df_p_qs, df_p_q_mean, df_p_q_std, \
                df_q_ps, df_q_p_mean, df_q_p_std = quantify_exposure_bias_runner(qeb_args,
                                                                                    archive_file,
                                                                                    oracle_test_filename,
                                                                                    qeb_output_dir,
                                                                                    cuda_device=cuda_device,
                                                                                    weights_file=weights_file,
                                                                                )

        metrics['exp_biases'] = exp_biases
        metrics['exp_bias_mean'] = exp_bias_mean
        metrics['exp_bias_std'] = exp_bias_std

        metrics['df_p_qs'] = df_p_qs
        metrics['df_p_q_mean'] = df_p_q_mean
        metrics['df_p_q_std'] = df_p_q_std

        metrics['df_q_ps'] = df_q_ps
        metrics['df_q_p_mean'] = df_q_p_mean
        metrics['df_q_p_std'] = df_q_p_std
        
        metric_list.append(metrics)

    for key, value, qeb_overides in exp_bias_inference_funcs():
        metrics = json.load(open(os.path.join(train_model_serialization_dir, metric_filename)))

        qeb_suffix = f"{key}_{value}"
        qeb_output_dir = os.path.join(run_serialization_dir, 'exp_bias', qeb_suffix)

        qeb_args = get_args(args=['quantify-exposure-bias',
                                    archive_file,
                                    '--output-dir', qeb_output_dir,
                                    '-o', qeb_overides])

        exp_biases, exp_bias_mean, exp_bias_std, \
            df_p_qs, df_p_q_mean, df_p_q_std, \
                df_q_ps, df_q_p_mean, df_q_p_std = quantify_exposure_bias_runner(qeb_args,
                                                                                    archive_file,
                                                                                    oracle_test_filename,
                                                                                    qeb_output_dir,
                                                                                    cuda_device=cuda_device,
                                                                                    num_length_samples=40);

        metrics['exp_biases'] = exp_biases
        metrics['exp_bias_mean'] = exp_bias_mean
        metrics['exp_bias_std'] = exp_bias_std

        metrics['df_p_qs'] = df_p_qs
        metrics['df_p_q_mean'] = df_p_q_mean
        metrics['df_p_q_std'] = df_p_q_std

        metrics['df_q_ps'] = df_q_ps
        metrics['df_q_p_mean'] = df_q_p_mean
        metrics['df_q_p_std'] = df_q_p_std
        
        metrics[key] = value
        metric_list.append(metrics)

    return metric_list
