
from typing import Dict
import argparse
import logging
import os
import random
import time

from overrides import overrides

from allennlp import __version__
from allennlp.commands.evaluate import Evaluate
from allennlp.commands.predict import Predict
from allennlp.commands.subcommand import Subcommand
from allennlp.commands.test_install import TestInstall
from allennlp.commands.find_learning_rate import FindLearningRate
from allennlp.commands.train import Train
from allennlp.commands.print_results import PrintResults

from allennlp.common.params import Params
from allennlp.commands.train import train_model
from allennlp.commands import ArgumentParserWithDefaults

from quant_exp_bias.commands.quantify_exposure_bias import QuantifyExposureBias, \
                                                            quantify_exposure_bias
from quant_exp_bias.commands.sample_oracle import SampleOracle, sample_oracle



# pylint: disable=protected-access

def get_args(args: argparse.Namespace = None): 
    """
    The :mod:`~allennlp.run` command only knows about the registered classes in the ``allennlp``
    codebase. In particular, once you start creating your own ``Model`` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag.
    """
    # pylint: disable=dangerous-default-value
    parser = ArgumentParserWithDefaults(description="Run AllenNLP", usage='%(prog)s', prog='qeb')
    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)


    parser.add_argument('--config', type=str, default='training_configs/artificial_grammar/artificial_grammar_composed.jsonnet', help='config files to run experiments.')

    parser.add_argument('--output_dir', type=str, default='results/artificial_grammar/', help='Output directory')

    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
            # Default commands
            "train": Train(),
            "evaluate": Evaluate(),
            "predict": Predict(),
            "test-install": TestInstall(),
            "find-lr": FindLearningRate(),
            "print-results": PrintResults(),
            "sample-oracle": SampleOracle(),
            "quantify-exposure-bias": QuantifyExposureBias()
            
    }
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(subparsers)
        subparser.add_argument(
            "--include-package",
            type=str,
            action="append",
            default=[],
            help="additional packages to include",
        )
    args, _ = parser.parse_known_args(args)
    return args

def sample_oracle_runner(args: argparse.Namespace, 
                         serialization_dir: str,
                         num_samples: int = None): 
    parameter_path = args.param_path
    overrides = args.overrides

    serialization_dir = os.path.join(serialization_dir, 'data')
    params = Params.from_file(parameter_path, overrides)
    num_samples = num_samples or args.num_samples
    dataset_filename = args.dataset_filename
    oracle_filename =  sample_oracle(params=params, 
                                     serialization_dir=serialization_dir,
                                    # X 1.1 for validation set.
                                     num_samples=int(num_samples * 1.1),
                                     dataset_filename=dataset_filename)

    oracle_train_filename = os.path.join(serialization_dir, 'oracle_samples-train.txt')
    oracle_dev_filename = os.path.join(serialization_dir, 'oracle_samples-dev.txt')
    oracle_test_filename = os.path.join(serialization_dir, 'oracle_samples-test.txt')

    with open(oracle_filename) as full_file, \
         open(oracle_train_filename, 'w') as train_file, \
         open(oracle_dev_filename, 'w') as dev_file, \
         open(oracle_test_filename, 'w') as test_file:
        num_lines = 0
        for line in full_file:
            num_lines += 1

        full_file.seek(0)

        for i,line in enumerate(full_file):
            line = line.strip()
            if i < num_samples:
                print(line, file=train_file)
            else:
                print(line, file=dev_file)

    return oracle_train_filename, oracle_dev_filename, oracle_test_filename

def train_runner(args: argparse.Namespace,
                 serialization_dir: str, 
                 recover:bool=False,
                 force:bool=False,
                ):

    serialization_dir = os.path.join(serialization_dir, 'training')
    params = Params.from_file(args.param_path, args.overrides)
    model = train_model(params,
                        serialization_dir=serialization_dir,
                        file_friendly_logging=True,
                        force=force,
                        recover=recover)

    return serialization_dir

def quantify_exposure_bias_runner(args: argparse.Namespace,
                                  archive_file: str,
                                  oracle_config: str,
                                  output_dir: str,
                                  cuda_device: int,
                                  weights_file: str = None,
                                  num_trials: int = None,
                                  num_length_samples: int = None,
                                  num_samples_per_length: int = None,
                                  ):
    num_trials = num_trials or args.num_trials
    num_samples_per_length = num_samples_per_length or args.num_samples_per_length
    num_length_samples = num_length_samples or args.num_length_samples
    return quantify_exposure_bias(archive_file=archive_file, 
                                  output_dir=output_dir, 
                                  oracle_config=oracle_config,
                                  num_trials=num_trials,
                                  num_length_samples=num_length_samples,
                                  num_samples_per_length=num_samples_per_length,
                                  cuda_device=cuda_device,
                                  overrides=args.overrides,
                                  weights_file=weights_file,
                                 )