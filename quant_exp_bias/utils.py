from allennlp.common.util import import_submodules
import_submodules("quant_exp_bias")
import argparse


from typing import Dict
import argparse
import logging
import os
import random
import time

from overrides import overrides

from allennlp import __version__
from allennlp.commands.configure import Configure
from allennlp.commands.elmo import Elmo
from allennlp.commands.evaluate import Evaluate
from allennlp.commands.fine_tune import FineTune
from allennlp.commands.make_vocab import MakeVocab
from allennlp.commands.predict import Predict
from allennlp.commands.dry_run import DryRun
from allennlp.commands.subcommand import Subcommand
from allennlp.commands.test_install import TestInstall
from allennlp.commands.find_learning_rate import FindLearningRate
from allennlp.commands.train import Train
from allennlp.commands.print_results import PrintResults

from allennlp.common.params import Params
from allennlp.commands.train import train_model
from allennlp.commands import ArgumentParserWithDefaults

from quant_exp_bias.commands.quantify_exposure_bias import QuantifyExposureBias, quantify_exposure_bias
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
            "configure": Configure(),
            "train": Train(),
            "evaluate": Evaluate(),
            "predict": Predict(),
            "make-vocab": MakeVocab(),
            "elmo": Elmo(),
            "fine-tune": FineTune(),
            "dry-run": DryRun(),
            "test-install": TestInstall(),
            "find-lr": FindLearningRate(),
            "print-results": PrintResults(),
            "sample-oracle": SampleOracle(),
            "quantify-exposure-bias": QuantifyExposureBias()
           
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
 
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
    oracle_filename =  sample_oracle(params, 
                                     serialization_dir,
                                     num_samples)
    oracle_train_filename = os.path.join(serialization_dir, 'oracle_samples-train.txt')
    oracle_dev_filename = os.path.join(serialization_dir, 'oracle_samples-dev.txt')

    with open(oracle_filename) as full_file, \
         open(oracle_train_filename, 'w') as train_file, \
         open(oracle_dev_filename, 'w') as dev_file:
        
        for line in full_file:
            line = line.strip()
            if random.random() < 0.1:
                print(line, file=dev_file)
            else:
                print(line, file=train_file)

    return oracle_train_filename, oracle_dev_filename

def train_runner(args: argparse.Namespace,
                 serialization_dir: str):

    serialization_dir = os.path.join(serialization_dir, 'training')
    params = Params.from_file(args.param_path, args.overrides)
    model = train_model(params,
                        serialization_dir=serialization_dir,
                        file_friendly_logging=True,
                        force=False)

    # HACK (Kushal): This is a hack to fix pool workers not dying
    # at the end of the train call. Somehow the __del__ method of the
    # model class is not be called. This is an issue, as for artificial 
    # language use case, the pool workers hog memory and if not cleaned, 
    # the redundant pool workers retain the memory leading to memory leak
    # resulting in oom errors.
    if model._decoder is not None and \
            model._decoder._oracle is not None and \
            model._decoder._oracle._pool is not None:
                model._decoder._oracle._pool.terminate()
                model._decoder._oracle._pool.join()
                time.sleep(2)

    return serialization_dir

def quantify_exposure_bias_runner(args: argparse.Namespace,
                                  archive_file: str,
                                  output_dir: str,
                                  cuda_device: int,
                                  weights_file: str = None,
                                  num_trials: int = None,
                                  num_length_samples: int = None,
                                  num_samples_per_length: int = None):

    num_trials = num_trials or args.num_trials
    num_samples_per_length = num_samples_per_length or args.num_samples_per_length
    num_length_samples = num_length_samples or args.num_length_samples
    
    return quantify_exposure_bias(archive_file=archive_file, 
                                  output_dir=output_dir, 
                                  num_trials=num_trials,
                                  num_length_samples=num_length_samples,
                                  num_samples_per_length=num_samples_per_length,
                                  cuda_device=cuda_device,
                                  overrides=args.overrides,
                                  weights_file=weights_file)
