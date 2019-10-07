"""
The ``sample-oracle`` subcommand allows you to create a vocabulary from
your dataset[s], which you can then reuse without recomputing it
each training run.

.. code-block:: bash

   $ allennlp sample-oracle --help
    usage: allennlp sample-oracle [-h] -s SERIALIZATION_DIR [-o OVERRIDES]
                               [--include-package INCLUDE_PACKAGE]
                               param_path

    Create a vocabulary from the specified dataset.

    positional arguments:
      param_path            path to parameter file describing the model and its
                            inputs

    optional arguments:
      -h, --help            show this help message and exit
      -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            directory in which to save the vocabulary directory
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
      --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
import argparse
import logging
import os

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment
from allennlp.data import Vocabulary
from quant_exp_bias.oracles.oracle_base import Oracle 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SampleOracle(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Create a vocabulary from the specified dataset.'''
        subparser = parser.add_parser(
                name, description=description, help='Sample oracle to generate dataset for quantifying exposure bias experiments.')
        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model and its inputs')

        subparser.add_argument('-s', '--serialization-dir',
                               required=True,
                               type=str,
                               help='directory in which to save the vocabulary directory')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('-n', '--num-samples',
                               type=int,
                               default=100000,
                               help='Number of samples to draw from oracle for training.')

        subparser.set_defaults(func=sample_oracle_from_args)

        return subparser


def sample_oracle_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to params.
    """
    parameter_path = args.param_path
    overrides = args.overrides
    serialization_dir = args.serialization_dir
    num_samples = args.num_samples

    params = Params.from_file(parameter_path, overrides)

    sample_oracle_from_params(params, serialization_dir, num_samples)


def sample_oracle_from_params(params: Params, 
                              serialization_dir: str, 
                              num_samples: int):
    prepare_environment(params)

    oracle_params = params.pop("oracle", {})
    os.makedirs(serialization_dir, exist_ok=True)
    oracle_file = os.path.join(serialization_dir, "oracle_samples.txt")

    if os.path.isfile(oracle_file):
       import time
       epoch_time = int(time.time())
       move_path = '.'.join([oracle_file, f'{epoch_time}'])
       logger.warn(f"Oracle Sample file already exists at {oracle_file}. Moving it to {move_path}")
       os.rename(oracle_file, move_path)

    oracle = Oracle.from_params(oracle_params)

    logger.info(f"writing the oracle samples to {oracle_file}.")
    with open(oracle_file, 'w') as oracle_output_file:
        for sample in oracle.sample_training_set(num_samples):
            print(sample, file=oracle_output_file)
    logger.info("done creating oracle samples")
