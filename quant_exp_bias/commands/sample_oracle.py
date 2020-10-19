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
import random
from datetime import datetime
import time
import math

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment
from allennlp.data import Vocabulary
from lmpl.oracles.oracle_base import Oracle 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Subcommand.register("sample-oracle")
class SampleOracle(Subcommand):
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Create a vocabulary from the specified dataset.'''
        subparser = parser.add_parser(
                self.name, description=description, help='Sample oracle to generate dataset for quantifying exposure bias experiments.')
        subparser.add_argument('param_path',
                               type=str,
                               help='path to config file to instantiate the oracle.')

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
                               default=10000,
                               help='Number of samples to draw from oracle for training.')
        
        subparser.add_argument('-f', '--dataset-filename',
                                type=str,
                                default=None,
                                help='File from which the dataset should be sampled.')

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
    dataset_filename = args.dataset_filename

    params = Params.from_file(parameter_path, overrides)

    sample_oracle(params, serialization_dir, num_samples, dataset_filename)


def sample_oracle(params: Params, 
                  serialization_dir: str, 
                  num_samples: int, 
                  dataset_filename: str) -> str:

    prepare_environment(params)

    logger.info(f"Num Samples: {num_samples}")
    os.makedirs(serialization_dir, exist_ok=True)
    oracle_filename = os.path.join(serialization_dir, "oracle_samples.txt")

    if os.path.isfile(oracle_filename):
       import time
       epoch_time = int(time.time())
       move_path = '.'.join([oracle_filename, f'{epoch_time}'])
       logger.warn(f"Oracle Sample file already exists at {oracle_filename}. Moving it to {move_path}")
       os.rename(oracle_filename, move_path)

    if dataset_filename is not None:
        # TODO (Kushal): Convert this to a generator.
        # TODO (Kushal): Maybe consider moving this out.
        filesize = 0
        with open(dataset_filename) as dataset_file:
            for line in dataset_file:
                if len(line.strip()) > 0:
                    filesize += 1


            # Get 20% valid set or 1000 examples, whichever is less.
            num_samples += max(1000, int(0.2 * num_samples))
            
            # sample_idxs = sorted(random.sample(range(filesize), num_samples))
            oracle_sample_iterator = []
            sample_count = 0
            dataset_file.seek(0)
            for i, line in enumerate(dataset_file):
                if random.random() < float(num_samples)/filesize:
                    oracle_sample_iterator.append(line.strip())
                    sample_count += 1

                if sample_count == num_samples:
                    break
    else:
        oracle_params = params.get('oracle', {})
        assert oracle_params is not None, \
            "Oracle should be specified in configuration."

        oracle = Oracle.from_params(oracle_params)
        oracle_sample_iterator = oracle.sample_training_set(num_samples)

    logger.info(f"writing the oracle samples to {oracle_filename}.")
    with open(oracle_filename, 'w') as oracle_file:
        for sample in oracle_sample_iterator:
            print(sample, file=oracle_file)

    logger.info("done creating oracle samples")
    return oracle_filename
