"""
The ``evaluate`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ allennlp evaluate --help
    usage: allennlp evaluate [-h] [--output-file OUTPUT_FILE]
                             [--weights-file WEIGHTS_FILE]
                             [--cuda-device CUDA_DEVICE] [-o OVERRIDES]
                             [--batch-weight-key BATCH_WEIGHT_KEY]
                             [--extend-vocab]
                             [--embedding-sources-mapping EMBEDDING_SOURCES_MAPPING]
                             [--include-package INCLUDE_PACKAGE]
                             archive_file input_file

    Evaluate the specified model + dataset

    positional arguments:
      archive_file          path to an archived trained model
      input_file            path to the file containing the evaluation data

    optional arguments:
      -h, --help            show this help message and exit
      --output-file OUTPUT_FILE
                            path to output file
      --weights-file WEIGHTS_FILE
                            a path that overrides which weights file to use
      --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
      --batch-weight-key BATCH_WEIGHT_KEY
                            If non-empty, name of metric used to weight the loss
                            on a per-batch basis.
      --extend-vocab        if specified, we will use the instances in your new
                            dataset to extend your vocabulary. If pretrained-file
                            was used to initialize embedding layers, you may also
                            need to pass --embedding-sources-mapping.
      --embedding-sources-mapping EMBEDDING_SOURCES_MAPPING
                            a JSON dict defining mapping from embedding module
                            path to embeddingpretrained-file used during training.
                            If not passed, and embedding needs to be extended, we
                            will try to use the original file paths used during
                            training. If they are not available we will use random
                            vectors for embedding extension.
      --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
from typing import Dict, Any
import argparse
import logging
import json
import os
import torch
import numpy as np

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import prepare_environment
from allennlp.common.checks import ConfigurationError, check_for_gpu


from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.training.util import evaluate
from allennlp.common import Params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class QuantifyExposureBias(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Evaluate the specified model + dataset'''
        subparser = parser.add_parser(
                name, description=description, help='Evaluate the specified model + dataset.')

        subparser.add_argument('archive_file', 
                               type=str, 
                               help='path to an archived trained model')

        subparser.add_argument('--output-dir', 
                               required=True,
                               type=str, 
                               help='path to output directory')

        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device',
                                 type=int,
                                 default=-1,
                                 help='id of GPU to use (if any)')

        subparser.add_argument('--num-samples-per-length',
                                 type=int,
                                 default=200,
                                 help='Number of samples to draw from $w_{1}^{n}~p$ for approximating expectation.')

        subparser.add_argument('--num-length-samples',
                                 type=int,
                                 default=10,
                                 help='Number of samples to draw from $n~\mathcal{N}$" + \
                                        "for approximating expectation over sequence lengths.')
        
        subparser.add_argument('--num-trials',
                                 type=int,
                                 default=10,
                                 help='Number of samples to draw from $n~\mathcal{N}$" + \
                                        "for approximating expectation over sequence lengths.')
        
        # TODO (Kushal): Add command line option to support getting context length distribution
        # from data.
        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.set_defaults(func=quantify_exposure_bias_from_args)

        return subparser

def quantify_exposure_bias_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return quantify_exposure_bias(archive_file=args.archive_file,
                                 output_dir=args.output_dir,
                                 num_trials=args.num_trials,
                                 num_length_samples=args.num_length_samples,
                                 num_samples_per_length=args.num_samples_per_length,
                                 cuda_device=args.cuda_device,
                                 overrides=args.overrides,
                                 weights_file=args.weights_file)

def quantify_exposure_bias(archive_file: str,
                           output_dir: str,
                           num_trials: int = 5,
                           num_length_samples: int = 50,
                           num_samples_per_length: int = 1024,
                           cuda_device: int = -1,
                           overrides: str = "",
                           weights_file: str = None):
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    
    logger.info(f'Num Trials: {num_trials}')
    logger.info(f'Num Length Samples: {num_length_samples}')
    logger.info(f'Num Samples Per Length: {num_samples_per_length}')

    # Load from archive
    archive = load_archive(archive_file, cuda_device, overrides, weights_file)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    output_dir_trail = None
    exp_biases = []
    input_dict = { "compute_exposure_bias": True }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


    logger.info("Metrics:")
    for trail_num in range(1, num_trials + 1):
        if output_dir:
            output_dir_trail = os.path.join(output_dir, str(trail_num))
            os.makedirs(output_dir_trail, exist_ok=True)

        for _ in range(num_length_samples):
            # sample sentence length
            input_dict['generation_batch_size'] = num_samples_per_length
            output_dict = model(**input_dict)
            
            metric_trial = model.get_metrics(reset=True, get_exposure_bias=True)

            for key, metric in metric_trial.items():
                logger.info("Trial: %3d :: %s: %4.2f", trail_num, key, metric)

            exp_biases.append(metric_trial['exposure_bias'])

        if output_dir_trail:                
            with open(os.path.join(output_dir_trail, 'generated.txt'), "w") as file:
                for seq in output_dict['predicted_sequences']:
                    print(seq, file=file)

    metrics = {
        'exposure_bias_mean': np.mean(exp_biases),
        'exposure_bias_std': np.std(exp_biases)
    }

    with open(os.path.join(output_dir, 'metrics.json'), "w") as file:
        json.dump(metrics, file, indent=4)
    
    logger.info("Exposure Bias Average:")
    logger.info("\t mean: %4.2f", metrics['exposure_bias_mean'])
    logger.info("\t std:  %4.2f", metrics['exposure_bias_std'])
    logger.info("Done!!")
    return exp_biases, metrics['exposure_bias_mean'], metrics['exposure_bias_std']
