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
from typing import Dict, Any, List, Iterable
import argparse
import logging
import json
import os
import torch
import numpy as np
import math
import random

import torch.nn.functional as F

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import prepare_environment
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.common import Params

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance


from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.training.util import evaluate
from allennlp.nn import util as nn_util

from quant_exp_bias.metrics.exposure_bias import ExposureBias
from quant_exp_bias.oracles.oracle_base import Oracle

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Subcommand.register("quantify-exposure-bias")
class QuantifyExposureBias(Subcommand):
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Evaluate the specified model + dataset'''
        subparser = parser.add_parser(
                self.name, description=description, help='Evaluate the specified model + dataset.')

        subparser.add_argument('archive_file', 
                               type=str, 
                               help='path to an archived trained model')

        subparser.add_argument('--oracle-config', 
                                type=str,
                                required=True,
                                help='config file for oracle.')

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
                                 default=360,
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
                                 oracle_config=args.oracle_config,
                                 num_trials=args.num_trials,
                                 num_length_samples=args.num_length_samples,
                                 num_samples_per_length=args.num_samples_per_length,
                                 cuda_device=args.cuda_device,
                                 overrides=args.overrides,
                                 weights_file=args.weights_file)

def quantify_exposure_bias(archive_file: str,
                           output_dir: str,
                           oracle_config,
                           num_trials: int = 5,
                           num_length_samples: int = 50,
                           num_samples_per_length: int = 360,
                           cuda_device: int = -1,
                           overrides: str = "",
                           weights_file: str = None, 
                           opt_level: str = None):
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True

    # Load from archive
    archive = load_archive(
                archive_file=archive_file, 
                cuda_device=cuda_device, 
                overrides=overrides, 
                weights_file=weights_file, 
                opt_level=opt_level)
    config = archive.config
    prepare_environment(config)
    config = dict(config)
    model = archive.model
    model.eval()

    generation_batch_size = config['model']['decoder'].get('generation_batch_size', num_samples_per_length)

    oracle_params_from_file = Params.from_file(oracle_config)
    oracle_params = oracle_params_from_file.get('oracle', {})
    assert oracle_params is not None, \
        "Oracle should be specified in configuration."
    
    oracle = Oracle.from_params(oracle_params)
    exposure_bias = ExposureBias(oracle)

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    validation_dataset_reader_params = config.pop("validation_dataset_reader", None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop("dataset_reader"))

    output_dir_trail = None
    exp_biases = []
    df_p_qs = []
    # df_q_ps = []
    H_m_o = []
    H_m_m = []
    # H_o_m = []
    # H_o_o = []

    input_dict = {
         "sample_rollouts": True,
        }
    input_dict['generation_batch_size'] = generation_batch_size

    logger.info(f'Num Trials: {num_trials}')
    logger.info(f'Num Length Samples: {num_length_samples}')
    logger.info(f'Num Samples Per Length: {input_dict["generation_batch_size"]}')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Metrics:")
    for trail_num in range(1, num_trials + 1):
        if output_dir:
            output_dir_trail = os.path.join(output_dir, str(trail_num))
            os.makedirs(output_dir_trail, exist_ok=True)
            # Delete all the previous context of the file. SO/2769061.
            open(os.path.join(output_dir_trail, 'model_sampled_generated.txt'), "w").close()
            # open(os.path.join(output_dir_trail, 'oracle_sampled_generated.txt'), "w").close()
            for sample_num in range(num_length_samples):

                output_dict = model(**input_dict)

                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict['predictions']

                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]

                model_sampled_model_probs = output_dict['model_sampled_model_probs']
                predicted_tokens = output_dict['predicted_tokens']
   
                # This +1 takes care of </S> prediction as predicted token are limited to
                # token before </S>.
                step_log_probs = F.log_softmax(output_dict['logits'].squeeze(1), dim=-1)
                model_sampled_model_seq_probs = torch.exp(torch.gather(step_log_probs, 2,
                                                                        best_predictions[:,1:].unsqueeze(2)) \
                                                                .squeeze(2))
                
                exp_bias_output_dict = exposure_bias(
                                            model_sampled_model_probs=model_sampled_model_probs,
                                            model_sampled_predictions=predicted_tokens,
                                            model_sampled_model_seq_probs=model_sampled_model_seq_probs.data.cpu(),
                                        )
                metric_trial = exposure_bias.get_metric(reset=True)

                exp_biases.append(metric_trial['exposure_bias'])
                df_p_qs.append(metric_trial['df_p_q'])

            if output_dir_trail:
                with open(os.path.join(output_dir_trail, 'model_sampled_generated.txt'), "a+") as file:
                    for seq, model_prob, oracle_prob, value in zip(output_dict['model_sampled_predicted_tokens'],
                                                                    exp_bias_output_dict['model_sampled_model_probs'],
                                                                    exp_bias_output_dict['model_sampled_oracle_probs'],
                                                                    exp_bias_output_dict['model_sampled_scores']):
                        print(f'{seq} P={model_prob:.4f} O={oracle_prob:.4f} Df_p_q={value:.4f}', file=file)
                        H_m_m.append(float(model_prob))
                        H_m_o.append(float(oracle_prob))
                    logger.info("Trial: %3d-%-3d :: %s: %-5.4f", trail_num, sample_num, "H_m_m", np.mean(H_m_m))
                    logger.info("Trial: %3d-%-3d :: %s: %-5.4f", trail_num, sample_num, "H_m_o", np.mean(H_m_o))


                # with open(os.path.join(output_dir_trail, 'oracle_sampled_generated.txt'), "a+") as file:
                #     for seq, model_prob, oracle_prob, value in zip(output_dict['oracle_sampled_predicted_tokens'],
                #                                                     output_dict['oracle_sampled_model_probs'],
                #                                                     output_dict['oracle_sampled_oracle_probs'],
                #                                                     output_dict['oracle_sampled_scores']):
                #         print(f'{seq} P={model_prob:.4f} O={oracle_prob:.4f} Df_q_p={value:.4f}', file=file)
                #         H_o_o.append(float(oracle_prob))
                #         H_o_m.append(float(model_prob))

    metrics = {
        'exposure_bias_mean': np.mean(exp_biases),
        'exposure_bias_std': np.std(exp_biases),
        'df_p_q_mean': np.mean(df_p_qs),
        'df_p_q_std': np.std(df_p_qs),
        # 'df_q_p_mean': np.mean(df_q_ps),
        # 'df_q_p_std': np.std(df_q_ps),
        'H_m_m_mean': np.mean(H_m_m),
        'H_m_m_std': np.std(H_m_m),
        'H_m_o_mean': np.mean(H_m_o),
        'H_m_o_std': np.std(H_m_o),
        # 'H_o_m_mean': np.mean(H_o_m),
        # 'H_o_m_std': np.std(H_o_m),
        # 'H_o_o_mean': np.mean(H_o_o),
        # 'H_o_o_std': np.std(H_o_o),
    }

    with open(os.path.join(output_dir, 'metrics.json'), "w") as file:
        json.dump(metrics, file, indent=4)

    logger.info("Exposure Bias Average:")
    logger.info("\t mean: %5.3f", metrics['exposure_bias_mean'])
    logger.info("\t std:  %5.3f", metrics['exposure_bias_std'])

    logger.info("H(M,O):")
    logger.info("\t mean: %5.3f", metrics['H_m_o_mean'])
    logger.info("\t std: %5.3f", metrics['H_m_o_std'])

    logger.info("H(M,M):")
    logger.info("\t mean: %5.3f", metrics['H_m_m_mean'])
    logger.info("\t std: %5.3f", metrics['H_m_m_std']) 

    logger.info("Done!!")
    # return exp_biases, metrics['exposure_bias_mean'], metrics['exposure_bias_std'], \
    #     df_p_qs, metrics['df_p_q_mean'], metrics['df_p_q_std'], \
    #     df_q_ps, metrics['df_q_p_mean'], metrics['df_q_p_std'], \
    #     metrics['H_m_m_mean'], metrics['H_m_m_std'], \
    #     metrics['H_m_o_mean'], metrics['H_m_o_std'], \
    #     metrics['H_o_m_mean'], metrics['H_o_m_std'], \
    #     metrics['H_o_o_mean'], metrics['H_o_o_std']

    return exp_biases, metrics['exposure_bias_mean'], metrics['exposure_bias_std'], \
        df_p_qs, metrics['df_p_q_mean'], metrics['df_p_q_std'], \
        metrics['H_m_m_mean'], metrics['H_m_m_std'], \
        metrics['H_m_o_mean'], metrics['H_m_o_std']