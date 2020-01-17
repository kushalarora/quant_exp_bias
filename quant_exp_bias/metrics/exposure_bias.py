from overrides import overrides
from typing import List

from allennlp.training.metrics.metric import Metric

from quant_exp_bias.oracles.oracle_base import Oracle

import logging
import torch
import numpy
import math
from functools import reduce

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Metric.register("exp_bias")
class ExposureBias(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """
    def __init__(self, 
                 oracle: Oracle) -> None:
        self._total_value = 0.0
        self._count = 0
        self._oracle = oracle

    @overrides
    def __call__(self,
                 model_sampled_losses: torch.FloatTensor,
                 model_sampled_predictions: List[str],
                 use_js: bool = False, 
                 oracle_sampled_losses: torch.FloatTensor = None,
                 oracle_sampled_predictions: List[str] = [],
                 ):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        # filtered_predictions = []
        # filtered_predictions_losses = []

        # If it is an empty sequence or 1 word sequence,
        # ignore it.
        # for i, prediction in enumerate(model_sampled_predictions):
        #     if len(prediction) > 1:
        #         filtered_predictions.append(prediction)
        #         filtered_predictions_losses.append(model_sampled_losses[i])

        # model_sampled_predictions = filtered_predictions
        # model_sampled_losses = model_sampled_losses.new_tensor(filtered_predictions_losses)
        
        # TODO (Kushal) Add comments to explain what is going on.
        # Compute DL(P||M)
        model_sampled_batch_size = len(model_sampled_predictions)
        model_sampled_model_probs = []
        kl_p_m = 0
        kl_p_m_count = 0

        model_sampled_oracle_probs = self._oracle.compute_sent_probs(model_sampled_predictions)
        for i in range(model_sampled_batch_size):
            value = 0

            model_sampled_model_prob_i =  math.exp(model_sampled_losses[i].item()/len(model_sampled_predictions[i]))
            model_sampled_model_probs.append(model_sampled_model_prob_i)
            
            if model_sampled_oracle_probs[i] <= 0:
                logging.warn(f"Failed to parse prediction: {model_sampled_predictions[i]}. Model Prob: {model_sampled_model_probs[i]}")

            M = (model_sampled_model_prob_i + model_sampled_oracle_probs[i] + 1e-45)/2
            P = model_sampled_model_prob_i
            value =  math.log(P/M)
            kl_p_m += value
            if  numpy.isneginf(value) or numpy.isposinf(value):
                # with a warning.
                logging.warn(f'{value}=log({P}/{M}) for {model_sampled_predictions[i]}.')
                continue
            kl_p_m_count += 1

        oracle_sampled_batch_size = len(oracle_sampled_predictions)
        oracle_sampled_model_probs = []
        kl_q_m = 0
        kl_q_m_count = 0

        # Compute DL(Q||M)
        oracle_sampled_oracle_probs = self._oracle.compute_sent_probs(oracle_sampled_predictions)
        for i in range(oracle_sampled_batch_size):
            value = 0
            oracle_sampled_model_prob_i =  math.exp(oracle_sampled_losses[i].item()/len(oracle_sampled_predictions[i]))
              
            if oracle_sampled_oracle_probs[i] <= 0:
                logging.warn(f"Failed to parse prediction: {oracle_sampled_predictions[i]}. Model Prob: {oracle_sampled_model_probs[i]}")

            oracle_sampled_model_probs.append(oracle_sampled_model_prob_i)

            M = (oracle_sampled_model_prob_i + oracle_sampled_oracle_probs[i] + 1e-45)/2
            Q = oracle_sampled_model_prob_i
            value = math.log(Q/M)
            kl_q_m += value
            if  numpy.isneginf(value) or numpy.isposinf(value):
                # with a warning.
                logging.warn(f'{value}=log({Q}/{M}) for {model_sampled_predictions[i]}.')
                continue
            kl_q_m_count += 1

        self._total_value += 0.5 * kl_p_m/max(kl_p_m_count, 1) + 0.5 * kl_q_m/max(kl_q_m_count, 1)

        return model_sampled_predictions, model_sampled_model_probs, model_sampled_oracle_probs

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = self._total_value 
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0