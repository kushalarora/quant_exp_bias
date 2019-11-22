from overrides import overrides
from typing import List

from allennlp.training.metrics.metric import Metric

from quant_exp_bias.oracles.oracle_base import Oracle

import logging
import torch
import numpy 
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
                 predictions_losses: torch.FloatTensor,
                 predictions: List[str]):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        filtered_predictions = []
        filtered_predictions_losses = []

        # If it is an empty sequence or 1 word sequence, 
        # ignore it.
        for i, prediction in enumerate(predictions):
            if len(prediction) > 1:
                filtered_predictions.append(prediction)
                filtered_predictions_losses.append(predictions_losses[i])

        predictions = filtered_predictions
        predictions_losses = predictions_losses.new_tensor(filtered_predictions_losses)
        batch_size = len(predictions)

        oracle_probs = self._oracle.compute_sent_probs(predictions)
        model_probs = []

        for i  in range(len(predictions)):
            model_log_prob_i = predictions_losses[i]/len(predictions[i])
            model_prob_i = torch.exp(model_log_prob_i)
            model_probs.append(model_prob_i)

        total_value = 0
        for i in range(batch_size):
            value = 0
            if oracle_probs[i] > 0:
                value = torch.log(model_probs[i]/oracle_probs[i]).item()
                if  numpy.isneginf(value) or numpy.isposinf(value):
                    # with a warning. 
                    logging.warn(f'{value}=log({model_probs[i]}/{oracle_probs[i]}) for {predictions[i]}.')
                    continue
            else:
                logging.warn(f"Failed to parse prediction: {predictions[i]}. Model Prob: {model_probs[i]}")
                continue
                
            self._count += 1
            self._total_value += value
    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0
