from overrides import overrides
from typing import List

from allennlp.training.metrics.metric import Metric

from quant_exp_bias.oracles.oracle_base import Oracle

import torch
import numpy 
from functools import reduce


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
        batch_size = len(predictions)
        oracle_probs = self._oracle.compute_sent_probs(predictions)
        model_probs = []

        for i  in range(len(predictions)):
            model_log_prob_i = predictions_losses[i]/len(predictions[i])
            model_prob_i = torch.exp(model_log_prob_i)
            model_probs.append(model_prob_i)

        total_value = 0
        for i in range(batch_size):
            try:
                value = torch.log(model_probs[i]/oracle_probs[i]).item()
            except Exception as e:
                print(e)
                print(oracle_probs[i])
                print(model_probs[i])
            if  numpy.isneginf(value) or numpy.isposinf(value):
                # with a warning. 
                print(f'{value}=log({model_probs[i]}/{oracle_probs[i]}) for {predictions[i]}.')
                continue
    
            total_value += value


        self._total_value += total_value
        self._count += batch_size

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
