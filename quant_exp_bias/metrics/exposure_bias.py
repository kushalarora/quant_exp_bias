from overrides import overrides
from typing import List

from allennlp.training.metrics.metric import Metric

from quant_exp_bias.oracles.oracle_base import Oracle

import logging
import torch
import numpy as np
import random
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
                 oracle: Oracle,
                 type:str = 'hellinger_squared',
                ) -> None:
        self._total_value = 0.0
        self._df_p_q = 0.0
        self._df_q_p = 0.0
        self._count = 0
        self._oracle = oracle
        self._type = type

        # D_f(P||Q) = \sum_{x in X} f(p(X)/q(x))q(x)
        self._Df = ExposureBias.DfBuilder(type)

    @overrides
    def __call__(self,
                 model_sampled_model_log_probs: torch.FloatTensor,
                 model_sampled_predictions: List[str],
                 use_js: bool = False, 
                 oracle_sampled_model_log_probs: torch.FloatTensor = None,
                 oracle_sampled_predictions: List[str] = [],
                 oracle_samples_seq_log_probs: torch.FloatTensor = None,
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
        df_p_q = 0
        df_p_q_count = 0
        df_p_qs = []

        model_sampled_oracle_probs = self._oracle.compute_sent_probs(model_sampled_predictions)
        for i in range(model_sampled_batch_size):
            value = 0
            if len(model_sampled_predictions[i]) == 0:
                continue

            model_sampled_model_prob_i =  math.exp(model_sampled_model_log_probs[i].item())
            model_sampled_model_probs.append(model_sampled_model_prob_i)

            # Here model_sampled_model_prob is Q because the samples
            # come from the model.
            Q = model_sampled_model_prob_i
            P = model_sampled_oracle_probs[i]
            value = self._Df(P, Q, len(model_sampled_predictions[i]))
            df_p_q += value
            df_p_qs.append(value)
            df_p_q_count += 1

            if random.random() < 0.2:
                logging.debug(f"Df_P_Q => P={P:.4f}, Q={Q:.4f}, Value={value:.4f}")

            if  np.isneginf(value) or np.isposinf(value):
                # with a warning.
                logging.warn(f'Df_P_Q => P={P:.4f}, Q={Q:.4f}, Value={value:.4f} for {model_sampled_predictions[i]}.')
                continue

        oracle_sampled_batch_size = len(oracle_sampled_predictions)
        oracle_sampled_model_probs = []
        df_q_ps = []
        df_q_p = 0
        df_q_p_count = 0

        # Compute DL(Q||M)
        oracle_sampled_oracle_probs = self._oracle.compute_sent_probs(oracle_sampled_predictions)
        for i in range(oracle_sampled_batch_size):
            value = 0
            if len(oracle_sampled_predictions[i]) == 0:
                continue
            
            oracle_sampled_model_prob_i =  math.exp(oracle_sampled_model_log_probs[i].item())

            oracle_sampled_model_probs.append(oracle_sampled_model_prob_i)
           
            # Here oracle_sampled_oracle_probs is Q because the samples
            # come from the oracle.
            Q = oracle_sampled_oracle_probs[i]
            P = oracle_sampled_model_prob_i
            value = self._Df(P, Q, len(oracle_sampled_predictions[i]))
            df_q_p += value
            df_q_ps.append(value)
            df_q_p_count += 1

            if random.random() < 0.2:
                logging.debug(f"Df_Q_P => Q={Q:.4f}, P={P:.4f}, Value={value:.4f}")

            if  np.isneginf(value) or np.isposinf(value):
                # with a warning.
                logging.warn(f'Df_Q_P ==> Q={Q:.4f}, P={P:.4f}, Value={value:.4f} for {model_sampled_predictions[i]:.4f}.')
                continue

        self._total_value += df_p_q/df_p_q_count + df_q_p/df_q_p_count
        self._df_p_q +=  df_p_q/df_p_q_count
        self._df_q_p += df_q_p/df_q_p_count

        logging.info(f"KL(P || M) = {df_p_q/df_p_q_count:.4f} \t KL(Q || M) = {df_q_p/df_q_p_count:.4f}")
        return model_sampled_predictions, model_sampled_model_probs, model_sampled_oracle_probs, df_p_qs, \
                oracle_sampled_predictions, oracle_sampled_model_probs, oracle_sampled_oracle_probs, df_q_ps

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        avg_exp_bias = self._total_value 
        avg_df_p_q = self._df_p_q
        avg_df_q_p = self._df_q_p

        if reset:
            self.reset()

        return avg_exp_bias, avg_df_p_q, avg_df_q_p

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._df_p_q = 0.0
        self._df_q_p = 0.0

    @staticmethod
    def DfBuilder(type='kl'):
        rfn = lambda p,q,n: np.exp(n * np.log(p/q))
        if type == 'kl':
            return lambda p,q,n: 0.5 * np.log(rfn(q,p,n))
        elif type == 'hellinger_squared':
            return lambda p,q,n: 0.5 * (np.sqrt(rfn(p,q,n)) - 1)**2
        elif type == 'tv':
            return lambda p,q,n: 0.5 * np.abs(rfn(p,q,n) - 1)
        elif type == 'js':
            return lambda p,q,n: 0.5 * np.log(2/(rfn(p,q,n) + 1))