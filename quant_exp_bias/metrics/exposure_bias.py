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


def rfn_prefix(p, q, prev_p_q, n): 
    return np.exp(1.0/n * ((n-1)*np.log(prev_p_q) + np.log(p) - np.log(q)))

def rfn_sequence(p, q, prev_p_q, n): 
    return np.exp(n * (np.log(p) - np.log(q)))


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
                 type: str = 'kl',
                 at_prefix_level: bool = False,
                 ) -> None:
        self._total_value = 0.0
        self._df_p_q = 0.0
        self._df_q_p = 0.0
        self._count = 0
        self._oracle = oracle
        self._type = type
        self._at_prefix_level = at_prefix_level

        # D_f(P||Q) = \sum_{x in X} f(p(X)/q(x))q(x)
        self._Df = ExposureBias.DfBuilder(type,
                                          rfn_prefix if at_prefix_level else rfn_sequence)

    @overrides
    def __call__(self,
                 model_sampled_model_probs: torch.FloatTensor,
                 model_sampled_predictions: List[str],
                 model_sampled_model_seq_probs: torch.FloatTensor,
                 use_js: bool = False,
                 oracle_sampled_model_probs: torch.FloatTensor = None,
                 oracle_sampled_predictions: List[str] = [],
                 oracle_sampled_model_seq_probs: torch.FloatTensor = None,
                 ):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        # TODO (Kushal) Add comments to explain what is going on.
        # Compute DL(P||M)
        model_sampled_batch_size = len(model_sampled_predictions)
        df_p_q = 0
        df_p_q_count = 0
        df_p_qs = []

        model_sampled_oracle_probs = []
        model_sampled_oracle_probs_and_seq_probs = self._oracle.compute_sent_probs(model_sampled_predictions)
        for i in range(model_sampled_batch_size):

            if len(model_sampled_predictions[i]) == 0:
                continue

            seq_len = len(model_sampled_predictions[i])
            if self._at_prefix_level:
                df_p_q_seq = 0
                prev_p_q = 1.0
                for j in range(seq_len):
                    # Here model_sampled_model_prob is Q because the samples
                    # come from the model.
                    P = model_sampled_oracle_probs_and_seq_probs[i][1][j]
                    Q = model_sampled_model_seq_probs[i][j].item()

                    value, prev_p_q = self._Df(P, Q, prev_p_q, j+1)
                    df_p_q_seq += 0.5 * value

                df_p_q += df_p_q_seq
                df_p_qs.append(df_p_q_seq/seq_len)
                df_p_q_count += seq_len
            else:
                P = model_sampled_oracle_probs_and_seq_probs[i][0]
                Q = model_sampled_model_probs[i].item()

                value, _ = self._Df(P, Q, 1.0, 1.0)
                df_p_q += 0.5 * value

                df_p_qs.append(value)
                df_p_q_count += 1

            model_sampled_oracle_probs.append(model_sampled_oracle_probs_and_seq_probs[i][0])
        oracle_sampled_batch_size = len(oracle_sampled_predictions)
        df_q_ps = []
        df_q_p = 0
        df_q_p_count = 0
        
        # Compute DL(Q||M)
        oracle_sampled_oracle_probs = []
        oracle_sampled_oracle_probs_and_seq_probs = self._oracle.compute_sent_probs(oracle_sampled_predictions)
        for i in range(oracle_sampled_batch_size):

            if len(oracle_sampled_predictions[i]) == 0:
                continue

            seq_len = len(oracle_sampled_predictions[i])
            if self._at_prefix_level:
                df_q_p_seq = 0
                prev_p_q = 1.0
                for j in range(seq_len):
                    # Here oracle_sampled_oracle_probs is Q because the samples
                    # come from the oracle.
                    P = oracle_sampled_oracle_probs_and_seq_probs[i][1][j]
                    Q = oracle_sampled_model_seq_probs[i][j].item()

                    value, prev_p_q = self._Df(Q, P, prev_p_q, j+1)
                    df_q_p_seq += 0.5 * value

                df_q_p += df_q_p_seq
                df_q_ps.append(df_q_p_seq/seq_len)
                df_q_p_count += seq_len
            else:
                P = oracle_sampled_oracle_probs_and_seq_probs[i][0]
                Q = oracle_sampled_model_probs[i].item()
                
                value, _ = self._Df(Q, P, 1.0, 1.0)
                df_q_p += 0.5 * value

                df_q_ps.append(value)
                df_q_p_count += 1

            oracle_sampled_oracle_probs.append(oracle_sampled_oracle_probs_and_seq_probs[i][0])

        self._total_value += df_p_q/df_p_q_count + df_q_p/df_q_p_count
        self._df_p_q += df_p_q/df_p_q_count
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
    def DfBuilder(type='kl', rfn=rfn_sequence):
        if type == 'abs_kl':
            return lambda p, q, prev_p_q, n: (np.abs(np.log(rfn(q, p, prev_p_q, n))), rfn(q, p, prev_p_q, n))
        if type == 'kl':
            return lambda p, q, prev_p_q, n: (np.log(rfn(q, p, prev_p_q, n)), rfn(q, p, prev_p_q, n))
        elif type == 'hellinger_squared':
            return lambda p, q, prev_p_q, n: ((np.sqrt(rfn(p, q, prev_p_q, n)) - 1)**2, rfn(p, q, prev_p_q, n))
        elif type == 'tv':
            return lambda p, q, prev_p_q, n: (np.abs(rfn(p, q, prev_p_q, n) - 1), rfn(p, q, prev_p_q, n))
        elif type == 'js':
            return lambda p, q, prev_p_q, n: (np.log(2/(rfn(p, q, prev_p_q, n) + 1)), rfn(p, q, prev_p_q, n))
