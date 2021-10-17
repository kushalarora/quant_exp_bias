from overrides import overrides
from typing import List

from allennlp.training.metrics.metric import Metric

from lmpl.oracles.oracle_base import Oracle

import logging
import torch
import numpy as np
import random
import math
from functools import reduce, partial

import torch.nn.functional as F

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Metric.register("exp_bias_v2")
class ExposureBiasV2(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """

    def __init__(self,
                 oracle: Oracle,
                 max_generated_len: int = 100
                ) -> None:
        self._total_value = 0.0
        self._df_p_q = 0.0
        self._count = 0
        self._oracle = oracle
        self.max_generated_len = max_generated_len

        self.idx2Q = [0.] * max_generated_len
        self.idx2lens = [0] * max_generated_len
        self.idx2count = [0] * max_generated_len

    @overrides
    def __call__(self,
                 model_sampled_predictions: List[str],
                 model_sampled_seq_lengths: List[int],
                 model_sampled_tokens: torch.LongTensor,
                 model_sampled_model_seq_probs: torch.FloatTensor,
                 model_sampled_step_log_probs: torch.FloatTensor,
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
        df_p_qs = []
        df_p_q_count_total = 0
        model_sampled_oracle_probs = []
        model_sampled_oracle_probs_and_seq_probs = self._oracle.compute_sent_probs(
                                                            model_sampled_predictions, 
                                                            model_sampled_tokens)

        for i in range(model_sampled_batch_size):
            if len(model_sampled_predictions[i]) == 0:
                continue

            values = []
            prev_p_qs = []
            seq_len = model_sampled_seq_lengths[i]

            assert model_sampled_seq_lengths[i] == model_sampled_oracle_probs_and_seq_probs[i][2]

            df_p_q_seq = 0
            df_p_q_count_seq = 0
            for j in range(1, seq_len):
                # Here model_sampled_model_prob is Q because the samples
                # come from the model.
                P_0_j = model_sampled_model_seq_probs[i][:j] \
                                        .log().sum(dim=-1).exp()
                
                model_sampled_oracle_step_prob = F.softmax(model_sampled_oracle_probs_and_seq_probs[i][-1][j], dim=-1)
                model_sampled_step_log_prob = model_sampled_step_log_probs[i, j, :-2]
                k_l = F.kl_div(model_sampled_step_log_prob, 
                                model_sampled_oracle_step_prob,
                                reduction='sum')

                value = k_l.item()
                
                values.append(value)
                df_p_q_seq += value
                df_p_q_count_seq += 1
                df_p_q_count_total += 1
                
                self.idx2Q[j] += df_p_q_seq
                self.idx2lens[j] += 1
                self.idx2count[j] += 1

            df_p_q += df_p_q_seq
            df_p_qs.append(df_p_q_seq/seq_len)

            model_sampled_oracle_probs.append(model_sampled_oracle_probs_and_seq_probs[i][0])
        
        self._total_value += df_p_q/df_p_q_count_total
        logging.info(f"KL(P || M) = {df_p_q/df_p_q_count_total:.4f}")

        return { "model_sampled_predictions": model_sampled_predictions, 
                  "model_sampled_oracle_probs": model_sampled_oracle_probs, 
                  "model_sampled_scores": df_p_qs,
                  "idx2Q": self.idx2Q,
                  "idx2lens": self.idx2lens,
                  "idx2count":  self.idx2count,
                }

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        avg_exp_bias = self._total_value
        avg_df_p_q = self._df_p_q

        if reset:
            self.reset()

        return {
            "exposure_bias": avg_exp_bias, 
            "df_p_q": avg_df_p_q
        }

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._df_p_q = 0.0