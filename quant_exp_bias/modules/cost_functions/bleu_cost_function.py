from typing import Dict, Optional, Tuple, Union, List

from fairseq.bleu import Scorer, SacrebleuScorer
from overrides import overrides

from allennlp.common.util import END_SYMBOL
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
import torch

from quant_exp_bias.modules.cost_functions import CostFunction

@CostFunction.register("bleu")
class BLEUCostFunction(CostFunction):
    """ This call computes BLEU loss function between prediction and 
        gold targets. This is used to train NMT model.
    """

    def __init__(self,
                 pad_token = 0,
                 eos_token = 4,
                 unk_token = 1,
                 use_decoded_inputs = True,
                 ):

        if use_decoded_inputs:
            self._scorer = SacrebleuScorer()
        else:
            self._scorer = Scorer(pad_token,
                                eos_token,
                                unk_token)

        self._use_decoded_inputs = use_decoded_inputs

    def __call__(self,
                 predictions: Union[torch.IntTensor, List[str]],
                 gold_labels: Union[torch.IntTensor, List[str]] = None,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predictions``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predictions``.
        """
        bleu_costs = []
        if self._use_decoded_inputs:
            total_count = 0
            for ref, pred  in zip(gold_labels, 
                                  predictions):
                self._scorer.add_string(' '.join(ref),
                                        ' '.join(pred))
                bleu_costs.append(self._scorer.score()/100.0 + 1e-45)
                self._scorer.reset()
            bleu_cost =  -1 * torch.tensor(bleu_costs).to(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')

        else:
            predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
            bleu_cost = -1 * self._scorer.add(gold_labels.type(torch.IntTensor),
                                                predictions.type(torch.IntTensor)).score()

        self._scorer.reset()
        return bleu_cost

    @overrides
    def takes_decoded_input(self):
        return self._use_decoded_inputs