from .lm_base import LMBase
from .lm_models import RNNModel, TransformerModel


class LSTMLM(LMBase):
    def __init__(self):
        pass

    @classmethod
    def build_model(cls,
                    model_name: str,
                    vocab_size: int,
                    device: torch.device,
                    config: Dict[str, Any]):
        """ TODO: Add docstring
        """
        return RNNModel(model_name, vocab_size,
                        config.emsize, config.nhid,
                        config.nlayers, config.dropout,
                        config.tied).to(device)

    def init_model_epoch(self):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        return self.model.init_hidden(self.batch_size)
