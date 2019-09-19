from allennlp.data.vocabulary import Vocabulary
from quant_exp_bias.models.lms.lm import LMBase
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction

@Model.register('quant_exp_lm')
class LMQuantExpModel(LMBase):
    """
    TODO (Kushal): Add doc string.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 max_decoding_steps: int,
                 target_embedding_dim: int,
                 target_output_dim: int,
                 generation_batch_size: int,
                 target_namespace: str = "tokens",
                 beam_size: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 use_bleu: bool = True,
                 dropout: float = None) -> None:
        
        super().__init__(vocab=vocab,
                         use_in_seq2seq_mode=False,
                         max_decoding_steps=max_decoding_steps,
                         target_embedding_dim=target_embedding_dim,
                         target_output_dim=target_output_dim,
                         generation_batch_size=generation_batch_size,
                         target_namespace=target_namespace,
                         beam_size=beam_size,
                         scheduled_sampling_ratio=scheduled_sampling_ratio,
                         use_bleu=use_bleu,
                         dropout=dropout)
