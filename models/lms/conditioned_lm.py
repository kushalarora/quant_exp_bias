

from allennlp.data.vocabulary import Vocabulary
from models.lms.lm import LMBase
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction

@Model.register('quant_exp_seq2seq')
class Seq2SeqQuantExpModel(LMBase):
    """
    TODO (Kushal): Add doc string.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 max_decoding_steps: int,
                 target_embedding_dim: int,
                 target_namespace: str = "tokens",
                 beam_size: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 use_bleu: bool = True,
                
                 # This fields will only come into play in Seq2Seq mode.
                 source_embedder: TextFieldEmbedder = None,
                 encoder: Seq2SeqEncoder = None,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None) -> None:
        
        super().__init__(vocab=vocab,
                         use_in_seq2seq_mode=True,
                         max_decoding_steps=max_decoding_steps,
                         generation_batch_size=10,
                         target_embedding_dim=target_embedding_dim,
                         target_namespace=target_namespace,
                         beam_size=beam_size,
                         scheduled_sampling_ratio=scheduled_sampling_ratio,
                         use_bleu=use_bleu,
                         source_embedder=source_embedder,
                         encoder=encoder,
                         attention=attention,
                         attention_function=attention_function)
