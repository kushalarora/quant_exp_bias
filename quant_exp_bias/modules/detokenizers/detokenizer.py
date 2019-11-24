from typing import List

from allennlp.common import Registrable
from allennlp.data.tokenizers.token import Token


class DeTokenizer(Registrable):
    """
    A ``DeTokenizer`` joins tokens to strings of text.  
    """

    #default_implementation = "default_detokenzier"


    def __call__(self, tokens_list: List[List[str]]) -> List[str]:
        """
        Actually implements detokenization by coverting list of tokens (in str form) to a string.

        Returns
        -------
        detokenized_str : ``str``
        """
        raise NotImplementedError
