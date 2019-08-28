from oracle_base import OracleBase

class FSAOracle(OracleBase):
    def __init__(self):
        pass

    def sample_training_set(self):
        return NotImplementedError

    def compute_sent_probs(self, sentences):
        return NotImplementedError


