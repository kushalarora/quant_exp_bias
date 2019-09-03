from lms import LMBase, LSTMLM
from oracles import OracleBase, ArtificialLanguageOracle
from typing import List


def compute_exposure_bias(oracle_model: OracleBase,
                          trained_lm_model: LMBase,
                          samples):
    pass
# This will Compute exposure bias (compute_exposure_bias method):
# For n times (this is for averaging due to sampling):
# Sample dataset from Oracle
# Train model on this sampled dataset
# Generate samples from the model.
# Compute Exposure Bias
# Add this to pandas object.

# This samples dataset from Oracle (sample_training_dataset):
# Upto dataset size:
# Sample a sequence from dataset.


def sample_training_dataset():
    return None

# This trains the model (train_lm_model):
# Take the dataset.
# Preprocess (like BPTT or splitting into <seq_len> sequences).
# for epoch in range(epochs):
    # train_lm


def train_lm_model():
    return None

# This generates sample from model (generate_samples):
# Take the model as input, and distribution over seq lengths.
# Return an iter over samples.


def generate_samples():
    return None

# For different dataset sizes:
    # compute exposure bias

# For small, medium, large models:
    # Compute exposure bias.

# For teacher forcing, scheduled sampling, differentiable scheduled sampling:
    # Compute exposure bias

# For different beam sizes, (beam-size=0, 1, 2, 4, 8, 16, 32):
    # Compute exposure bias

# For different global methods:
    # Compute exposure bias.
