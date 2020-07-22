{
  "dataset_reader": {
    "type": "quant_exp_language_modeling",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "target_tokens"
      },
    },
    "start_tokens": ["<S>"],
    "end_tokens": ["</S>"]
  },
  "train_data_path": "test/data/lm/sentences.txt",
  "validation_data_path": "test/data/lm/sentences.txt",
  "model": {
    "type": "quant_exp_lm",
    "target_namespace": "target_tokens",
    "target_output_dim": 10, 
    "target_embedding_dim": 30,
    "generation_batch_size": 32, 
    "max_decoding_steps": 20,
    "beam_size": 5
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 10,
    "cuda_device" : -1,
    "optimizer": {
      "type": "sgd",
      "lr": 0.01
    },
    "log_batch_size_period": 1
  }
}
