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
  "train_data_path": "data/ptb/ptb.train.txt",
  "validation_data_path": "data/ptb/ptb.valid.txt",
  //"train_data_path": "test/data/lm/sentences.txt",
  //"validation_data_path": "test/data/lm/sentences.txt",
  "model": {
    "type": "quant_exp_lm",
    "target_namespace": "target_tokens",
    "target_output_dim": 300, 
    "target_embedding_dim": 300,
    "generation_batch_size": 32, 
    "max_decoding_steps": 400,
    "beam_size": 1,
    "use_bleu" : false,
    "dropout": 0.5,
    "oracle": {
      "type": "artificial_lang_oracle",
      "num_samples": 100 
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 256
  },
  "trainer": {
    "num_epochs": 100,
    "cuda_device" : 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "patience": 5,
    "log_batch_size_period": 50
  }
}
