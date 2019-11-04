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
  "model": {
    "type": "quant_exp_lm",
    "target_namespace": "target_tokens",
    "target_output_dim": 1200, 
    "target_embedding_dim": 400,
    "num_decoder_layers": 4,
    "generation_batch_size": 32, 
    "max_decoding_steps": 400,
    "beam_size": 1,
    "use_bleu" : false,
    "dropout": 0.5,
    "start_token": "<S>",
    "end_token": "</S>"
  },
  "iterator": {
    "type": "basic",
    "batch_size": 128
  },
  "trainer": {
    "num_epochs": 100,
    "cuda_device" : 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "max",
        "patience": 0
    },
    "patience": 5,
    "should_log_learning_rate": true,
    "log_batch_size_period": 500,
  }
}
