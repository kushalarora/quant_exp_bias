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
      "type": "quant_exp_composed_lm",
      "use_in_seq2seq_mode": false,
      "decoder": {
        "type": "quant_exp_auto_regressive_seq_decoder",
        "max_decoding_steps": 400,
        "generation_batch_size": 32, 
        "decoder_net": {
          "type": "quant_exp_bias_lstm_cell",
          "decoding_dim": 1200, 
          "target_embedding_dim": 400,
          # This doesn't seem to be working as of
          # now.
          // "num_decoder_layers": 4,
        },
        "target_embedder": {
          "vocab_namespace": "target_tokens",
          "embedding_dim": 400
        },
        "use_in_seq2seq_mode": false,
        "target_namespace": "target_tokens",
        "beam_size": 1,
        "use_bleu" : false,
        "dropout": 0.2,
        "sample_output": true,
        "start_token": "<S>",
        "end_token": "</S>",
        




      }
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
