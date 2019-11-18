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
    "train_data_path": std.extVar("ARTIFICIAL_GRAMMAR_TRAIN"),
    "validation_data_path": std.extVar("ARTIFICIAL_GRAMMAR_DEV"), 
    "model": {
      "type": "quant_exp_composed_lm",
      "use_in_seq2seq_mode": false,
      "decoder": {
        "type": "quant_exp_searnn_decoder",
        "max_decoding_steps": 50,
        "generation_batch_size": 32, 
        "rollin_mode": "teacher_forcing",
        "rollout_mode": "reference",
        "decoder_net": {
          "type": "quant_exp_bias_lstm_cell",
          "decoding_dim": 300, 
          "target_embedding_dim": 300,
        },
        "target_embedder": {
          "vocab_namespace": "target_tokens",
          "embedding_dim": 300
        },
        "use_in_seq2seq_mode": false,
        "target_namespace": "target_tokens",
        "beam_size": 1,
        "use_bleu" : false,
        "dropout": 0.2,
        "sample_output": true,
        "start_token": "<S>",
        "end_token": "</S>",
        "oracle": {
          "type": "artificial_lang_oracle",
          "grammar_file": std.extVar("FSA_GRAMMAR_FILENAME"),
          "parallelize": true
        },
      }
    },
    "iterator": {
      "type": "basic",
      "batch_size": 512, 
    },
    "trainer": {
      "num_epochs": 50,
      "cuda_device" : 0,
      "optimizer": {
        "type": "adam",
        "lr": 0.01
      },
      "learning_rate_scheduler": {
          "type": "reduce_on_plateau",
          "factor": 0.5,
          "mode": "max",
          "patience": 0
      },
      "patience": 5, 
      "should_log_learning_rate": true,
      "log_batch_size_period": 50,
      "num_serialized_models_to_keep": -1
    },
  }
  
