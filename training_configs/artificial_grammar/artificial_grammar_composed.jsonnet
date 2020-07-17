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
    "train_data_path": std.extVar("TRAIN_FILE"),
    "validation_data_path": std.extVar("DEV_FILE"),
    "model": {
      "type": "quant_exp_composed_lm",
      "use_in_seq2seq_mode": false,
      "decoder": {
        "type": "quant_exp_auto_regressive_seq_decoder",
        "max_decoding_steps": 50,
        "decoder_net": {
          "type": "quant_exp_bias_lstm_cell",
          "decoding_dim": 100,
          "target_embedding_dim": 100,
        },
        "target_embedder": {
          "vocab_namespace": "target_tokens",
          "embedding_dim": 100
        },
        "use_in_seq2seq_mode": false,
        "target_namespace": "target_tokens",
        "beam_size": 1,
        "use_bleu" : false,
        "dropout": 0.2,
        "sample_output": true,
        "start_token": "<S>",
        "end_token": "</S>",
        "mask_pad_and_oov": true,
        "oracle": {
          "type": "artificial_lang_oracle",
          "grammar_file": std.extVar("FSA_GRAMMAR_FILENAME"),
          "parallelize": true,
          "num_threads": 32,
        },
      }
    },
    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size": 128,  
      }
    },
    "trainer": {
      "num_epochs": 50,
      "cuda_device" : 0,
      "validation_metric": "-perplexity",
      "grad_clipping": 5.0,
      "optimizer": {
        "type": "adam",
        "lr": 0.01
      },
      // "learning_rate_scheduler": {
      //     "type": "exponential",
      //     "gamma": 0.99,
      // },
      "learning_rate_scheduler": {
          "type": "reduce_on_plateau",
          "factor": 0.5,
          "mode": "min",
          "patience": 2
      },
      "patience": 10,
      "checkpointer": {
        "num_serialized_models_to_keep": 1,
      },
    },
  }

