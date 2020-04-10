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
    "vocabulary": {
      "directory_path": std.extVar("VOCAB_PATH"),
    },
    "train_data_path": std.extVar("TRAIN_FILE"),
    "validation_data_path": std.extVar("DEV_FILE"),
    "model": {
      "type": "quant_exp_composed_lm",
      "use_in_seq2seq_mode": false,
      "decoder": {
        "type": "quant_exp_reinforce_decoder",
        "max_decoding_steps": 50,
        "generation_batch_size": 32,
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
        "oracle": {
          "type": "artificial_lang_oracle",
          "grammar_file": std.extVar("FSA_GRAMMAR_FILENAME_SMOOTHED"),
          "parallelize": true,
          "max_len": 50,
        },
        "rollout_cost_function": {
          "type": "noisy_oracle",
          "oracle": {
            "type": "artificial_lang_oracle",
            "grammar_file": std.extVar("FSA_GRAMMAR_FILENAME"),
          },
        },
        // "rollout_cost_function": {
        //   "type": "hamming",
        // },
      },
      "initializer": [
          ["_decoder._decoder_net.*|_decoder._output_projection*|_decoder.target_embedder*|_decoder._dropout",
            {
              "type": "pretrained",
              "weights_file_path": std.extVar("WEIGHT_FILE_PATH"),
              "parameter_name_overrides": {},
            },
          ],
        ]
    },
    "iterator": {
      "type": "bucket",
      "sorting_keys": [["target_tokens", "num_tokens"]],
      "batch_size": 256,

      // This is needed stupidly for bucket iterator to work.
      "max_instances_in_memory": 50000
    },
    "validation_iterator": {
      "type": "basic",
      "batch_size": 1000
    },
    "trainer": {
      "num_epochs": 50,
      "cuda_device" : 0,
       "validation_metric": "-perplexity",
      "optimizer": {
        "type": "adam",
        "lr": 0.001
      },
      // "learning_rate_scheduler": {
      //     "type": "exponential",
      //     "gamma": 0.99,
      // },
      "patience": 10,
      "should_log_learning_rate": true,
      "log_batch_size_period": 50,
      "num_serialized_models_to_keep": -1
    },
  }
