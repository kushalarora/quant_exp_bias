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
      "directory_path": "results/artificial_grammar/artificial_lang/dataset_experiments/02_18_2020_17_13_06/1000/0/training/vocabulary"
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
              "weights_file_path": "results/artificial_grammar/artificial_lang/dataset_experiments/02_18_2020_17_13_06/1000/0/training/best.th",
              "parameter_name_overrides": {},
            },
          ],
        ]
    },
    "iterator": {
      "type": "bucket",
      "sorting_keys": [["target_tokens", "num_tokens"]],
      "batch_size": 32, 
      
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
      //     "type": "reduce_on_plateau",
      //     "factor": 0.5,
      //     "mode": "min",
      //     "patience": 2
      // },
      "patience": 10,
      "should_log_learning_rate": true,
      "log_batch_size_period": 50,
      "num_serialized_models_to_keep": -1
    },
  }
