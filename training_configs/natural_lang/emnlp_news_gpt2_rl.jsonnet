{
    "dataset_reader": {
      "type": "quant_exp_language_modeling",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "namespace": "target_tokens"
        },
      },
      "tokenizer": {
        "type": "pretrained_transformer",
        "model_name": "gpt2",
      },
    },
    "vocabulary": {
        "directory_path": std.extVar("VOCAB_PATH"),
    },
    // "train_data_path": "data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered.2000000",
    "train_data_path": std.extVar("TRAIN_FILE"),
    "validation_data_path": std.extVar("DEV_FILE"),
    // "validation_data_path": "data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered.dev",
    "model": {
      "type": "quant_exp_composed_lm",
      "use_in_seq2seq_mode": false,
      "decoder": {
        "type": "quant_exp_reinforce_decoder",
        "generation_batch_size": 128,
        "max_decoding_steps": 50,
        "decoder_net": {
          "type": "quant_exp_bias_lstm_cell",
          "decoding_dim": 300, 
          "target_embedding_dim": 300,
          "num_decoder_layers": 1,
        },
        "target_embedder": {
          "vocab_namespace": "target_tokens",
          "embedding_dim": 300,
        },
        "use_in_seq2seq_mode": false,
        "target_namespace": "target_tokens",
        "beam_size": 1,
        "use_bleu" : false,
        "sample_output": true,
        "dropout": 0.2,
        "start_token": "@@@@",
        "end_token": "####",
        "oracle": {
          "type": "gpt2_oracle",
          "model_name": "gpt2",
          "batch_size": 10,
        },
        "detokenizer": {
          "type": "gpt2_detokenizer",
          "model_name": "gpt2"
        },
        "rollout_cost_function": {
          "type": "noisy_oracle",
          "add_brevity_penalty": true,
          "oracle": {
            "type": "gpt2_oracle",
            "model_name": "distilgpt2",
            "batch_size": 16,
            "cuda_device": -2,
          },
        },
        "rollout_ratio": 0.33,
        "rollin_rollout_mixing_coeff": 0.5,
        "detach_rollin_logits": false,
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
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 16  

    }
  },
  "validation_data_loader": {
      "batch_sampler": {
          "type": "bucket",
          "batch_size": 32
      }
  },
  "trainer": {
    "num_epochs": 20,
    "opt_level": "O2",
    "cuda_device" : 0,
    "grad_clipping": 5.0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001,
    },
    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "min",
        "patience": 0
    },
    "patience": 10,
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
  }
}
