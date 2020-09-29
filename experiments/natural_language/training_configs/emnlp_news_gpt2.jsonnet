local learning_rate_scheduler = {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "min",
        "patience": 2
    };
local optimizer = {
      "type": "adam",
      "lr": 0.001,
    };

local validation_metric = '-perplexity';

local batch_size = 48; 

local dropout_ratio = 0.2;

local dataset_reader =  {
      "type": "lmpl_language_modeling",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "namespace": "target_tokens"
        },
      },
      "tokenizer": {
        // "type": "pretrained_transformer",
        "type": "qeb_pretrained_transformer",
        "model_name": "gpt2",
        "start_tokens": ["@@@@"],
        "end_tokens": ["####"],
        // "do_lowercase": false,
      },
    };

local oracle = {
          "type": "gpt2_oracle",
          "model_name": "gpt2",
          "batch_size": 16,
          "cuda_device": 0,
      };

local decoder_type = "lmpl_auto_regressive_seq_decoder";

{
    "random_seed": null,
    "dataset_reader": dataset_reader,
    // {
    //   "type": "sharded",
    //   "base_reader": dataset_reader,
    // },
    // "vocabulary": {
    //    "directory_path": "training_configs/natural_lang/vocab/",
    // },
    // "train_data_path": "data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered.2000000",
    "train_data_path": std.extVar("TRAIN_FILE"),
    "validation_data_path": std.extVar("DEV_FILE"),
    // "validation_data_path": "data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered.dev",
    "model": {
      "type": "lmpl_composed_lm",
      "use_in_seq2seq_mode": false,
      "decoder": {
        "type": decoder_type,
        "generation_batch_size": 64,
        "max_decoding_steps": 50,
        "decoder_net": {
          "type": "lmpl_lstm_cell",
          "decoding_dim": 300, 
          "target_embedding_dim": 300,
          "num_decoder_layers": 1,
          "dropout": dropout_ratio,
        },
        "target_embedder": {
          "vocab_namespace": "target_tokens",
          "embedding_dim": 300,
        },
        "loss_criterion": {
          "type": "mle",
        },
        "use_in_seq2seq_mode": false,
        // "mask_padding_and_start": false,
        "target_namespace": "target_tokens",
        "beam_size": 1,
        "use_bleu" : false,
        "dropout": dropout_ratio,
        "start_token": "@@@@",
        "end_token": "####",
        "sample_rollouts": true,
        "detokenizer": {
          "type": "gpt2_detokenizer",
          "model_name": "gpt2"
        },
        // "token_based_metric": {
        //   "type": "oracle_likelihood",
        //   "oracle": oracle,
        //   "log_cost": true,
        // },
      }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": batch_size,
      "sorting_keys": ["target_tokens"],
    },
    "num_workers": 1,
    "pin_memory": true,
  },
  "trainer": {
    "num_epochs": 20,
    "validation_metric": validation_metric,
    "cuda_device" : 0,
    // "use_amp": true,
    "grad_clipping": 1.0,
    "optimizer": optimizer,
    "learning_rate_scheduler": learning_rate_scheduler,
    "patience": 5,
    "checkpointer": {
      "num_serialized_models_to_keep": 20,
    },
  },
}
