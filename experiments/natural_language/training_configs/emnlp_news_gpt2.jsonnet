local learning_rate_scheduler = {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "min",
        "patience": 0
    };
local optimizer = {
      "type": "adam",
      "lr": 0.001,
    };

local validation_metric = '-perplexity';

local batch_size = 96; 

local decoder_type = "lmpl_auto_regressive_seq_decoder";
{
    "dataset_reader": {
      "type": "lmpl_language_modeling",
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
        "generation_batch_size": 256,
        "max_decoding_steps": 50,
        "decoder_net": {
          "type": "lmpl_lstm_cell",
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
        "dropout": 0.2,
        "start_token": "@@@@",
        "end_token": "####",
        "detokenizer": {
          "type": "gpt2_detokenizer",
          "model_name": "gpt2"
        },
      }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": batch_size,

    }
  },
  "trainer": {
    "num_epochs": 20,
    "validation_metric": validation_metric,
    "cuda_device" : 0,
    "grad_clipping": 5.0,
    "optimizer": optimizer,
    "learning_rate_scheduler": learning_rate_scheduler,
    "patience": 5,
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
  }
}
