local learning_rate_scheduler = {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "min",
        "patience": 3
    };
local optimizer = {
      "type": "adam",
      "lr": 0.001,
    };

local validation_metric = '-perplexity';

local gpt2_dir="/home/karora/scratch/huggingface/gpt2/";

local dropout_ratio = 0.4;

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
        "model_name": gpt2_dir,
        "start_tokens": ["@@@@"],
        "end_tokens": ["####"],
        // "do_lowercase": false,
      },
    };

local oracle = {
          "type": "gpt2_oracle",
          "model_name": gpt2_dir,
          "batch_size": 16,
          "cuda_device": 0,
      };

local decoder_type = "lmpl_auto_regressive_seq_decoder";

local distributed = std.extVar("DISTRIBUTED");
local ngpu=std.parseInt(std.extVar("NUM_GPUS"));

local wandb_run_name=std.extVar("WANDB_RUN_NAME");
local wandb_project_name=std.extVar("WANDB_PROJECT_NAME");

local batch_size = 48;

local gpus(ngpu) =
  if ngpu == 1 then [0]
  else if ngpu == 2 then [0, 1]
  else if ngpu == 3 then [0, 1, 2]
  else if ngpu == 4 then [0, 1, 2, 3]
  else error "invalid option: " + std.manifestJson(ngpu);

local stringToBool(s) =
  if s == "true" then true
  else if s == "false" || s == '' || s == null then false
  else error "invalid boolean: " + std.manifestJson(s);

{
    "random_seed": null,
    "numpy_seed": null,
    "pytorch_seed": null,
    // "dataset_reader": dataset_reader,
    "dataset_reader": {
      "type": "sharded",
      "base_reader": dataset_reader,
    },
    // "validation_dataset_reader": dataset_reader,
    "vocabulary": {
      "type": "from_files",
      "directory": "experiments/natural_language/vocab/",
    },
    // "evaluate_on_test": true,
    // "train_data_path": "data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered.2000000",
    "train_data_path": std.extVar("TRAIN_FILE"),
    "validation_data_path": std.extVar("DEV_FILE"),
    // "test_data_path": "data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered.test",
    // "test_data_path": "data/ted_talks.txt",
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
          "labeling_smooting_ratio": 0.0,
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
        // "token_based_metric": {
        //   "type": "oracle_likelihood",
        //   "oracle": oracle,
        //   "log_cost": true,
        // },
      },
      "detokenizer": {
        "type": "gpt2_detokenizer",
        "model_name": gpt2_dir,
      },
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": batch_size/ngpu,
      "sorting_keys": ["target_tokens"],
    },
    // "num_workers": 1,
    // "pin_memory": true,
  },
  "trainer": {
    "num_epochs": 20,
    "validation_metric": validation_metric,
    "cuda_device" : 0,
    // "use_amp": true,
    "grad_norm": 0.1,
    "optimizer": optimizer,
    "learning_rate_scheduler": learning_rate_scheduler,
    "patience": 5,
    "checkpointer": {
      "keep_most_recent_by_count": 20,
    },
    "callbacks": [{
          "type": 'tensorboard',
          // "project": wandb_project_name,
          // "name": wandb_run_name,
          "should_log_learning_rate": true,
          "summary_interval": 10,
          // "watch_model": false,
          // "files_to_save": [],
    }],
    "num_gradient_accumulation_steps": 1,

  },
  "distributed":  if stringToBool(distributed) then { "cuda_devices": gpus(ngpu),} else null,
}
