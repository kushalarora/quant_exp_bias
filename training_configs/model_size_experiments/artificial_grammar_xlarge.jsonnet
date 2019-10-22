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
    "type": "quant_exp_lm",
    "target_namespace": "target_tokens",
    "target_output_dim": 300, 
    "target_embedding_dim": 1200,
    "num_decoder_layers": 2,
    "generation_batch_size": 32, 
    "max_decoding_steps": 50,
    "beam_size": 1,
    "use_bleu" : false,
    "dropout": 0.1,
    "oracle": {
      "type": "artificial_lang_oracle",
      "grammar_file": "grammar_templates/zipf_grammar_2_24.txt",
      "parallelize": true
    },
    "start_token": "<S>",
    "end_token": "</S>"
  },
  "iterator": {
    "type": "basic",
    "batch_size": 256
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
