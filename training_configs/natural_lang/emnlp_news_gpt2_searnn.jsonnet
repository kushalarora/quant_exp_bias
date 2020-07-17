local emnlp_gpt2_searnn_config = import "emnlp_news_gpt2.jsonnet";

local rollout_cost_function = {
                                "type": "noisy_oracle",
                                "add_brevity_penalty": true,
                                "oracle": {
                                  "type": "gpt2_oracle",
                                  "model_name": "distilgpt2",
                                  "batch_size": 16,
                                  "cuda_device": -2,
                                }
                              };

emnlp_gpt2_searnn_config + {
  'model': {
    'decoder': {
          "type": "quant_exp_searnn_decoder",
          "generation_batch_size": 128,
          "rollin_mode":  std.extVar("rollin_mode"),
          "rollout_mode": std.extVar("rollout_mode"),
          "temperature": 5,
          "num_neighbors_to_add": 0,
          "num_tokens_to_rollout": 25,
          "rollout_ratio": 0.25,
          "rollout_cost_function": rollout_cost_function,
    },
  },
  "data_loader": {
    "batch_sampler": {
      "batch_size": 16,
    },
  },
  "trainer": {
    "validation_metric": "-loss",
  },
}