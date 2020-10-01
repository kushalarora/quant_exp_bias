local emnlp_gpt2_searnn_config = import "emnlp_news_gpt2.jsonnet";

local rollout_cost_function = {
          "type": "noisy_oracle",
          "add_brevity_penalty": true,
          "oracle": {
            "type": "gpt2_oracle",
            "model_name": "distilgpt2",
            "batch_size": 16,
            "cuda_device": -2,
          },
          "log_cost": true,
        };
local loss_criterion = {
          "type": "searnn-kl",
          "rollout_cost_function": rollout_cost_function,
          "temperature": 10,
      };

emnlp_gpt2_searnn_config + {
  'model'+: {
    'decoder'+: {
          "type": "lmpl_searnn_decoder",
          "generation_batch_size": 128,
          "rollin_mode":  std.extVar("rollin_mode"),
          "rollout_mode": std.extVar("rollout_mode"),
          "num_neighbors_to_add": 0,
          "num_random_tokens_to_add": 0,
          "num_tokens_to_rollout": 20,
          "rollout_ratio": 0.15,
          "loss_criterion": loss_criterion,
          "include_first": true,
          "include_last": true,
          "max_num_contexts": 12,
          // "max_sampling_noise": 1e-2,
          "sampling_temperature": 10,
          // "rollout_reference_policy": "oracle",
    },
  },
  "data_loader"+: {
    "batch_sampler"+: {
      "batch_size": 3,
    },
  },
  "trainer"+: {
    "validation_metric": "-loss",
    "num_gradient_accumulation_steps": 1,
    "use_amp": false,
  },
}