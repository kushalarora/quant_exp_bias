local emnlp_gpt2_searnn_config = import "emnlp_news_gpt2.jsonnet";

local rollout_cost_function = {
          "type": "noisy_oracle",
          "add_brevity_penalty": true,
          "oracle": {
            "type": "gpt2_oracle",
            "model_name": "gpt2",
            "batch_size": 16,
            "cuda_device": -2,
          }
        };
local loss_criterion = {
          "type": "searnn-kl",
          "rollout_cost_function": rollout_cost_function,
          "temperature": 50,
      };

emnlp_gpt2_searnn_config + {
  'model'+: {
    'decoder'+: {
          "type": "lmpl_searnn_decoder",
          "generation_batch_size": 128,
          "rollin_mode":  std.extVar("rollin_mode"),
          "rollout_mode": std.extVar("rollout_mode"),
          "num_neighbors_to_add": 10,
          "num_tokens_to_rollout": 20,
          "rollout_ratio": 0.33,
          "loss_criterion": loss_criterion,
          "include_first": true,
          "include_last": true,
          "max_num_contexts": 10,
          // "rollout_reference_policy": "oracle",

    },
  },
  "data_loader"+: {
    "batch_sampler"+: {
      "batch_size": 3,
    },
  },
  "trainer"+: {
    "num_gradient_accumulation_steps": 6,
  },
}