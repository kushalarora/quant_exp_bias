local emnlp_gpt2_searnn_config = import "emnlp_news_gpt2.jsonnet";

local oracle =  {
            "type": "gpt2_oracle",
            "model_name": "gpt2",
            "batch_size": 16,
            "cuda_device": -2,
          };

local rollout_cost_function = {
          "type": "noisy_oracle",
          "add_brevity_penalty": false,
          "oracle": oracle,
          "log_cost": true,
        };

local loss_criterion = {
          "type": "searnn-kl",
          "rollout_cost_function": rollout_cost_function,
          // "rollin_rollout_mixing_coeff": 0.50,
          "normalize_kl_only_over_samples": false,
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
          "num_random_tokens_to_add": 11,
          "num_tokens_to_rollout": 12,
          "rollout_ratio": 0.33,
          "loss_criterion": loss_criterion,
          "include_first": true,
          "include_last": false,
          "max_num_contexts": 20,
          // "add_noise_to_sampling": false,
          "max_sampling_noise": 1e-3,
          "sampling_temperature": 1,
          // "rollout_reference_policy": "oracle",
          // "token_based_metric": {
          //   "type": "exp_bias",
          //   "oracle": oracle,
          // },
    },
  },
  "data_loader"+: {
    "batch_sampler"+: {
      "batch_size": 2,
    },
  },
  "trainer"+: {
    "num_epochs": 20,
    "validation_metric": "-loss",
    "num_gradient_accumulation_steps": 16,
    "use_amp": false,
  },
}