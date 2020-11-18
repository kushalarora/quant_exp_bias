local emnlp_gpt2_rl_config = import "emnlp_news_gpt2.jsonnet";
local warm_start_model = std.extVar("WARM_START_MODEL");

local rollout_cost_function = {
                                "type": "noisy_oracle",
                                "add_brevity_penalty": true,
                                // "log_cost": false,
                                "log_cost": true,
                                "oracle": {
                                  "type": "gpt2_oracle",
                                  "model_name": "gpt2",
                                  "batch_size": 16,
                                  "cuda_device": -2,
                                }
                              };
local loss_criterion = {
          "type": "risk",
          // "type": "reinforce",
          "temperature": 1,
          "rollout_cost_function": rollout_cost_function,
          "detach_rollin_logits": false,
          "entropy_regularization_coeff": 1e-5,
          // "rollin_rollout_mixing_coeff": 0.001,
          "normalize_by_mean_std": true,
      };
emnlp_gpt2_rl_config + {
      "vocabulary": {
        "type": "from_files",
        "directory": warm_start_model + "/vocabulary",
      },
      "train_data_path": warm_start_model + "/../data/oracle_samples-train_*.txt",
      "validation_data_path": warm_start_model + "/../data/oracle_samples-dev_*.txt",
      "model"+: {
        "decoder"+: {
          "type": "lmpl_reinforce_decoder",
          "generation_batch_size": 128,
          "loss_criterion": loss_criterion,
          "rollout_ratio": 1.0,
          "max_num_contexts": 10,
          "min_num_contexts": 10,
          // "rollout_iter_start_pct": 10,
          // "rollout_iter_end_pct": 80,
          "include_first": true,
          "include_last": true,
          // "beam_size": 2,
        },
        "initializer": {
          "regexes": [
            ["_decoder._decoder_net.*|_decoder._output_projection*|_decoder.target_embedder*",
              {
                "type": "pretrained",
                "weights_file_path": warm_start_model + "/best.th",
              },
            ],
          ],
        },
      },
      "data_loader"+: {
        "batch_sampler"+: {
          "batch_size": 5,
        },
      },
      "trainer"+: {
        "validation_metric": "-cost",
        "optimizer"+: {
          // "type": "sgd",
          "lr": 0.0001,
        },
        "num_epochs": 10,
        "num_gradient_accumulation_steps": 15,
      },
    }
