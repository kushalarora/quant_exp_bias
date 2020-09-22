local emnlp_gpt2_rl_config = import "emnlp_news_gpt2.jsonnet";
local warm_start_model = std.extVar("WARM_START_MODEL");

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
local loss_criterion = {
          "type": "reinforce",
          "temperature": 1,
          "rollout_cost_function": rollout_cost_function,
          "detach_rollin_logits": false,
      };
emnlp_gpt2_rl_config + {
      "vocabulary": {
        "type": "from_files",
        "directory": warm_start_model + "/vocabulary",
      },
      "train_data_path": warm_start_model + "../data/oracle_samples-train.txt",
      "validation_data_path": warm_start_model + "../data/oracle_samples-dev.txt",
      "model"+: {
        "decoder"+: {
          "type": "lmpl_reinforce_decoder",
          "generation_batch_size": 128,
          "loss_criterion": loss_criterion,
          "rollout_ratio": 0.5,
          "include_last": true,
          "include_first": true,
        },
        "initializer": {
          "regexes": [
            ["_decoder._decoder_net.*|_decoder._output_projection*|_decoder.target_embedder*|_decoder._dropout",
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
          "batch_size": 20,
        },
      },
      "trainer"+: {
        "validation_metric": "-cost",
      },
    }
