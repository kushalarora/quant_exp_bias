local artificial_grammar_searnn_config = import "artificial_grammar_composed.jsonnet";

local rollout_cost_function = {
          "type": "noisy_oracle",
          "oracle": {
            "type": "artificial_lang_oracle",
            "grammar_file": std.extVar("FSA_GRAMMAR_FILENAME_COST_FUNC"),
          },
        };

local loss_criterion = {
          "type": "searnn-kl",
          "rollout_cost_function": rollout_cost_function,
          "temperature": 1,
      };

artificial_grammar_searnn_config + {
    "model"+: {
      "decoder"+: {
        "type": "lmpl_searnn_decoder",
        "loss_criterion": loss_criterion,
        "rollin_mode":  std.extVar("rollin_mode"),
        "rollout_mode": std.extVar("rollout_mode"),
        "rollout_ratio": 0.25,
        "rollin_rollout_mixing_coeff": 0.0,
      }
    },
    "data_loader"+: {
      "batch_sampler"+: {
        "batch_size": 64,
      },
    },
    "trainer"+: {
      "validation_metric": "-loss",
    },
  }
