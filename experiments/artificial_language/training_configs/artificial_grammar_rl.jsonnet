local artificial_grammar_rl_config = import "artificial_grammar_composed.jsonnet";

local rollout_cost_function = {
          "type": "noisy_oracle",
          "oracle": {
            "type": "artificial_lang_oracle",
            "grammar_file": std.extVar("FSA_GRAMMAR_FILENAME_COST_FUNC"),
          },
        };

local loss_criterion = {
          "type": "reinforce",
          "rollout_cost_function": rollout_cost_function,
      };

artificial_grammar_rl_config + {
    "vocabulary": {
      "type": "from_files",
      "directory": std.extVar("VOCAB_PATH"),
    },
    "model"+: {
      "decoder"+: {
        "type": "lmpl_reinforce_decoder",
        "loss_criterion": loss_criterion,
        "rollin_rollout_mixing_coeff": 0.5,
        "detach_rollin_logits": false,
      },
      "initializer": {
        "regexes": [
          ["_decoder._decoder_net.*|_decoder._output_projection*|_decoder.target_embedder*|_decoder._dropout",
            {
              "type": "pretrained",
              "weights_file_path": std.extVar("WEIGHT_FILE_PATH"),
              "parameter_name_overrides": {},
            },
          ],
        ],
      },
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
