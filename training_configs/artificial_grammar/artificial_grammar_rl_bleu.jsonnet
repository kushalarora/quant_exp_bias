local artificial_grammar_rl_blue_config = import "artificial_grammar_rl.jsonnet";

local rollout_cost_bleu_function = {
          "type": "bleu",
        };

artificial_grammar_rl_blue_config + {
    "model"+: {
      "decoder"+: {
        "rollout_cost_function": rollout_cost_bleu_function,
      },
    },
  }