# coding: utf-8
import itertools
import os
import json

from random import randint
from time import sleep

from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run, get_grammar_iterator, \
                             get_mean_std_results, get_result_iterator

args = get_experiment_args("artificial_language", "reinforce_ablation_experiments")

main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('artificial_lang/reinforce_ablation_experiments',
                                        output_dir=args.output_dir,
                                        param_path='experiments/artificial_language/training_configs/artificial_grammar_rl.jsonnet',
                                        debug=args.debug,
                                        offline=args.offline,
                                        experiment_text=args.exp_msg,
                                       )

num_samples_and_runs = [(10000, 4)]

def rollout_cost_function_configs(cost_func, mixing_coeff):
    if cost_func == 'bleu':
        overrides_dict = { "type": "bleu",}
        temperature = 10
    elif cost_func == 'noisy_oracle':
        overrides_dict=  {
            "type": "noisy_oracle",
            "oracle": {
                "type": "artificial_lang_oracle",
                "grammar_file":  os.environ["FSA_GRAMMAR_FILENAME_COST_FUNC"],
            }
        }
        temperature = 1
    overrides_dict = {
        "model": {
            "decoder": {
                "rollout_cost_function": overrides_dict,
                "rollin_rollout_mixing_coeff": mixing_coeff,
                "temperature": temperature,
            }
        }
    }
    experiment.log_parameters(overrides_dict)
    return json.dumps(overrides_dict)

import pdb; pdb.set_trace()
samples2pretrained_model = {
    1000: '/scratch/karora/quant_exp_bias/artificial_lang/dataset_experiments/04_30_2020_17_19_43/1000/0/',
}

def reinforce_experiments(main_args,
                          orig_serialization_dir,
                          param_path,
                          num_samples,
                          num_runs,
                          rollout_cost_funcs,
                          mixing_coeffs,
                        ):

    pretrained_model = samples2pretrained_model[num_samples]
    os.environ['VOCAB_PATH'] = os.path.join(pretrained_model, 'training/vocabulary')
    os.environ['WEIGHT_FILE_PATH'] = os.path.join(pretrained_model, 'training/best.th')

    # Setup variables needed later.
    for cost_func, mixing_coeff in itertools.product(rollout_cost_funcs, mixing_coeffs):
        serialization_dir = os.path.join(orig_serialization_dir, f'{cost_func}_{mixing_coeff}')
        for grammars_and_vocabularies in get_grammar_iterator(experiment,
                                                                args.grammar_templates, 
                                                                args.vocab_distributions,
                                                                num_runs):
            num_run, grammar_template_file, vocab_dist, \
                shall_generate_grammar_file, grammar_params = grammars_and_vocabularies
            run_metrics = one_exp_run(serialization_dir=serialization_dir,
                                        num_samples=num_samples,
                                        run=num_run,
                                        param_path=param_path,
                                        shall_generate_grammar_file=False,
                                        grammar_file_epsilon_0=os.path.join(
                                        pretrained_model, 'epsilon_0_grammar.txt'),
                                        grammar_file_epsilon=os.path.join(
                                        pretrained_model, 'epsilon_0.0001_grammar.txt'),
                                    )

            assert len(run_metrics) == 1, \
                'For this experiment, there should only be one final metric object for a run.'
            run_metrics = run_metrics[0]

            for result in get_result_iterator(run_metrics):
                experiment.log(result)

            mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
            mean_results.update(grammar_params)
            mean_results.update({
                'cost_func': cost_func,
                'mixing_coeff': mixing_coeff,
            })
            experiment.log(mean_results)


if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        reinforce_experiments(main_args, serialization_dir,
                              param_path, num_samples, num_runs,
                              args.rollout_cost_funcs, args.mixing_coeffs)
else:
    reinforce_experiments(main_args, serialization_dir,
                          param_path, args.num_samples, args.num_runs,
                          args.rollout_cost_funcs, args.mixing_coeffs)
