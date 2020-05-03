
# coding: utf-8
import itertools
import os

from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run, get_mean_std_results, \
                             get_result_iterator, get_grammar_iterator, \
                             get_rollout_cost_function_configs


args = get_experiment_args("artificial_language", "searnn_ablation_experiments")

main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('artificial_lang/searnn_ablation_experiments', 
                                        output_dir=args.output_dir,
                                        param_path='training_configs/artificial_grammar/artificial_grammar_searnn.jsonnet',
                                        offline=args.offline,
                                        debug=args.debug,
                                        experiment_text=args.exp_msg,
                                       )

rollin_rollout_cost_func_configs = [x for x in itertools.product(args.rollins, args.rollouts, args.rollout_cost_funcs, args.mixing_coeff)]

num_samples_and_runs = [(10000,4)]

def searnn_experiments(rollin_rollout_configs,
                        main_args,
                        serialization_dir,
                        param_path,
                        num_samples,
                        num_runs,
                      ):
    # Setup variables needed later.
    step = 0
    orig_serialization_dir = serialization_dir
    for rollin_policy, rollout_policy, cost_func, mixing_coeff in rollin_rollout_cost_func_configs:
        os.environ['rollin_mode'] = rollin_policy
        os.environ['rollout_mode'] = rollout_policy
        overrides_func=get_rollout_cost_function_configs("artificial_language", cost_func, 
                                                            mixing_coeff, args.temperature)

        serialization_dir = os.path.join(orig_serialization_dir, f'{rollin_policy}_{rollout_policy}_{cost_func}_{mixing_coeff}')
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
                                        overides_func=overrides_func, 
                                        grammar_template=grammar_template_file,
                                        shall_generate_grammar_file=shall_generate_grammar_file,
                                        vocabulary_distribution=vocab_dist,
                                    )

            assert len(run_metrics) == 1, \
                'For this experiment, there should only be one final metric object for a run.'
            run_metrics = run_metrics[0]
            
            for result in get_result_iterator(run_metrics):
                experiment.log_metrics(result, step=step)
                step += 1

            mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
            mean_results.update(grammar_params)
            mean_results.update({
                'rollin_policy': rollin_policy,
                'rollout_policy': rollout_policy,
                'cost_func': cost_func,
                'mixing_coeff': mixing_coeff,
            })
            experiment.log_metrics(mean_results, step=step)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        searnn_experiments(rollin_rollout_cost_func_configs, main_args,
                           serialization_dir, param_path, num_samples, num_runs)
else:
    searnn_experiments(rollin_rollout_cost_func_configs, main_args, serialization_dir, param_path, args.num_samples, args.num_runs)
