
# coding: utf-8
import itertools
import os

from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run, get_mean_std_results, \
                             get_result_iterator, get_grammar_iterator

args = get_experiment_args("artificial_language", "searnn_experiments")

main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('artificial_lang/searnn_experiments', 
                                        output_dir=args.output_dir,
                                        param_path='experiments/artificial_language/training_configs/artificial_grammar_searnn.jsonnet',
                                        debug=args.debug,
                                        offline=args.offline,
                                        experiment_text=args.exp_msg,
                                    )

rollin_rollout_configs = [x for x in itertools.product(args.rollins, args.rollouts)]

num_samples_and_runs = [(1000, 4), (10000,2), (100000,2)]

def searnn_experiments(rollin_rollout_configs,
                        main_args,
                        serialization_dir,
                        param_path,
                        num_samples,
                        num_runs,
                      ):
    # Setup variables needed later.
    orig_serialization_dir = serialization_dir
    for rollin_policy, rollout_policy in rollin_rollout_configs:
        os.environ['rollin_mode'] = rollin_policy
        os.environ['rollout_mode'] = rollout_policy

        serialization_dir = os.path.join(orig_serialization_dir, f'{rollin_policy}_{rollout_policy}')
        for grammars_and_vocabularies in get_grammar_iterator(experiment,
                                                                args.grammar_templates, 
                                                                args.vocab_distributions,
                                                                num_runs):
            num_run, grammar_template_file, vocab_dist, \
                shall_generate_grammar_file, grammar_params = grammars_and_vocabularies

            run_metrics,_ = one_exp_run(serialization_dir=serialization_dir, 
                                        num_samples=num_samples,
                                        run=num_run,
                                        param_path=param_path,
                                        oracle_config=args.oracle_config,
                                        grammar_template=grammar_template_file,
                                        shall_generate_grammar_file=shall_generate_grammar_file,
                                        vocabulary_distribution=vocab_dist,
                                    )

            assert len(run_metrics) == 1, \
                'For this experiment, there should only be one final metric object for a run.'
            run_metrics = run_metrics[0]

            for result in get_result_iterator(run_metrics):
                experiment.log(result)

            mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
            mean_results.update(grammar_params)
            mean_results.update({
                'rollin_policy': rollin_policy,
                'rollout_policy': rollout_policy,
            })
            experiment.log(mean_results)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        searnn_experiments(rollin_rollout_configs, main_args, serialization_dir, param_path, num_samples, num_runs)
else:
    searnn_experiments(rollin_rollout_configs, main_args, serialization_dir, param_path, args.num_samples, args.num_runs)
