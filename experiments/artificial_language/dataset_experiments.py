
# coding: utf-8
from experiments.util import initialize_experiments,  get_experiment_args, \
                             one_exp_run, get_grammar_iterator, \
                             get_mean_std_results, get_result_iterator

args = get_experiment_args("artificial_language", "dataset_experiments")

main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('artificial_lang/dataset_experiments',
                                        param_path='experiments/artificial_language/training_configs/artificial_grammar_composed.jsonnet',
                                        output_dir=args.output_dir, 
                                        debug=args.debug,
                                        offline=args.offline,
                                        experiment_text=args.exp_msg,
                                       )

dataset_experiments_params = [(100, 8), (1000,6) , (10000, 4), (100000, 2), (1000000, 1)]

def dataset_experiments(main_args, serialization_dir, param_path,
                        num_samples, num_runs,
                       ):
    step = 0
    for grammars_and_vocabularies in get_grammar_iterator(experiment,
                                                          args.grammar_templates, 
                                                          args.vocab_distributions,
                                                          num_runs):
        num_run, grammar_template_file, vocab_dist, \
            shall_generate_grammar_file, grammar_params = grammars_and_vocabularies
            
        run_metrics, run_serialization_dir = one_exp_run(serialization_dir=serialization_dir,
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
            experiment.log_metrics(result, step=step)
            step += 1

        mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
        mean_results.update(grammar_params)
        experiment.log_metrics(mean_results, step=step)
    return run_serialization_dir

if args.all:
    for num_samples, num_runs in dataset_experiments_params:
        dataset_experiments(main_args, serialization_dir, param_path, num_samples, num_runs)
else:
    dataset_experiments(main_args, serialization_dir, param_path, args.num_samples, args.num_runs)
