# coding: utf-8
import os

from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run,  get_grammar_iterator, \
                             get_mean_std_results, get_result_iterator, \
                             get_model_overrides_func

args = get_experiment_args("artificial_language", "model_size_experiments")

main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('artificial_lang/model_size_experiments', 
                                        param_path='experiments/artificial_language/training_configs/artificial_grammar_composed.jsonnet',
                                        output_dir=args.output_dir,
                                        debug=args.debug,
                                        offline=args.offline,
                                        experiment_text=args.exp_msg,
                                       )

model_size_configs  = {
    'xsmall' : (10, 10, 1),
    'small': (50, 50, 1),
    'medium': (100, 100, 1),
    'large': (300, 300, 1),
    'xlarge': (300, 1200, 1)
}
experiment.log_parameters(model_size_configs)

num_samples_and_runs = [(1000, 8), (10000,4), (100000,2)]

def model_size_experiments(model_sizes,
                            main_args, 
                            serialization_dir, 
                            param_path,
                            num_samples,
                            num_runs,
                           ):
    step = 0
    # Setup variables needed later.
    orig_serialization_dir = serialization_dir
    for model_size in model_sizes:
        model_overrides_func = get_model_overrides_func(*model_size_configs[model_size])
        serialization_dir = os.path.join(orig_serialization_dir, model_size)

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
                                        overides_func=model_overrides_func,
                                     )
        
            assert len(run_metrics) == 1, \
                'For this experiment, there should only be one final metric object for a run.'
            run_metrics = run_metrics[0]

            for result in get_result_iterator(run_metrics):
                result['model_size'] = model_size
                experiment.log_metrics(result, step=step)
                step += 1

            mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
            mean_results['model_size'] = model_size
            mean_results.update(grammar_params)
            experiment.log_metrics(mean_results, step=step)

    experiment.log_asset_folder(serialization_dir, log_file_name=True, recursive=True)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        model_size_experiments(args.model_sizes, main_args, serialization_dir, param_path, num_samples, num_runs)
else:
    model_size_experiments(args.model_sizes, main_args, serialization_dir, param_path, args.num_samples, args.num_runs)
