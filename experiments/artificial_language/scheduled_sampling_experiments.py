# coding: utf-8
import os

from experiments.util import initialize_experiments, get_experiment_args, \
                                calculate_ss_k, one_exp_run, get_grammar_iterator, \
                                get_mean_std_results, get_result_iterator, \
                                get_scheduled_sampling_overrides_func

args = get_experiment_args("artificial_language", "scheduled_sampling_experiments")

k = lambda ratio_level: calculate_ss_k(args.num_samples, args.batch_size, 
                                        args.num_batches, ratio_level=ratio_level)

scheduled_sampling_ratios  = [
        ('uniform', 0.0, -1),('uniform', 0.05, -1), ('uniform', 0.1, -1), ('uniform', 0.25, -1), ('uniform', 0.5, -1),  # Fixed SS ratio
        ('quantized', 1.0, k(0.25)),  # Quantized increase ss ratio.
        ('quantized', 1.0, k(0.5)),  # Quantized increase ss ratio.
        ('quantized', 1.0, k(0.75)),  # Quantized increase ss ratio.
        ('linear', 1.0, k(0.25)),  # Linearly increase ss ratio.
        ('linear', 1.0, k(0.5)),  # Linearly increase ss ratio.
        ('linear', 1.0, k(0.75)),  # Linearly increase ss ratio.
]

main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('artificial_lang/scheduled_sampling_experiments',
                                        param_path='experiments/artificial_language/training_configs/artificial_grammar_composed.jsonnet',
                                        output_dir=args.output_dir,
                                        debug=args.debug,
                                        offline=args.offline,
                                        experiment_text=args.exp_msg,
                                       )

num_samples_and_runs = [(1000, 1), (10000,1), (100000,1)]

def scheduled_sampling_experiments(scheduled_sampling_ratios,
                                    main_args,
                                    serialization_dir,
                                    param_path,
                                    num_samples,
                                    num_runs,
                                  ):
    step = 0
    orig_serialization_dir = serialization_dir
    for ss_type, ss_ratio, ss_k in scheduled_sampling_ratios:
        serialization_dir = os.path.join(orig_serialization_dir, f'{ss_type}_{ss_ratio}_{ss_k}')
        overrides_func = get_scheduled_sampling_overrides_func(ss_type, ss_ratio, ss_k)
        
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
                                        grammar_template=grammar_template_file,
                                        shall_generate_grammar_file=shall_generate_grammar_file,
                                        vocabulary_distribution=vocab_dist,
                                        overides_func=overrides_func,
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
                    'scheduled_sampling_ratio': ss_ratio,
                    'scheduled_sampling_k': ss_k,
                    'scheduled_sampling_type': ss_type,
                    'final_ss_ratio': run_metrics['validation_ss_ratio'],
                    'best_val_ss_ratio': run_metrics['best_validation_ss_ratio'],
            })
            experiment.log_metrics(mean_results, step=step)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        scheduled_sampling_experiments(scheduled_sampling_ratios,
                                        main_args,
                                        serialization_dir,
                                        param_path,
                                        num_samples,
                                        num_runs)
else:
    scheduled_sampling_experiments(scheduled_sampling_ratios,
                                main_args,
                                serialization_dir,
                                param_path,
                                args.num_samples,
                                args.num_runs)
