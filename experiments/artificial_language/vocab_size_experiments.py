
# coding: utf-8
import itertools
import os

from experiments.util import initialize_experiments,  get_experiment_args, \
                             one_exp_run, get_mean_std_results, get_result_iterator

args = get_experiment_args("artificial_language", "vocabulary_experiments")

main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('artificial_lang/vocabulary_size_experiments', 
                                        param_path='experiments/artificial_language/training_configs/artificial_grammar_composed.jsonnet',
                                        output_dir=args.output_dir,
                                        debug=args.debug,
                                        offline=args.offline,
                                        experiment_text=args.exp_msg,
                                    )

vocabulary_sizes  = [6, 12, 24, 48]
vocab_distributions = ['zipf', 'uniform']
grammar_templates = ['grammar_templates/grammar_1.template', \
                     'grammar_templates/grammar_2.template', \
                     'grammar_templates/grammar_3.template' ]

num_samples_and_runs = [(10000, 4)]

grammar_vocab_size_and_dist = [x for x in itertools.product(grammar_templates, 
                                                            vocabulary_sizes, 
                                                            vocab_distributions)]

def vocabulary_size_experiments(grammar_vocab_size_and_dist,
                                main_args,
                                serialization_dir,
                                param_path,
                                num_samples,
                                num_runs,
                               ):

    orig_serialization_dir = serialization_dir
    for grammar_template, size, dist in grammar_vocab_size_and_dist:
        vsexp_serialization_dir = os.path.join(orig_serialization_dir,  
                                                f'{grammar_template}_{dist}_{size}')
        for num_run in range(num_runs):
            run_metrics,_ = one_exp_run(serialization_dir=vsexp_serialization_dir,
                                        num_samples=num_samples,
                                        run=num_run,
                                        param_path=param_path,
                                        oracle_config=args.oracle_config,
                                        grammar_template=grammar_template,
                                        vocabulary_size=size,
                                        vocabulary_distribution=dist,
                                        shall_generate_grammar_file=True,
                                     )

            assert len(run_metrics) == 1, \
                'For this experiment, there should only be one final metric object for a run.'
            run_metrics = run_metrics[0]
        
            for result in get_result_iterator(run_metrics):
                experiment.log(result)

            mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
            mean_results.update({
                'grammar_template': grammar_template,
                'vocab_distribution': dist,
                'vocab_size': size,
            })
            experiment.log(mean_results)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        vocabulary_size_experiments(grammar_vocab_size_and_dist,
                                main_args,
                                serialization_dir,
                                param_path,
                                num_samples,
                                num_runs)
else:
    vocabulary_size_experiments(grammar_vocab_size_and_dist,
                                main_args,
                                serialization_dir,
                                param_path,
                                args.num_samples,
                                args.num_runs)
