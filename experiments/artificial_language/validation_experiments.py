# coding: utf-8

import glob
import json
import os
import re

from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run, get_grammar_iterator, \
                             get_mean_std_results, get_result_iterator


args = get_experiment_args("artificial_language", "validation_experiments")

main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('artificial_lang/validation_experiments', 
                                        param_path='experiments/artificial_language/training_configs/artificial_grammar_composed.jsonnet',
                                        output_dir=args.output_dir,
                                        debug=args.debug,
                                        offline=args.offline,
                                        experiment_text=args.exp_msg,
                                       )

num_samples_and_runs = [(1000, 8), (10000, 4), (100000, 2)]

def validation_exp_bias_epochs_func(train_model_serialization_dir):
    epoch_files = glob.glob(os.path.join(train_model_serialization_dir + '/model_state_epoch_*.th'))
    epochs =[int(re.search('epoch_([0-9]+).th', fname).group(1)) for fname in epoch_files]

    for epoch in epochs:
        qeb_suffix = f'epoch_{epoch}'
        metrics_filename = f'metrics_epoch_{epoch}.json'
        yield (epoch, qeb_suffix, metrics_filename)
        
def validation_experiments(main_args,
                           serialization_dir,
                           param_path,
                           num_samples,
                           num_runs,
                           ):
    step = 0
    overrides = json.dumps({'trainer': {'num_epochs': 50, 'patience': None}})

    for grammars_and_vocabularies in get_grammar_iterator(experiment,
                                                          args.grammar_templates, 
                                                          args.vocab_distributions,
                                                          num_runs):
        num_run, grammar_template_file, vocab_dist, \
            shall_generate_grammar_file, grammar_params = grammars_and_vocabularies
        run_metrics_list = one_exp_run(serialization_dir=serialization_dir,
                                        num_samples=num_samples,
                                        run=num_run,
                                        param_path=param_path,
                                        overides_func=lambda: overrides,
                                        exp_bias_epochs_func=validation_exp_bias_epochs_func,
                                        num_trials=5,
                                        num_length_samples=5,
                                        num_samples_per_length=100,
                                        grammar_template=grammar_template_file,
                                        shall_generate_grammar_file=shall_generate_grammar_file,
                                        vocabulary_distribution=vocab_dist,
                                      )
        for run_metrics in run_metrics_list:
            epoch=run_metrics['epoch']

            for result in get_result_iterator(run_metrics):
                experiment.log_metrics(result, step=step)
                step += 1

            mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
            mean_results.update(grammar_params)
            mean_results['epoch'] = epoch
            experiment.log_metrics(mean_results, step=step)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        validation_experiments(main_args, serialization_dir,
                               param_path, num_samples, num_runs)
else:
    validation_experiments(main_args, serialization_dir,
                           param_path, args.num_samples, args.num_runs)
