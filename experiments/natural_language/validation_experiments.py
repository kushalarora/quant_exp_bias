# coding: utf-8
import glob
import json
import numpy as np
import os
import re

from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run, get_grammar_iterator, \
                             get_mean_std_results, get_result_iterator

args = get_experiment_args("natural_language", "validation_experiments")


main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('natural_lang/validation_experiments',
                                        output_dir=args.output_dir,
                                        param_path = 'experiments/natural_language/training_configs/emnlp_news_gpt2.jsonnet',
                                        debug=args.debug,
                                        experiment_text=args.exp_msg,
                                    )

num_samples_and_runs = [(50000, 6), (500000, 4), (2000000, 2)]

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
    overrides = json.dumps({'trainer': {'num_epochs': 20, 'patience': None}})

    for num_run in range(num_runs):
        run_metrics_list,_ = one_exp_run(serialization_dir=serialization_dir, 
                                        num_samples=num_samples,
                                        run=num_run, 
                                        param_path=param_path,
                                        oracle_config=args.oracle_config,
                                        overides_func=lambda:overrides,
                                        exp_bias_epochs_func=validation_exp_bias_epochs_func,
                                        sample_from_file=True,
                                        dataset_filename='data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered',
                                        num_trials=5,
                                        num_length_samples=5,
                                        num_samples_per_length=32,
                                        run_serialization_dir=args.run_serialization_dir,
                                        only_quantify=args.only_quantify,
                                        recover=args.recover,
                                      )
        for run_metrics in run_metrics_list:
            epoch = run_metrics['epoch']

            for result in get_result_iterator(run_metrics):
                experiment.log(result)

            mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
            mean_results['epoch'] = epoch
            experiment.log(mean_results)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        validation_experiments(main_args, serialization_dir, param_path, num_samples, num_runs)
else:
    validation_experiments(main_args, serialization_dir, param_path, args.num_samples, args.num_runs)
