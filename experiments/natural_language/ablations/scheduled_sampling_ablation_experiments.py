
# coding: utf-8

import os
import sys

import random
# Ablation experiments will be done on 
# a single run and hence we fix the seed
# so that it uses the same dataset split.
random.seed(220488)

from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run, get_mean_std_results, \
                             get_result_iterator, calculate_ss_k,\
                             get_scheduled_sampling_overrides_func, \
                             get_scheduled_sampling_configs

args = get_experiment_args("natural_language", "scheduled_sampling_ablation_experiments")

main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('natural_lang/scheduled_sampling_ablation_experiment',
                                        output_dir=args.output_dir,
                                        param_path = 'training_configs/natural_lang/emnlp_news_gpt2.jsonnet',
                                        debug=args.debug,
                                        offline=args.offline,
                                        experiment_text=args.exp_msg,
                                    )


scheduled_sampling_dict = get_scheduled_sampling_configs(args.num_samples,
                                                         args.batch_size, 
                                                         args.num_epochs)

num_samples_and_runs = [(100000,4)]

experiment.log_parameters({'serialization_dir': serialization_dir,
                          'main_args': main_args,
                          'param_path': param_path,
                          'experiment_id': experiment_id})

def scheduled_sampling_ablation_experiments(scheduled_sampling_dict, 
                                    main_args, 
                                    serialization_dir, 
                                    param_path,
                                    num_samples,
                                    num_runs,
                                    ss_configs,
                                  ):
    step = 0
    orig_serialization_dir = serialization_dir
    for ss_config in ss_configs:
        ss_type, ss_ratio, ss_k = scheduled_sampling_dict[ss_config]
        serialization_dir = os.path.join(orig_serialization_dir, f'{ss_type}_{ss_ratio}_{ss_k}')
        overrides_func = get_scheduled_sampling_overrides_func(ss_type, ss_ratio, ss_k)

        for num_run in range(num_runs):
            run_metrics = one_exp_run(serialization_dir=serialization_dir, 
                                        num_samples=num_samples,
                                        run=num_run,
                                        param_path=param_path, 
                                        overides_func=overrides_func,
                                        sample_from_file=True,
                                        dataset_filename='data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered',
                                        num_trials=5,
                                        num_length_samples=5,
                                      )

            assert len(run_metrics) == 1, \
                'For this experiment, there should only be one final metric object for a run.'
            run_metrics = run_metrics[0]

            for result in get_result_iterator(run_metrics):
                experiment.log_metrics(result, step=step)
                step += 1

            mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
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
        scheduled_sampling_ablation_experiments(scheduled_sampling_dict,
                                                main_args,
                                                serialization_dir,
                                                param_path,
                                                num_samples,
                                                num_runs,
                                                args.ss_configs)
else:
    scheduled_sampling_ablation_experiments(scheduled_sampling_dict,
                                            main_args,
                                            serialization_dir,
                                            param_path,
                                            args.num_samples,
                                            args.num_runs, 
                                            args.ss_configs)