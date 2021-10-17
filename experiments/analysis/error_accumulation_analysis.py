# coding: utf-8

from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run, get_mean_std_results, get_result_iterator

import os
import glob
import json
import numpy as np

args = get_experiment_args("natural_language", "error_accumulation_analysis")


# ## Basic Setup of grammar and global variables like serialization directory and training config file

main_args, serialization_dir, param_path, experiment_id, experiment = initialize_experiments('error_accumulation_analysis',
                                                                                 output_dir=args.output_dir,
                                                                                 param_path='experiments/natural_language/training_configs/emnlp_news_gpt2.jsonnet',
                                                                                 debug=args.debug,
                                                                                 offline=args.offline,
                                                                                 experiment_text=args.exp_msg,
                                                                                )

max_sequence_lengths = [15, 20, 25, 30, 35, 40, 45, 50]
experiment.log_parameters({'serialization_dir': serialization_dir,
                          'main_args': main_args,
                          'experiment_id': experiment_id})

def error_accumulation_analysis(main_args,
                                run_serialization_dir,
                                param_path,
                                oracle_config,
                                experiment=None,
                               ):
    def qeb_max_sequence_overrides_func():
        for max_sequence_length in max_sequence_lengths:
            overrides = json.dumps({'model':{
                                        'decoder': {
                                            'max_decoding_steps': max_sequence_length,
                                         },
                                    }})
            yield ('max_decoding_steps', max_sequence_length, overrides)

    run_metrics_list = one_exp_run(only_quantify=True,
                                    run_serialization_dir=run_serialization_dir,
                                    train_model_serialization_dir=os.path.join(run_serialization_dir, 'training'),
                                    oracle_config=oracle_config,
                                    exp_bias_inference_funcs=qeb_max_sequence_overrides_func,
                                    num_trials=10,
                                    num_length_samples=20,
                                  )

    for run_metrics in run_metrics_list:
        epoch = run_metrics['epoch']
        for result in get_result_iterator(run_metrics):
            if experiment:
                experiment.log(result)

        mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
        if experiment:
            experiment.log(mean_results)

        run_metrics_lists.append(run_metrics)
    return run_metrics_lists, run_serialization_dir

run_serialization_dir = args.exp_dir
error_accumulation_analysis(
    main_args=main_args, 
    run_serialization_dir=run_serialization_dir, 
    param_path=param_path,
    experiment=experiment,
    oracle_config=args.oracle_config,
)
