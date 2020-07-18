
# coding: utf-8
import itertools
import os

import random
# Ablation experiments will be done on 
# a single run and hence we fix the seed
# so that it uses the same dataset split.
random.seed(220488)

import sys

from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run, get_mean_std_results, \
                             get_result_iterator, \
                             get_rollout_cost_function_configs

args = get_experiment_args("natural_language", "searnn_ablation_experiments")

main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('natural_lang/searnn_ablation_experiments', 
                                        output_dir=args.output_dir,
                                        param_path='experiments/natural_language/training_configs/emnlp_news_gpt2_searnn.jsonnet',
                                        debug=args.debug,
                                        offline=args.offline,
                                        experiment_text=args.exp_msg,
                                       )

rollin_rollout_cost_func_configs = [x for x in itertools.product(args.rollins, args.rollouts, args.rollout_cost_funcs, args.mixing_coeffs)]

num_samples_and_runs = [(50000,2)]

def searnn_ablation_experiments(rollin_rollout_configs,
                                main_args,
                                serialization_dir,
                                param_path,
                                num_samples,
                                num_runs,
                               ):
    step = 0
    orig_serialization_dir = serialization_dir
    for rollin_policy, rollout_policy, cost_func, mixing_coeff in rollin_rollout_cost_func_configs:
        os.environ['rollin_mode'] = rollin_policy
        os.environ['rollout_mode'] = rollout_policy

        serialization_dir = os.path.join(orig_serialization_dir, f'{rollin_policy}_{rollout_policy}_{cost_func}_{mixing_coeff}')
        overrides_func=get_rollout_cost_function_configs("natural_language", cost_func, 
                                                            mixing_coeff, args.temperature, 
                                                            args.neighbors)
        
        for num_run in range(num_runs):
            run_metrics = one_exp_run(serialization_dir=serialization_dir, 
                                        num_samples=num_samples,
                                        run=num_run,
                                        param_path=param_path,
                                        overides_func=overrides_func, 
                                        sample_from_file=True,
                                        dataset_filename='data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered',
                                        run_serialization_dir=args.run_serialization_dir,
                                        only_quantify=args.only_quantify,
                                        recover=args.recover,
                                     )

            assert len(run_metrics) == 1, \
                'For this experiment, there should only be one final metric object for a run.'
            run_metrics = run_metrics[0]
            
            for result in get_result_iterator(run_metrics):
                experiment.log_metrics(result, step=step)
                step += 1

            mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
            mean_results.update({
                'rollin_policy': rollin_policy,
                'rollout_policy': rollout_policy,
                'cost_func': cost_func,
                'mixing_coeff': mixing_coeff,
            })
            experiment.log_metrics(mean_results, step=step)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        searnn_ablation_experiments(rollin_rollout_cost_func_configs, main_args,
                           serialization_dir, param_path, num_samples, num_runs)
else:
    searnn_ablation_experiments(rollin_rollout_cost_func_configs, main_args, serialization_dir, param_path, args.num_samples, args.num_runs)
