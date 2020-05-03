# coding: utf-8
import json
import itertools
import os

from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run, get_mean_std_results, \
                             get_result_iterator

args = get_experiment_args("natural_language", "searnn_experiments")

main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('natural_lang/searnn_experiments',
                                        output_dir=args.output_dir,
                                        param_path = 'training_configs/natural_lang/emnlp_news_gpt2_searnn.jsonnet',
                                        debug=args.debug,
                                        offline=args.offline,
                                        experiment_text=args.exp_msg,
                                    )

rollin_rollout_configs = [x for x in itertools.product(args.rollins, args.rollouts)]

num_samples_and_runs = [(50000, 4), (500000, 2), (2000000, 2)]

def searnn_experiments(rollin_rollout_configs,
                            num_samples_and_runs,
                            main_args,
                            serialization_dir,
                            param_path,
                            num_samples,
                            num_runs,
                      ):
    step = 0
    # Setup variables needed later.
    orig_serialization_dir = serialization_dir
    for rollin_policy, rollout_policy in rollin_rollout_configs:
        os.environ['rollin_mode'] = rollin_policy
        os.environ['rollout_mode'] = rollout_policy

        serialization_dir = os.path.join(orig_serialization_dir, f'{rollin_policy}_{rollout_policy}')
        for num_run in range(num_runs):
            run_metrics = one_exp_run(serialization_dir=serialization_dir,
                                        num_samples=num_samples,
                                        run=num_run,
                                        param_path=param_path,
                                        sample_from_file=True,
                                        dataset_filename='data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered',
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
            })
            experiment.log_metrics(mean_results, step=step)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        searnn_experiments(rollin_rollout_configs, num_samples_and_runs, main_args, serialization_dir, param_path, num_samples, num_runs)
else:
    searnn_experiments(rollin_rollout_configs, num_samples_and_runs, main_args, serialization_dir, param_path, args.num_samples, args.num_runs)
