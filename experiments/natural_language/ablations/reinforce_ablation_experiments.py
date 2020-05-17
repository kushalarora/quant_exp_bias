# coding: utf-8
import itertools
import os

from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run, get_rollout_cost_function_configs, \
                             get_mean_std_results, get_result_iterator

args = get_experiment_args("natural_language", "reinforce_ablation_experiments")

main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('natural_lang/reinforce_ablation_experiments',
                                        output_dir=args.output_dir,
                                        param_path='training_configs/natural_lang/emnlp_news_gpt2_rl.jsonnet',
                                        debug=args.debug,
                                        offline=args.offline,
                                        experiment_text=args.exp_msg,
                                       )

num_samples_and_runs = [(10000, 4)]


samples2pretrained_model = {
    10000: '/home/karora/scratch/quant_exp_bias/natural_lang/dataset_experiments/05_13_2020_01_36_02/10000/0/',
}

def reinforce_ablation_experiments(main_args,
                          orig_serialization_dir,
                          param_path,
                          num_samples,
                          num_runs,
                          rollout_cost_funcs,
                          mixing_coeffs,
                        ):

    pretrained_model = samples2pretrained_model[num_samples]
    os.environ['VOCAB_PATH'] = os.path.join(pretrained_model, 'training/vocabulary')
    os.environ['WEIGHT_FILE_PATH'] = os.path.join(pretrained_model, 'training/best.th')

    # Setup variables needed later.
    step = 0
    for cost_func, mixing_coeff in itertools.product(rollout_cost_funcs, mixing_coeffs):
        serialization_dir = os.path.join(orig_serialization_dir, f'{cost_func}_{mixing_coeff}')
        overrides_func=get_rollout_cost_function_configs("natural_language", cost_func, 
                                                            mixing_coeff, args.temperature)
        
        for num_run in range(num_runs):
            run_metrics = one_exp_run(serialization_dir=serialization_dir,
                                        num_samples=num_samples,
                                        run=num_run,
                                        param_path=param_path,
                                        overides_func=overrides_func, 
                                        sample_from_file=True,
                                        dataset_filename='data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered',
                                        oracle_train_filename=os.path.join(pretrained_model, 'data/oracle_samples-train.txt'),
                                        oracle_dev_filename=os.path.join(pretrained_model, 'data/oracle_samples-dev.txt'),
                                     )

            assert len(run_metrics) == 1, \
                'For this experiment, there should only be one final metric object for a run.'
            run_metrics = run_metrics[0]

            for result in get_result_iterator(run_metrics):
                experiment.log_metrics(result, step=step)
                step += 1

            mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
            mean_results.update({
                'cost_func': cost_func,
                'mixing_coeff': mixing_coeff,
            })
            experiment.log_metrics(mean_results, step=step)


if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        reinforce_ablation_experiments(main_args, serialization_dir,
                              param_path, num_samples, num_runs,
                              args.rollout_cost_funcs, args.mixing_coeffs)
else:
    reinforce_ablation_experiments(main_args, serialization_dir,
                          param_path, args.num_samples, args.num_runs,
                          args.rollout_cost_funcs, args.mixing_coeffs)
