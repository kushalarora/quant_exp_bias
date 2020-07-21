# coding: utf-8
import itertools
import os 

from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run, get_mean_std_results, \
                             get_result_iterator

args = get_experiment_args("natural_language", "reinforce_experiments")

main_args, serialization_dir, param_path, experiment_id, \
        experiment = initialize_experiments('natural_lang/reinforce_experiments',
                                            output_dir=args.output_dir,
                                            param_path='experiments/natural_language/training_configs/emnlp_news_gpt2_rl.jsonnet',
                                            debug=args.debug,
                                            offline=args.offline,
                                            experiment_text=args.exp_msg,
                                        )

num_samples_and_runs = [(10000, 4), (50000, 2), (2000000, 2)]

samples2pretrained_model = {
    10000: 'results/artificial_grammar/natural_lang/dataset_experiments/03_15_2020_00_43_12/10000/0/',
    50000: 'results/artificial_grammar/natural_lang/dataset_experiments/03_14_2020_22_59_49/50000/0/',
}

def reinforce_experiments(main_args,
                          serialization_dir,
                          param_path,
                          num_samples,
                          num_runs,
                          ):

    pretrained_model = samples2pretrained_model[num_samples]
    os.environ['VOCAB_PATH'] = os.path.join(pretrained_model, 'training/vocabulary')
    os.environ['WEIGHT_FILE_PATH'] = os.path.join(pretrained_model, 'training/best.th')

    # Setup variables needed later.
    step = 0
    orig_serialization_dir = serialization_dir

    serialization_dir = os.path.join(orig_serialization_dir)
    for num_run in range(num_runs):
        run_metrics = one_exp_run(serialization_dir=serialization_dir,
                                    num_samples=num_samples,
                                    run=num_run,
                                    param_path=param_path,
                                    oracle_config=args.oracle_config,
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
        experiment.log_metrics(mean_results, step=step)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        reinforce_experiments(main_args, serialization_dir,
                              param_path, num_samples, num_runs)
else:
    reinforce_experiments(main_args, serialization_dir,
                          param_path, args.num_samples, args.num_runs)
