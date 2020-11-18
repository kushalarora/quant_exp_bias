# coding: utf-8
import itertools
import os 

from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run, get_mean_std_results, \
                             get_result_iterator
from experiments.natural_language.dataset_experiments import dataset_experiments

args = get_experiment_args("natural_language", "reinforce_experiments")

main_args, serialization_dir, param_path, experiment_id, \
        experiment = initialize_experiments('natural_lang/reinforce_experiments',
                                            output_dir=args.output_dir,
                                            param_path='experiments/natural_language/training_configs/emnlp_news_gpt2_risk.jsonnet',
                                            debug=args.debug,
                                            offline=args.offline,
                                            experiment_text=args.exp_msg,
                                        )

num_samples_and_runs = [(10000, 4), (50000, 2), (2000000, 2)]



def reinforce_experiments(main_args,
                          serialization_dir,
                          param_path,
                          num_samples,
                          num_runs,
                          use_pretrained_model=False,
                        ):

    if use_pretrained_model:
        assert "WARM_START_MODEL" in os.environ, \
            "WARM_START_MODEL env. variable is needed for reinforcement learning experiments."
    else:
        if args.only_quantify:
            pretrained_model = args.run_serialization_dir
        else:
            _, run_serialization_dir = dataset_experiments(
                                        main_args=main_args, 
                                        serialization_dir=os.path.join(serialization_dir, "warm_start_model"),
                                        param_path='experiments/natural_language/training_configs/emnlp_news_gpt2.jsonnet', 
                                        num_samples=num_samples, 
                                        num_runs=1,
                                        oracle_config=args.oracle_config,
                                        experiment=experiment,
                                        donot_quantify=True,
                                    )
            pretrained_model = run_serialization_dir
        
        os.environ['WARM_START_MODEL'] = os.path.join(pretrained_model, 'training/')
    # Setup variables needed later.
    step = 0
    orig_serialization_dir = serialization_dir
    serialization_dir = os.path.join(orig_serialization_dir)
    for num_run in range(num_runs):
        run_metrics,_ = one_exp_run(serialization_dir=serialization_dir,
                                    num_samples=num_samples,
                                    run=num_run,
                                    param_path=param_path,
                                    oracle_config=args.oracle_config,
                                    sample_from_file=True,
                                    dataset_filename='data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered',
                                    oracle_train_filename=os.path.join(pretrained_model, 'data/oracle_samples-train.txt'),
                                    oracle_dev_filename=os.path.join(pretrained_model, 'data/oracle_samples-dev.txt'),
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
