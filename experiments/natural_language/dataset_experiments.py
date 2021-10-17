
# coding: utf-8
from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run, get_mean_std_results, get_result_iterator

def dataset_experiments(main_args,
                        serialization_dir,
                        param_path,
                        num_samples,
                        num_runs,
                        oracle_config,
                        experiment=None,
                        only_quantify=False,
                        run_serialization_dir=None,
                        recover=False,
                        donot_quantify=False,
                       ):

    run_metrics_lists = []
    for num_run in range(num_runs):
        run_metrics, run_serialization_dir = one_exp_run(serialization_dir=serialization_dir,
                                    num_samples=num_samples,
                                    run=num_run,
                                    param_path=param_path,
                                    oracle_config=oracle_config,
                                    sample_from_file=True,
                                    dataset_filename='data/wmt_news_2017/news.2017.en.shuffled.deduped.filtered',
                                    run_serialization_dir=run_serialization_dir,
                                    only_quantify=only_quantify,
                                    recover=recover,
                                    donot_quantify=donot_quantify,
                                 )
        if len(run_metrics) == 0:
            continue

        run_metrics = run_metrics[0]

        for result in get_result_iterator(run_metrics):
            if experiment:
                experiment.log(result)

        mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
        if experiment:
            experiment.log(mean_results)

        run_metrics_lists.append(run_metrics)
    return run_metrics_lists, run_serialization_dir

if __name__ == '__main__':
    args = get_experiment_args("natural_language", "dataset_experiments")

    main_args, serialization_dir, param_path, experiment_id, \
            experiment = initialize_experiments('natural_lang/dataset_experiments',
                                                output_dir=args.output_dir,
                                                param_path='experiments/natural_language/training_configs/emnlp_news_gpt2.jsonnet',
                                                debug=args.debug,
                                                offline=args.offline,
                                                experiment_text=args.exp_msg,
                                            )

    dataset_experiments_params = [(10000, 8), (50000, 6) , (500000, 4), (2000000, 2), (5000000, 1)]

    if args.all:
        for num_samples, num_runs in dataset_experiments_params:
            dataset_experiments(
                main_args=main_args, 
                serialization_dir=serialization_dir, 
                param_path=param_path, 
                num_samples=num_samples, 
                num_runs=num_runs, 
                experiment=experiment,
                oracle_config=args.oracle_config,
                only_quantify=args.only_quantify,
                run_serialization_dir=args.run_serialization_dir,
                recover=args.recover,
                )
    else:
        dataset_experiments(
                main_args=main_args, 
                serialization_dir=serialization_dir, 
                param_path=param_path, 
                num_samples=args.num_samples, 
                num_runs=args.num_runs, 
                experiment=experiment,
                oracle_config=args.oracle_config,
                only_quantify=args.only_quantify,
                run_serialization_dir=args.run_serialization_dir,
                recover=args.recover,
               )
