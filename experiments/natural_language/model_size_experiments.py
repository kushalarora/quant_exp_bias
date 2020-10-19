# coding: utf-8
import os

from experiments.util import initialize_experiments, get_experiment_args, \
                             one_exp_run, get_mean_std_results, \
                             get_result_iterator, get_model_overrides_func

args = get_experiment_args("natural_language", "model_size_experiments")

main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('natural_lang/model_size_experiments',
                                        output_dir=args.output_dir,
                                        param_path = 'experiments/natural_language/training_configs/emnlp_news_gpt2.jsonnet',
                                        debug=args.debug,
                                        offline=args.offline,
                                        experiment_text=args.exp_msg,
                                       )

msz2configs  = { # model_size_str => (embed_dim, hidden_dim, num_layers)
    'xsmall': (100, 100, 1),
    'small': (300, 300, 1),
    'medium': (300, 800, 1),
    'large': (300, 2400,  1),
    'xlarge': (300, 2400, 2)
    }
experiment.log_parameters(msz2configs)

num_samples_and_runs = [(50000, 4), (500000,2), (2000000,2)]

def model_size_experiments(model_sizes,
                            main_args,
                            serialization_dir,
                            param_path,
                            num_samples,
                            num_runs,
                           ):
    step = 0
    # Setup variables needed later.
    orig_serialization_dir = serialization_dir
    for model_size in model_sizes:
        model_overrides_func = get_model_overrides_func(*msz2configs[model_size])
        serialization_dir = os.path.join(orig_serialization_dir, model_size)

        for num_run in range(num_runs):
            run_metrics, _ = one_exp_run(serialization_dir=serialization_dir, 
                                        num_samples=num_samples,
                                        run=num_run, 
                                        param_path=param_path,
                                        oracle_config=args.oracle_config,
                                        overides_func=model_overrides_func,
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
                result['model_size'] = model_size
                experiment.log_metrics(result, step=step)
                step += 1

            mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
            mean_results['model_size'] = model_size
            experiment.log_metrics(mean_results, step=step)

    experiment.log_asset_folder(serialization_dir, log_file_name=True, recursive=True)


if args.all:
    model_sizes = msz2configs.keys()
    for num_samples, num_runs in num_samples_and_runs:
        model_size_experiments(model_sizes, main_args,
                                serialization_dir, param_path, num_samples, num_runs)
else:
    # You can specify the model sizes you want to run, else it runs everything.
    model_sizes = args.model_sizes or msz2configs.keys()
    model_size_experiments(model_sizes, main_args, serialization_dir, 
                            param_path, args.num_samples, args.num_runs)
