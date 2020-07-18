
# coding: utf-8
import itertools
import os

from experiments.util import initialize_experiments, get_experiment_args, \
                             get_grammar_iterator, one_exp_run, \
                            get_mean_std_results, get_result_iterator

args = get_experiment_args("artificial_language", "reiforce_experiments")

main_args, serialization_dir, param_path, experiment_id, \
    experiment = initialize_experiments('artificial_lang/reinforce_experiments',
                                        output_dir=args.output_dir,
                                        param_path='experiments/artificial_language/training_configs/artificial_grammar_rl.jsonnet',
                                        offline=args.offline,
                                        debug=args.debug,
                                        experiment_text=args.exp_msg,
                                       )

num_samples_and_runs = [(1000, 4), (10000, 2), (100000, 2)]

samples2pretrained_model = {
    1000: 'results/artificial_grammar/artificial_lang/dataset_experiments/04_07_2020_16_45_21/1000/0/',
    10000: 'results/artificial_grammar/artificial_lang/dataset_experiments/03_18_2020_00_04_07/10000/0/',
    50000: 'results/artificial_grammar/artificial_lang/dataset_experiments/03_18_2020_00_04_07/50000/0/',
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

    step = 0
    orig_serialization_dir = serialization_dir

    serialization_dir = os.path.join(orig_serialization_dir)
    for grammars_and_vocabularies in get_grammar_iterator(experiment,
                                                          args.grammar_templates, 
                                                          args.vocab_distributions,
                                                          num_runs):
        num_run, grammar_template_file, vocab_dist, \
            shall_generate_grammar_file, grammar_params = grammars_and_vocabularies
        run_metrics = one_exp_run(serialization_dir=serialization_dir,
                                    num_samples=num_samples,
                                    run=num_run,
                                    param_path=param_path,
                                    grammar_file_epsilon_0=os.path.join(pretrained_model,
                                                                        'epsilon_0_grammar.txt'),
                                    grammar_file_epsilon=os.path.join(pretrained_model,
                                                                      'epsilon_0.0001_grammar.txt'),
                                    grammar_template=grammar_template_file,
                                    shall_generate_grammar_file=shall_generate_grammar_file,
                                    vocabulary_distribution=vocab_dist,
                                  )

        assert len(run_metrics) == 1, \
            'For this experiment, there should only be one final metric object for a run.'

        run_metrics = run_metrics[0]
        for result in get_result_iterator(run_metrics):
            experiment.log_metrics(result, step=step)
            step += 1

        mean_results = get_mean_std_results(num_run, num_samples, run_metrics)
        mean_results.update(grammar_params)
        experiment.log_metrics(mean_results, step=step)

if args.all:
    for num_samples, num_runs in num_samples_and_runs:
        reinforce_experiments(main_args, serialization_dir,
                              param_path, num_samples, num_runs)
else:
    reinforce_experiments(main_args, serialization_dir,
                          param_path, args.num_samples, args.num_runs)