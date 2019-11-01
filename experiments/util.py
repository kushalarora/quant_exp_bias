import json
import os
import sys
import wandb

from datetime import datetime
from typing import Dict, List, Callable, Tuple, Union

from allennlp.common import Params
from allennlp.common.util import import_submodules
import_submodules("quant_exp_bias")
from quant_exp_bias.utils import (get_args, quantify_exposure_bias_runner, 
                  sample_oracle_runner, train_runner)

from quant_exp_bias.oracles.artificial_grammar_oracle import ArtificialLanguageOracle
from quant_exp_bias.utils import get_args


OverrideFuncType = Callable[[], Dict[str, Union[float, str, int]]]
ExpBiasEpochsFuncType = Callable[[str], List[Tuple[int, str, str]]]

def run_on_cluster(job_name, job_id, conda_env,
                   nodes=1, gpu=0, account=None, 
                   local=False, memory="40 GB", 
                   cores=8, log_dir='logs/',
                   walltime="20:00:00"):
    def func_wrapper_outer(func):
        def func_wrapper(*args, **kwargs):
            import dask
            import wandb

            from dask_jobqueue import SLURMCluster
            from dask.distributed import Client
            from dask.distributed import progress

            def func2(*args, **kwargs):
                wandb.init(project='quantifying_exposure_bias', 
                            name=job_name,
                            id=f'{job_name}-{job_id}', 
                            dir=log_dir,
                            sync_tensorboard=False)
                return func(*args, **kwargs)

            if local:
                return func2(*args, **kwargs)

            env_extra =  [f'source activate {conda_env}'] if conda_env else []
            job_extra = [f"--gres=gpu:{gpu}"] if gpu > 0 else []

            cluster = SLURMCluster(
                            job_name=job_name,
                            project=account,    
                            memory=memory,
                            env_extra=env_extra,
                            job_extra=job_extra,
                            cores=cores,
                            walltime=walltime,
                            log_directory=log_dir)
            cluster.scale(nodes)
            client = Client(cluster)
            try:
                future = client.submit(func2, *args, **kwargs)
                progress(future)
                results =  client.gather(future)
                return results
            except Exception as e:
                print(e)
                os.system('pkill -f ray')
                os.system('ps -ef | grep ray')
                raise e
            client.close()
            cluster.close()
        return func_wrapper
    return func_wrapper_outer

def initialize_experiments(experiment_name: str, 
                           grammar_template: str='grammar_templates/grammar_1.template',
                           vocabulary_size: int=6,
                           vocabulary_distribution: str='uniform'):
    # Ipython by default adds some arguments to sys.argv.
    #  We don't want those arguments, hence we pass [] here.
    #
    # The deafult argument get_args is args=None. 
    # This translates to parsing sys.argv. This is useful
    # in case we run the method from a python file but not here.
    # Hence, we keep the default argument as None but pass [] for 
    # ipython notebook.
    main_args = get_args(args=[])

    experiment_id = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    serialization_dir = os.path.join(main_args.output_dir, experiment_name, experiment_id)
    param_path = main_args.config

    grammar_string = ArtificialLanguageOracle.generate_grammar_string(grammar_template_file=grammar_template,
                                                                            vocabulary_size=vocabulary_size, 
                                                                            vocabulary_distribution=vocabulary_distribution)
    os.makedirs(serialization_dir, exist_ok=True)
    grammar_filename = os.path.join(serialization_dir, 'grammar.txt')
    with open(grammar_filename, 'w') as f:
        f.write(grammar_string)
    os.environ["FSA_GRAMMAR_FILENAME"]  = grammar_filename
    os.environ['ARTIFICIAL_GRAMMAR_TRAIN'] = ""
    os.environ['ARTIFICIAL_GRAMMAR_DEV'] = ""

    return main_args, serialization_dir, param_path, experiment_id

def default_overides_func():
    return '{}'

def default_exp_bias_epochs_func(train_model_serialization_dir):
    epoch = -1; qeb_suffix = ''; metrics_filename='metrics.json'
    return [(epoch, qeb_suffix, metrics_filename)]

def one_exp_run(serialization_dir:str, 
                num_samples:int, 
                run:int, 
                param_path:str,
                overides_func:OverrideFuncType = default_overides_func,
                exp_bias_epochs_func:ExpBiasEpochsFuncType = default_exp_bias_epochs_func):
    run_serialization_dir = os.path.join(serialization_dir, str(num_samples), str(run))
    overrides = overides_func()

    sample_oracle_args = get_args(args=['sample-oracle', 
                                            param_path, 
                                            '-s', run_serialization_dir, 
                                            '-n', str(num_samples),
                                            '-o',  overrides])
    oracle_train_filename, oracle_dev_filename = sample_oracle_runner(sample_oracle_args, 
                                                                        run_serialization_dir)

    os.environ['ARTIFICIAL_GRAMMAR_TRAIN'] = oracle_train_filename
    os.environ['ARTIFICIAL_GRAMMAR_DEV'] = oracle_dev_filename

    train_args = get_args(args=['train' , 
                                    param_path, 
                                    '-s', run_serialization_dir, 
                                    '-o',  overrides])
    trainer_params = Params.from_file(train_args.param_path, train_args.overrides)
    cuda_device = trainer_params['trainer']['cuda_device']
    train_model_serialization_dir = train_runner(train_args, 
                                                run_serialization_dir);

    archive_file = os.path.join(train_model_serialization_dir, 'model.tar.gz')
    metric_list = []
    for epoch, qeb_suffix, metric_filename in exp_bias_epochs_func(train_model_serialization_dir):
        qeb_output_dir = os.path.join(run_serialization_dir, 'exp_bias', qeb_suffix)
        metrics = json.load(open(os.path.join(train_model_serialization_dir, metric_filename)))

        # This is only needed when doing validation experiments.
        weights_file = None
        if epoch != -1:
            weights_file = os.path.join(train_model_serialization_dir, f'model_state_epoch_{epoch}.th')

        qeb_args = get_args(args=['quantify-exposure-bias', 
                                    archive_file, 
                                    '--output-dir', qeb_output_dir, 
                                    '--weights-file', weights_file,
                                    '-o',  overrides])
        exp_biases, exp_bias_mean, exp_bias_std = quantify_exposure_bias_runner(qeb_args, 
                                                                                archive_file,
                                                                                qeb_output_dir,
                                                                                cuda_device=cuda_device,
                                                                                weights_file=weights_file);
        
        metrics['exp_biases'] = exp_biases
        metrics['exp_bias_mean'] = exp_bias_mean
        metrics['exp_bias_std'] = exp_bias_std
        metric_list.append(metrics)
    return metric_list
