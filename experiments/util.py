import json
import os
import sys
import wandb

from datetime import datetime
from typing import Dict, List

from allennlp.common import Params
from allennlp.common.util import import_submodules
from quant_exp_bias.utils import (get_args, quantify_exposure_bias_runner, 
                  sample_oracle_runner, train_runner)

import_submodules("quant_exp_bias")
from quant_exp_bias.oracles.artificial_grammar_oracle import ArtificialLanguageOracle
from quant_exp_bias.utils import get_args

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