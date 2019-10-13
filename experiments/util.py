import os
import sys
from allennlp.common.util import import_submodules
from datetime import datetime

import_submodules("quant_exp_bias")
from quant_exp_bias.utils import get_args

def run_on_cluster(job_name, account=None):
    def func_wrapper_outer(func):
        def func_wrapper(*args, **kwargs):
            import dask

            from dask_jobqueue import SLURMCluster
            from dask.distributed import Client
            from dask.distributed import progress

            cluster = SLURMCluster(
                            job_name=job_name,
                            project=account,    
                            memory="60 GB",
                            env_extra=['source activate quant_exp', 'pkill -f ray'],
                            job_extra=['--gres=gpu:1'],
                            cores=16,
                            walltime="4:00:00",
                            log_directory='logs/')
            cluster.scale(1)
            client = Client(cluster)
            try:
                future = client.submit(func, *args, **kwargs)
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

def initialize_experiments():
    
    FSA_GRAMMAR_STRING = """
                            q0 -> 'S' q1 [0.9900] | 'a' q1 [0.0025] | 'b' q1 [0.0025] | 'c' q1 [0.0025] | 'E' q1 [0.0025]
                            q1 -> 'S' q1 [0.0025] | 'a' q1 [0.3000] | 'b' q1 [0.3000] | 'c' q1 [0.3000] | 'E' q1 [0.0025]
                            q1 -> 'S' q2 [0.0025] | 'a' q2 [0.0300] | 'b' q2 [0.0300] | 'c' q2 [0.0300] | 'E' q2 [0.0025]
                            q2 -> 'S' [0.0025] | 'a' [0.0025] | 'b' [0.0025] | 'c' [0.0025] | 'E' [0.9900]
                        """
        
    os.environ["FSA_GRAMMAR_STRING"] = FSA_GRAMMAR_STRING
    os.environ['ARTIFICIAL_GRAMMAR_TRAIN'] = ""
    os.environ['ARTIFICIAL_GRAMMAR_DEV'] = ""


    num_sample_oracles = 10
    num_trials = 10
    num_samples_per_length=2000

    # Ipython by default adds some arguments to sys.argv.
    #  We don't want those arguments, hence we pass [] here.
    #
    # The deafult argument get_args is args=None. 
    # This translates to parsing sys.argv. This is useful
    # in case we run the method from a python file but not here.
    # Hence, we keep the default argument as None but pass [] for 
    # ipython notebook.
    main_args = get_args(args=[])

    serialization_dir = os.path.join(main_args.output_dir, datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    param_path = main_args.config
    return main_args, serialization_dir, param_path

