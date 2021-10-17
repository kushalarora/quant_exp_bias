
from experiments.util import initialize_experiments, get_experiment_args

import os
import glob
import json
import numpy as np
import math
args = get_experiment_args("natural_language", "error_accumulation_analysis")


main_args, serialization_dir, param_path, experiment_id, experiment = initialize_experiments('error_accumulation_analysis',
                                                                                 output_dir=args.output_dir,
                                                                                 param_path='experiments/natural_language/training_configs/emnlp_news_gpt2.jsonnet',
                                                                                 debug=args.debug,
                                                                                 offline=args.offline,
                                                                                 experiment_text=args.exp_msg,
                                                                                )
hist = {}
run_serialization_dir = args.exp_dir
max_sequence_lengths = [10, 20, 30, 40, 50]
experiment.log_parameters({'serialization_dir': run_serialization_dir,
                          'main_args': main_args,
                          'experiment_id': experiment_id})
for generated_filename in glob.glob(run_serialization_dir + '/exp_bias/*/model_sampled_generated.txt'):
  with open(generated_filename) as generated_file:
    for line in generated_file:
      splits = line.strip().split()
      tokens = splits[:-3]
      num_tokens = len(tokens)
      exp_value = float(splits[-1].split('=')[-1])
      bin = math.floor(num_tokens/5) * 5
      if bin not in hist:
          hist[bin] = [0., 0, 0]
          
      hist[bin][0] += exp_value 
      hist[bin][1] += num_tokens
      hist[bin][2] += 1
import pdb; pdb.set_trace()
