
import json

import numpy
import torch
from allennlp.common.testing import AllenNlpTestCase
from experiments.util import get_experiment_args

class UtilTest(AllenNlpTestCase):
    def test_generate_grammar_file(self):
      pass

    def test_initialize_experiments(self):
      pass

    def test_last_exp_bias_epoch_func(self):
      pass

    def test_generate_dataset_from_dataset_file(self):
      pass

    def test_generate_dataset_w_o_generating_grammar(self):
      pass
    
    def test_generate_dataset_w_generating_grammar(self):
      pass

    def test_get_experiments_args(self):
      # Verify default oracle config is correct:
      args = get_experiment_args(experiment_type='artificial_language', 
                                 experiment_name='dataset_experiments')
      
      assert args.oracle_config == 'experiments/artificial_language/training_configs/artificial_grammar_oracle.jsonnet'

      # Verify default oracle config is correct:
      args = get_experiment_args(experiment_type='natural_language', 
                                 experiment_name='dataset_experiments')
      
      assert args.oracle_config == 'experiments/natural_language/training_configs/gpt2_oracle.jsonnet'

      # Verify model size experiments have model sizes arguments:
      args = get_experiment_args(experiment_type='natural_language', 
                            experiment_name='model_size_experiments')
      assert hasattr(args, 'model_sizes')

      # Verify artificial language experiment has vocab and grammar arguments.
      args = get_experiment_args(experiment_type='artificial_language', 
                            experiment_name='dataset_experiments')
      assert hasattr(args, 'vocab_distributions')
      assert hasattr(args, 'grammar_templates')

      # TODO (Kushal:) Finish writing this test and other tests.
      
    def test_calculate_ss_k(self):
      pass 

    def test_get_grammar_iterator(self):
      pass

    def test_get_result_iterator(self):
      pass 

    def test_get_rollout_cost_function_configs(self):
      pass