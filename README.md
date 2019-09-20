# Quantifying Exposure Bias
Experiments to quantify exposure bias.

# Data:
Data we will use for training. 

# Results:
Results of our experiments:

# Commands to run:
```bash
python main.py
```


python quant_exp_bias/run.py train test/fixtures/lm/experiment_unsampled.jsonnet -s /scratch/quant_exp_bias/results/lm_"$(date '+%Y_%m_%d_%H_%M')" --include-package quant_exp_bias 

python quant_exp_bias/run.py train test/fixtures/seq2seq/experiments.json -s /scratch/quant_exp_bias/results/seq2seq_"$(date '+%Y_%m_%d_%H_%M')" --include-package quant_exp_bias 

Run PTB experiments:
python quant_exp_bias/run.py train training_configs/ptb_lm.jsonnet -s /scratch/quant_exp_bias/results/ptb_"$(date '+%Y_%m_%d_%H_%M_%S')" --include-package quant_exp_bias 

# License:
See the [LICENSE](LICENSE) file in the repo for more details.
