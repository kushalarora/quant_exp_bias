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


allennlp train test/fixtures/lm/experiment_unsampled.jsonnet -s /scratch/quant_exp_bias/results/lm_"$(date '+%Y_%m_%d_%H_%M')" --include-package models --include-package dataset_readers

allennlp train test/fixtures/seq2seq/experiments.json -s /scratch/quant_exp_bias/results/seq2seq_"$(date '+%Y_%m_%d_%H_%M')" --include-package models --include-package dataset_readers

Run PTB experiments:
allennlp train training_configs/ptb_lm.jsonnet -s /scratch/quant_exp_bias/results/ptb_"$(date '+%Y_%m_%d_%H_%M_%S')" --include-package models --include-package dataset_readers

# License:
See the [LICENSE](LICENSE) file in the repo for more details.
