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


IWSLT2014 de-en

MLE w/o Dropout

env DIR=results/iwslt_mle/"$(date '+%Y_%m_%d_%H_%M')"; sbatch -o logs/iwslt_mle-%J.log -e logs/iwslt_mle-%J.log -J iwslt_mle launcher_basic.sh python -u quant_exp_bias/run.py train training_configs/iwslt14_de_en.json -s ${DIR} --include-package quant_exp_bias

MLE w Dropout 0.3
export DIR=results/iwslt_mle_w_dropout_0.3/"$(date '+%Y_%m_%d_%H_%M')"; sbatch -J iwslt_mle_w_dropout_0.3 -o logs/iwslt_mle_w_dropout_0.3-%J.log -e logs/iwslt_mle_w_dropout_0.3-%J.log launcher_basic.sh python -u quant_exp_bias/run.py train training_configs/iwslt14_de_en_dropout_0.3.json -s ${DIR} --include-package quant_exp_bias


SS Experiments:

0.1 
env DIR=results/iwslt_ss_0.1/"$(date '+%Y_%m_%d_%H_%M')"; sbatch -o logs/iwslt_ss_0.1-%J.log -e logs/iwslt_ss_0.1-%J.log -J iwslt_ss_0.1 launcher_basic.sh python -u quant_exp_bias/run.py train training_configs/iwslt14_de_en_ss_0.1.json -s ${DIR} --include-package quant_exp_bias

env DIR=results/iwslt_ss_0.2/"$(date '+%Y_%m_%d_%H_%M')"; sbatch -o logs/iwslt_ss_0.2-%J.log -e logs/iwslt_ss_0.2-%J.log -J iwslt_ss_0.2 launcher_basic.sh python -u quant_exp_bias/run.py train training_configs/iwslt14_de_en_ss_0.2.json -s ${DIR} --include-package quant_exp_bias

env DIR=results/iwslt_ss_0.4/"$(date '+%Y_%m_%d_%H_%M')"; sbatch -o logs/iwslt_ss_0.4-%J.log -e logs/iwslt_ss_0.4-%J.log -J iwslt_ss_0.4 launcher_basic.sh python -u quant_exp_bias/run.py train training_configs/iwslt14_de_en_ss_0.4.json -s ${DIR} --include-package quant_exp_bias

# License:
See the [LICENSE](LICENSE) file in the repo for more details.
