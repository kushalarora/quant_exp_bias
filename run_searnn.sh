#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=30000M
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
###########################

source activate quant_exp


set -eux

export NUM_SAMPLES=10000
export rollin_mode=${rollin_mode:='teacher_forcing'}
export rollout_mode=${rollout_mode:='reference'}

export CONFIG_FILE=training_configs/searnn/artificial_grammar_searnn.jsonnet
export indir="searnn_${rollin_mode}_${rollout_mode}/$(date '+%Y_%m_%d_%H_%M')/"
#export FSA_GRAMMAR_FILENAME='grammar_templates/zipf_grammar_2_6.txt'
 export FSA_GRAMMAR_FILENAME='grammar_templates/default_grammar.txt'
export ARTIFICIAL_GRAMMAR_TRAIN="results/artificial_grammar/${indir}/oracle_samples_train.txt"
export ARTIFICIAL_GRAMMAR_DEV="results/artificial_grammar/${indir}/oracle_samples_dev.txt"


python quant_exp_bias/run.py sample-oracle ${CONFIG_FILE} -s results/artificial_grammar/${indir} -n ${NUM_SAMPLES} --include-package quant_exp_bias
head -n $((${NUM_SAMPLES} * 9 / 10)) results/artificial_grammar/${indir}/oracle_samples.txt > results/artificial_grammar/${indir}/oracle_samples_train.txt
tail -n $((${NUM_SAMPLES} / 10)) results/artificial_grammar/${indir}/oracle_samples.txt > results/artificial_grammar/${indir}/oracle_samples_dev.txt



python quant_exp_bias/run.py train  ${CONFIG_FILE} -s results/artificial_grammar/${indir}/"train_output" --include-package quant_exp_bias
python quant_exp_bias/run.py quantify-exposure-bias results/artificial_grammar/${indir}/"train_output/model.tar.gz" --include-package quant_exp_bias --cuda-device 0 --output-dir results/artificial_grammar/${indir}/exp_bias/ --num-trials 20 
