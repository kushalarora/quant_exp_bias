#!/bin/sh

set -eux

export NUM_SAMPLES=30000
export indir="$(date '+%Y_%m_%d_%H_%M')/"

python quant_exp_bias/run.py sample-oracle training_configs/artificial_grammar_sample_oracle.jsonnet -s results/artificial_grammar/${indir} --include-package quant_exp_bias
head -n $((${NUM_SAMPLES} * 9 / 10)) results/artificial_grammar/${indir}/oracle_samples.txt > results/artificial_grammar/${indir}/oracle_samples_train.txt
tail -n $((${NUM_SAMPLES} / 10)) results/artificial_grammar/${indir}/oracle_samples.txt > results/artificial_grammar/${indir}/oracle_samples_dev.txt

export ARTIFICIAL_GRAMMAR_TRAIN="results/artificial_grammar/${indir}/oracle_samples_train.txt"
export ARTIFICIAL_GRAMMAR_DEV="results/artificial_grammar/${indir}/oracle_samples_dev.txt"


python quant_exp_bias/run.py train training_configs/artificial_grammar.jsonnet -s results/artificial_grammar/${indir}/"train_output" --include-package quant_exp_bias
python quant_exp_bias/run.py quantify-exposure-bias results/artificial_grammar/${indir}/"train_output/model.tar.gz" --include-package quant_exp_bias --cuda-device 0 --output-dir results/artificial_grammar/2019_10_04_20_08/exp_bias/ --num-trials 20 
