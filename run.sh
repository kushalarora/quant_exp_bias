#!/bin/sh

set -eux

export NUM_SAMPLES=10000
export ARTIFICIAL_GRAMMAR_TRAIN="results/artificial_grammar/oracle_samples_train.txt"
export ARTIFICIAL_GRAMMAR_DEV="results/artificial_grammar/oracle_samples_dev.txt"
export indir="$(date '+%Y_%m_%d_%H_%M')"

python quant_exp_bias/run.py sample-oracle training_configs/artificial_grammar_sample_oracle.jsonnet -s results/artificial_grammar/${indir} --include-package quant_exp_bias
head -n $((${NUM_SAMPLES} * 9 / 10)) results/artificial_grammar/${indir}/oracle_samples.txt > results/artificial_grammar/${indir}/oracle_samples_train.txt
tail -n $((${NUM_SAMPLES} / 10)) results/artificial_grammar/${indir}/oracle_samples.txt > results/artificial_grammar/${indir}/oracle_samples_dev.txt



python quant_exp_bias/run.py train training_configs/artificial_grammar.jsonnet -s result/artificial_grammar/${indir} --include-package quant_exp_bias
