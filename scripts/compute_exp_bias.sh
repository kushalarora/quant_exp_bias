#!/bin/sh
set -eux
python quant_exp_bias/run.py quantify-exposure-bias ${1}/"training/" --oracle-config experiments/natural_language/training_configs/gpt2_oracle.jsonnet --include-package quant_exp_bias --include-package lmpl --cuda-device 0 --output-dir ${1}/tmp/exp_bias/ --num-trials 10 "${@:2}" 
