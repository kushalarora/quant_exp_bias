#!/bin/sh
set -eux
python quant_exp_bias/run.py quantify-exposure-bias ${1}/"training/" --include-package quant_exp_bias --cuda-device 0 --output-dir ${1}/tmp/exp_bias/ --num-trials 10 "${@:2}" 
