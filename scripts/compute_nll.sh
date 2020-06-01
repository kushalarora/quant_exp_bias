 python quant_exp_bias/run.py compute-nll ${1}/training/ --output-dir ${1}/nll --include-package quant_exp_bias --cuda-device 0 --num-trials 5 "${@:2}" 
