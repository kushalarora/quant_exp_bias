
python -u quant_exp_bias/run.py train training_configs/natural_lang/ptb_gpt2.jsonnet -s /scratch/quant_exp_bias/results/ptb_"$(date '+%Y_%m_%d_%H_%M_%S')" --include-package quant_exp_bias
python quant_exp_bias/run.py  quantify-exposure-bias /scratch/quant_exp_bias/results/ptb_2019_11_21_23_21_10/model.tar.gz   --include-package quant_exp_bias --cuda-device 0 --output-dir /scratch/quant_exp_bias/results/exp_bias/  --num-trials 20
