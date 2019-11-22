
#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=30000M
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
###########################

source activate quant_exp

export indir="ptb_gpt2_$(date '+%Y_%m_%d_%H_%M')/"


set -eux

python -u quant_exp_bias/run.py train training_configs/natural_lang/ptb_gpt2.jsonnet -s /scratch/quant_exp_bias/results/${indir} --include-package quant_exp_bias
python quant_exp_bias/run.py  quantify-exposure-bias ${indir}/model.tar.gz   --include-package quant_exp_bias --cuda-device 0 --output-dir /scratch/quant_exp_bias/results/exp_bias/  --num-trials 20
