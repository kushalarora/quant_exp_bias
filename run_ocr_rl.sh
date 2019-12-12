#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=30000M
#SBATCH --mail-type=ALL
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
###########################

set -eux


export indir="rl/$(date '+%Y_%m_%d_%H_%M')/"

python -u quant_exp_bias/run.py train training_configs/ocr/ocr_rl.jsonnet -s results/ocr/${indir} --include-package quant_exp_bias
