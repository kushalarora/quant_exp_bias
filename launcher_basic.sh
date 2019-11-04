#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem=60000M
#SBATCH --mail-type=ALL
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
###########################


set -x
source activate quant_exp_2 
$@
