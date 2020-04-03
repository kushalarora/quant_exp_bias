#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=1
#SBATCH --mem=60000M
#SBATCH --mail-type=ALL
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.out
###########################

set -x
source ~/.bashrc
conda deactivate
module load httpproxy
conda activate quant_exp
$@
