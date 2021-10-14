#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=1
#SBATCH --mem=60000M
#SBATCH --mail-type=ALL,TIME_LIMIT,BEGIN,END,FAIL
#SBATCH --mail-user=arorakus@mila.quebec
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.out
###########################

set -x
export NUM_GPUS=${NUM_GPUS:=2}
export DISTRIBUTED=${DISTRIBUTED:="false"}
module load cuda/10.1
module load httpproxy
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=.:${PYTHONPATH}
source ~/envs/qeb/bin/activate
$@
