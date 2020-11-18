#!/bin/sh
# Scheduled Sampling Ablation Experiments


set -eux
for ss in "u_0.05" "u_0.10" "u_0.25" "linear"; do
  for arg in 100000,4,4:00:00; do 
      IFS=","; set -- $arg; 
      echo ${@}
      for ((i=1;i<=$2;i++)); do
         sbatch -t $3 -J ss_ablation_$1_${ss} ./launcher_basic.sh python -u experiments/natural_language/scheduled_sampling_experiments.py --num_samples $1 --num_runs 1 --ss_configs ${ss};
      done
  done
done