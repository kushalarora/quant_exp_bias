!#/bin/bash
set -eux

for i in 10000,4,14:00:00,true,4 25000,4,24:00:00,true,4 100000,4,3-00:00:00,true,4 500000,2,3-10:00:00,true,4; do
    IFS=","; set -- $i;
     for ((i=1;i<=$2;i++)); do
        sbatch -t $3 --gres=gpu:$5 --export="DISTRIBUTED=$4,NUM_GPUS=$5" -J nl_searnn_${1}_${i} ./launcher_basic.sh python -u experiments/natural_language/searnn_experiments.py --num_samples $1 --num_runs 1 --rollins teacher_forcing --rollouts reference;
    done
done