!#/bin/bash
set -eux

for i in 10000,4,03:00:00,true,4 25000,4,06:00:00,true,4 100000,4,16:00:00,true,4 500000,4,2-00:00:00,true,4; do 
    IFS=","; set -- $i; 
    for ((j=1;j<=$2;j++)); do
        sbatch -t $3 --gres=gpu:$5 --export="DISTRIBUTED=$4,NUM_GPUS=$5" -J nl_min_risk_${1}_${j} ./launcher_basic.sh python -u experiments/natural_language/reinforce_experiments.py --num_samples $1 --num_runs 1;
    done
done