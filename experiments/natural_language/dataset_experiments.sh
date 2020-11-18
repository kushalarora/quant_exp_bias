!#/bin/bash
set -eux

for i in 10000,4,00:45:00,false,1 25000,4,1:00:00,false,1 100000,4,2:30:00,false,1 500000,4,12:00:00,false,1 2000000,4,18:00:00,true,4; do 
    IFS=","; set -- $i; 
    for ((j=1;j<=$2;j++)); do
        sbatch -t $3 --gres=gpu:$5 --export="DISTRIBUTED=$4,NUM_GPUS=$5" -J nl_dse_${1}_${j} ./launcher_basic.sh python -u experiments/natural_language/dataset_experiments.py --num_samples $1 --num_runs 1;
    done
done