!#/bin/bash
set -eux

for i in 10000,4,00:30:00 25000,4,1:00:00 100000,4,2:00:00, 500000,4,12:00:00 2000000,2,1-00:00:00; do 
    IFS=","; set -- $i; 
    for ((i=1;i<=$2;i++)); do
        sbatch -t $3 -J nl_dse_${1}_${i} ./launcher_basic.sh python -u experiments/natural_language/dataset_experiments.py --num_samples $1 --num_runs 1;
    done
done