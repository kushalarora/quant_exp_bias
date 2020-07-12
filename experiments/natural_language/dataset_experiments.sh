!#/bin/bash
set -eux

for i in 10000,4,1:00:00 25000,4,3:00:00 100000,4,6:00:00, 500000,2,1-00:00:00 2000000,1,3-00:00:00; do 
    IFS=","; set -- $i; 
    for i in $(seq 1 $2); do
        sbatch -t $3 -J nl_dse_$1 ./launcher_basic.sh python -u experiments/natural_language/dataset_experiments.py --num_samples $1 --num_runs 1
    done
done
