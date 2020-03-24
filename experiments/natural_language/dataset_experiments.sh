!#/bin/bash
set -eux

for i in 10000,8,12:00:00 25000,6,16:00:00 100000,4,16:00:00, 500000,2,2-00:00:00 2000000,1,3-00:00:00; do 
    IFS=","; set -- $i; 
    sbatch -t $3 -J nl_dse_$1 ./launcher_basic.sh python -u experiments/natural_language/dataset_experiments.py --num_samples $1 --num_runs $2
done
