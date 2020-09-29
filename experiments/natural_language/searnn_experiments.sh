!#/bin/bash
set -eux

for i in 10000,4,3:00:00 25000,4,10:00:00, 100000,4,2-10:00:00; do
    IFS=","; set -- $i;
     for ((i=1;i<=$2;i++)); do
        sbatch -t $3 -J al_searnn_${1}_${i} ./launcher_basic.sh python -u experiments/natural_language/searnn_experiments.py --num_samples $1 --num_runs 1;
    done
done