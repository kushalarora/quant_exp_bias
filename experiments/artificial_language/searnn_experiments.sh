!#/bin/bash
set -eux

for i in 1000,8,2-00:00:00 10000,4,2-00:00:00, 100000,2,2-00:00:00; do 
    IFS=","; set -- $i;
    sbatch -t $3 -J al_searnn_$1 ./launcher_basic.sh python -u experiments/artificial_language/searnn_experiments.py --num_samples $1 --num_runs $2
done