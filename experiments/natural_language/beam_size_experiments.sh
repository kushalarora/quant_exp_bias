!#/bin/bash
set -eux

for i in 50000,8,2-00:00:00 500000,4,2-00:00:00, 2000000,2,3-00:00:00; do 
    IFS=","; set -- $i; 
    sbatch -t $3 -J nl_bse_$1 ./launcher_basic.sh python -u experiments/natural_language/beam_size_experiments.py --num_samples $1 --num_runs $2
done
