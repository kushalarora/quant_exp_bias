!#/bin/bash
set -eux

for i in 100000,4,2-00:00:00; do 
    IFS=","; set -- $i; 
    sbatch --mem 120G -t $3 -J nl_val_$1 ./launcher_basic.sh python -u experiments/natural_language/validation_experiments.py --num_samples $1 --num_runs $2
done
