!#/bin/bash
set -eux

for model_size in xsmall small medium large xlarge; do 
    for i in 25000,4,6:00:00 100000,4,12:00:00 500000,4,2-00:00:00; do 
        IFS=","; set -- $i; 
        sbatch -t $3 -J nl_mse_${1} ./launcher_basic.sh python -u experiments/natural_language/model_size_experiments.py --num_samples $1 --num_runs $2 --model_sizes ${model_size};
    done
done