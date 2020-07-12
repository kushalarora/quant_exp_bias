!#/bin/bash
set -eux

for rollin in learned teacher_forcing; do
    for rollout in learned mixed reference; do
        for i in 1000,6,1-24:00:00 10000,3,1-24:00:00, 50000,1,2-24:00:00; do
            IFS=","; set -- $i;
            sbatch -c 32 -t $3 -J al_searnn_$1 ./launcher_basic.sh python -u experiments/artificial_language/searnn_experiments.py --num_samples $1 --num_runs $2 --rollins ${rollin} --rollouts ${rollout}
        done
    done
done
