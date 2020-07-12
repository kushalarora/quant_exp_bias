!#/bin/sh

set -eux

# Dataset Experiments
for i in 100,10,3:00:00 1000,8,6:00:00 10000,6,6:00:00 50000,4,12:00:00 500000,3,23:00:00; do
    IFS=","; set -- $i;
    sbatch -J al_dse_$1 -t $3 ./launcher_basic.sh python -u experiments/artificial_language/dataset_experiments.py --num_samples $1 --num_runs $2
done

# Model Size Experiments:
for i in 1000,6,12:00:00 10000,4,12:00:00 50000,3,23:00:00; do
    IFS=","; set -- $i;
    sbatch -J al_mse_$1 -t $3 ./launcher_basic.sh python -u experiments/artificial_language/model_size_experiments.py --num_samples $1 --num_runs $2
done

# Validation Experiments:
for i in 10000,4,24:00:00; do
    IFS=","; set -- $i;
    sbatch -J al_val_$1 -t $3 ./launcher_basic.sh python -u experiments/artificial_language/validation_experiments.py --num_samples $1 --num_runs $2
done

# Vocab Size Experiments:
for i in 1000,6,24:00:00 10000,4,1-8:00:00 50000,3,2-24:00:00; do
    IFS=","; set -- $i;
    sbatch -J al_vse_$1 -c 32 -t $3 ./launcher_basic.sh python -u experiments/artificial_language/vocab_size_experiments.py --num_samples $1 --num_runs $2
done

# Scheduled Sampling Experiments:
for i in 1000,6,12:00:00 10000,4,24:00:00 50000,3,24:00:00; do
    IFS=","; set -- $i;
    sbatch -J al_ss_$1 -t $3 ./launcher_basic.sh python -u experiments/artificial_language/scheduled_sampling_experiments.py --num_samples $1 --num_runs $2
done

# Reinforce Experiments:
for i in 1000,6,1-24:00:00 10000,4,2-24:00:00 50000,3,2-24:00:00; do
    IFS=","; set -- $i;
    sbatch -c 36 -J al_reinforce_$1 -c 32 -t $3 ./launcher_basic.sh python -u experiments/artificial_language/reinforce_experiments.py --num_samples $1 --num_runs $2
done
