!#/bin/sh

# Dataset Experiments
sbatch -J al_dse -t 1-00:00:00 ./launcher_basic.sh python -u experiments/artificial_language/dataset_experiments.py

# Model Size Experiments:
sbatch -J al_mse -t 1-00:00:00 ./launcher_basic.sh python -u experiments/artificial_language/model_size_experiments.py

# Beam Size Experiments:
sbatch -J al_bse -t 1-00:00:00 ./launcher_basic.sh python -u experiments/artificial_language/beam_size_experiments.py

# Validation Experiments:
sbatch -J al_val -t 2-00:00:00 ./launcher_basic.sh python -u experiments/artificial_language/validation_experiments.py

# Vocab Size Experiments:
sbatch -J al_vse -t 2-00:00:00 ./launcher_basic.sh python -u experiments/artificial_language/vocab_size_experiments.py

# Scheduled Sampling Experiments:
sbatch -J al_ss -t 2-00:00:00 ./launcher_basic.sh python -u experiments/artificial_language/scheduled_sampling_experiments.py


