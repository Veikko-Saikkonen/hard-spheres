#!/bin/bash
#SBATCH --job-name=ccgan-training
#SBATCH --account=project_2010169
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:1
module load pytorch
cd ../ # Change to the directory where the script is located

# Define name of the experiment
# Experiment name
EXPERIMENT_NAME="2-with-dist"
RESULTS_DIR="results/${EXPERIMENT_NAME}"

mkdir -p "${RESULTS_DIR}"

python3 train.py\\
        --training_data ../data/processed/samples\\
        --gen_int 5\\
        --gsave_freq 20\\
        --n_save 20\\
        --print_freq 5\\
        --msave_freq 50\\
        --msave_dir "${RESULTS_DIR}"\\
        --gsave_dir "${RESULTS_DIR}"\\
        --gen_channels_1 256\\
        --latent_dim 128\\
        --gen_label_dim 32\\
        --disc_label_dim 32\\
        --weight_dist 0.05