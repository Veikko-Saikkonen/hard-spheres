#!/bin/bash
#SBATCH --job-name=ccgan-training
#SBATCH --account=project_2010169
#SBATCH --output="{EXPERIMENT_NAME}.out"
#SBATCH --error="{EXPERIMENT_NAME}.err"
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:1

# Create a virtual environment for the project if it doesn't exist
if [ ! -d ".venv" ]; then
    bash create-venv.sh
    echo "Creating virtual environment..."
else
    echo "Virtual environment already exists. Activating it."
fi
source .venv/bin/activate

# Make sure required modules are loaded
pip install -r additional_requirements_with_dist.txt

# Define name of the experiment
# Experiment name
EXPERIMENT_NAME="2-with-dist"
RESULTS_DIR="results/${EXPERIMENT_NAME}"

# If directory already exists, remove it
if [ -d "${RESULTS_DIR}" ]; then
    echo "Directory ${RESULTS_DIR} already exists. Removing it."
    rm -rf "${RESULTS_DIR}"
fi

mkdir -p "${RESULTS_DIR}"

python3 train.py\
        --training_data ../data/processed/samples\
        --gen_int 5\
        --gsave_freq 20\
        --n_save 20\
        --print_freq 5\
        --msave_freq 50\
        --msave_dir "${RESULTS_DIR}/"\
        --gsave_dir "${RESULTS_DIR}/"\
        --gen_channels_1 256\
        --latent_dim 128\
        --gen_label_dim 32\
        --disc_label_dim 32\
        --weight_dist 0.05\


