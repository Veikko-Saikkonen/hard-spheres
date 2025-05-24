#!/bin/bash -l
# Job name and output files
#SBATCH --job-name=ccgan-training-5-full-scale
#SBATCH --account=project_2010169
#SBATCH --output=5-full-scale.out
#SBATCH --error=5-full-scale.err
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1

EXPERIMENT_NAME="5-full-scale"
echo "Entering experiment ${EXPERIMENT_NAME}..."

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    bash create-venv.sh
else
    echo "Virtual environment already exists. Activating it..."
    # Load custom module
    module load pytorch/2.2
    source .venv/bin/activate
fi

# Define name of the experiment
# Experiment name
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
        --batch_size 64\
        --msave_dir "${RESULTS_DIR}/"\
        --gsave_dir "${RESULTS_DIR}/"\
        --gen_channels_1 256\
        --latent_dim 256\
        --gen_label_dim 128\
        --disc_label_dim 128\
        --weight_dist 0.2\
        --g_lr 0.001\
        --coord_lr 0.0005\
        --dist_lr 0.0005\
        --step_size 50\
        --gamma 0.90\
