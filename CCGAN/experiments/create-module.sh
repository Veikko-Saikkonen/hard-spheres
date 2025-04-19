#!/bin/bash -l
#SBATCH --job-name=ccgan-create-venv
#SBATCH --output=ccgan-create-venv.out
#SBATCH --error=ccgan-create-venv.err
#SBATCH --account=project_2010169
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000


# Create a virtual environment for the project if it doesn't exist
echo "Creating module..."
module load pytorch/2.2
export PYTHONUSERBASE=./ccgan-userbase
pip install --user --upgrade pip
pip install --user --upgrade setuptools
pip install --user --upgrade wheel

# Make sure required modules are loaded
pip install --user -r requirements.txt
pip install --user --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html