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
module load pytorch/2.2
echo "Creating virtual environment..."
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade wheel

# Make sure required modules are loaded
pip install -r requirements.txt
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html