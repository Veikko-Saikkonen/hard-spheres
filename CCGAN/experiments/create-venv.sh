#!/bin/bash -l
#SBATCH --job-name=ccgan-create-venv
#SBATCH --output=ccgan-create-venv.out
#SBATCH --error=ccgan-create-venv.err
#SBATCH --account=project_2010169
#SBATCH --partition=small
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24000


# Create a virtual environment for the project if it doesn't exist
module load pytorch/2.2
echo "Creating virtual environment..."
python3.9 -m venv --system-site-packages .venv
source .venv/bin/activate
echo "Updating pip, setuptools, and wheel..."
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade wheel
echo "Installing requirements..."
# Make sure required modules are loaded
pip install -r requirements.txt
# Install pytorch3d
# If operating system is Linux, use the following command
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
fi
# If operating system is MacOS, use the following command
if [[ "$OSTYPE" == "darwin"* ]]; then
    MACOSX_DEPLOYMENT_TARGET=10.14 CC=clang CXX=clang++ pip install "git+https://github.com/facebookresearch/pytorch3d.git"
fi

echo "Virtual environment created and requirements installed."
