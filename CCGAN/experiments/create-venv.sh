#!/bin/bash
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
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
module load pytorch/2.2

# Make sure required modules are loaded
pip install -r requirements.txt