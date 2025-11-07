#!/bin/bash

#SBATCH --job-name=ab12057_picollm             # Update with your NYU NetID
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
## SBATCH --partition=a100_1
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/scratch/ab12057/pico-llm       # Update with your NYU NetID
#SBATCH --mail-type=all
#SBATCH --mail-user=ab12057@nyu.edu             # Update with your NYU NetID


echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Current working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the environment
source .venv/bin/activate

# Run the training script with pico_llm.sh configurations
bash src/sh/pico_llm.sh

# Deactivate the environment
deactivate

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
