#!/bin/bash

# --- SLURM Directives ---
#SBATCH --job-name=sasp_train
#SBATCH --output=SLURM/sasp_train_%j.out       # Standard output file (%j will be replaced by jobid)
#SBATCH --error=SLURM/sasp_train_%j.err        # Standard error file
#SBATCH --time=01:00:00                 # Max wall-time (e.g., 4 hours). Adjust as needed.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=szhao@uchc.edu

# --- Resource Requests ---
#SBATCH --job-name=sasp
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --qos=general
#SBATCH --mem=20G

SEED=$1      # First argument passed to the script
ALPHA=$2     # Second argument passed to the script

echo "Starting job on $(hostname)"
echo "Current working directory: $(pwd)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"

source ~/.bashrc
conda activate main

python GAE_train.py --seed "$SEED" --alpha "$ALPHA"

conda deactivate
