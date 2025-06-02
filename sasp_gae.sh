#!/bin/bash

# --- SLURM Directives ---
#SBATCH --job-name=sasp
#SBATCH --output=SLURM/sasp_%j.out       # Standard output file (%j will be replaced by jobid)
#SBATCH --error=SLURM/sasp_%j.err        # Standard error file
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shangshu.zhao@uconn.edu

# --- Resource Requests ---
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G

SEED=$1

echo "Starting job on $(hostname)"
echo "Current working directory: $(pwd)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"

module purge

source /gpfs/homefs1/shz19039/miniconda3/etc/profile.d/conda.sh
conda activate climate

python GAE_train.py --seed "$SEED"

conda deactivate
