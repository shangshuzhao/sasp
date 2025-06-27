#!/bin/bash

# --- SLURM Directives ---
#SBATCH --job-name=sasp
#SBATCH --output=output_%j.out       # Standard output file (%j will be replaced by jobid)
#SBATCH --error=error_%j.err        # Standard error file
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shangshu.zhao@uconn.edu

# --- Resource Requests ---
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

echo "Starting job on $(hostname)"
echo "Current working directory: $(pwd)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"

module purge

source /gpfs/homefs1/shz19039/miniconda3/etc/profile.d/conda.sh
conda activate climate

ALPHAS=(0.004 0.005 0.007 0.009 0.015 0.02)

# Loop over all combinations
for alpha in "${ALPHAS[@]}"; do
    echo "Running seed=$1, alpha=$alpha"
    python train_tgae.py --seed $1 --alpha $alpha
done

conda deactivate
