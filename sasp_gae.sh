#!/bin/bash

# --- SLURM Directives ---
#SBATCH --job-name=sasp
#SBATCH --output=out.out       # Standard output file (%j will be replaced by jobid)
#SBATCH --error=err.err        # Standard error file
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

ALPHAS=(0.005 0.006 0.007 0.008 0.009 0.01)
SEEDS=(1 2 3 4 5)

# Loop over all combinations
for seed in "${SEEDS[@]}"; do
    echo "Running seed=$seed, alpha=$alpha"
    python train_tgae.py --seed $seed --alpha $1
    python encode_sasp_ukb.py --seed $seed --alpha $1
    python encode_sasp_medex.py --seed $seed --alpha $1
done

conda deactivate
