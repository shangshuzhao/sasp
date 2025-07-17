#!/bin/bash

# --- SLURM Directives ---
#SBATCH --job-name=sasp
#SBATCH --output=out_%j.out       # Standard output file (%j will be replaced by jobid)
#SBATCH --error=err_%j.err        # Standard error file
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

SEEDS=(1 2 3 4 5)

# Loop over all combinations
for seed in "${SEEDS[@]}"; do
    echo "Running seed=$seed, alpha=$1"
    python tune_v1.py --prefex "t1" --alpha $1 --seed $seed
    python encode_sasp_ukb.py --prefex "t1" --alpha $1 --seed $seed
    python encode_sasp_medex.py --prefex "t1" --alpha $1 --seed $seed
done

conda deactivate
