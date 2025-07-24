#!/bin/bash

# --- SLURM Directives ---
#SBATCH --job-name=sasp
#SBATCH --output=%j.out       # Standard output file (%j will be replaced by jobid)
#SBATCH --error=%j.err        # Standard error file
#SBATCH --time=12:00:00
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

ALPHAS=(0.03 0.04 0.06 0.08 0.09)
SEEDS=(1 2 3 4 5)
prefix="age"

for alpha in "${ALPHAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running bn=$1 alpha=$alpha seed=$seed"
        python train_tgae.py  --prefix $prefix --bn $1 --alpha $alpha --seed $seed
        python encode_sasp_ukb.py --prefix $prefix --bn $1 --alpha $alpha --seed $seed
        python encode_sasp_medex.py --prefix $prefix --bn $1 --alpha $alpha --seed $seed
    done
done

conda deactivate
