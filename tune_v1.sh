#!/bin/bash

# --- SLURM Directives ---
#SBATCH --job-name=sasp
#SBATCH --output=%j.out       # Standard output file (%j will be replaced by jobid)
#SBATCH --error=%j.err        # Standard error file
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

ALPHAS=(0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09)
SEEDS=(1 2 3 4 5)
prefix="tune1"
bn=6

for seed in "${SEEDS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        echo "Running seed=$seed, alpha=$alpha"
        python tune_v1.py --prefix $prefix --bn $bn --alpha $alpha --seed $seed
        python encode_sasp_ukb.py --prefix $prefix --bn $bn --alpha $alpha --seed $seed
        python encode_sasp_medex.py --prefix $prefix --bn $bn --alpha $alpha --seed $seed
    done
done

conda deactivate
