#!/bin/bash
#SBATCH --job-name=QAD-MultiGuard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm_%j.out
#SBATCH --error=outputs/slurm_%j.err
#SBATCH --partition=gpu_h100

# SLURM H100 batch job -- QAD-MultiGuard paper-grade run
set -euo pipefail
cd /path/to/h100_package_realeval

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_FX_REMOVE_RANDOM_SEED=1

echo "========================================="
echo " SLURM H100 Job Started"
echo " Job ID: $SLURM_JOB_ID"
echo " Node:   $(hostname)"
echo " GPUs:   $(nvidia-smi -L | wc -l)"
echo "========================================="

python -m experiments.runner --exp all --paper --resume --benchmark

echo "========================================="
echo " Job Complete"
echo "========================================="
