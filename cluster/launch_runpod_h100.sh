#!/bin/bash
# launch_runpod_h100.sh — RunPod single-card H100 launch
# Optimized for 1×H100 SXM 80GB · 8 vCPU · PyTorch 2.8
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_FX_REMOVE_RANDOM_SEED=1

echo "========================================="
echo " RunPod Single-Card H100 Launch"
echo "========================================="
echo " Step 1/3: Storage check..."
python -m experiments.runner --storage-check || echo "  ⚠ Storage check incomplete, continuing..."
echo " Step 2/3: Hardware check..."
python -m experiments.runner --check || true
echo " Step 3/3: Full paper-grade run..."
exec python -m experiments.runner --exp all --paper --resume --benchmark
