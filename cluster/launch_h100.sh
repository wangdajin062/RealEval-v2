#!/bin/bash
# launch_h100.sh — H100 cluster launch (8×H100)
# Usage: NGPU=8 ./cluster/launch_h100.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

NGPU="${NGPU:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_FX_REMOVE_RANDOM_SEED="${TORCHINDUCTOR_FX_REMOVE_RANDOM_SEED:-1}"

echo "========================================="
echo " H100 Cluster Launch (${NGPU}×H100)"
echo "========================================="
echo " CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo " OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo " PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo ""

exec python -m experiments.runner --exp all --paper --resume --benchmark
