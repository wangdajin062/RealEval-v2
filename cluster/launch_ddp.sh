#!/bin/bash
# launch_ddp.sh — Multi-GPU DDP launch (exp1 training)
# Usage: NGPU=8 ./cluster/launch_ddp.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

NGPU="${NGPU:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
echo "========================================="
echo " Launching DDP Training (${NGPU} GPUs)"
echo "========================================="

torchrun \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    -m experiments.runner --exp 1 --paper --resume
