#!/bin/bash
# setup_runpod.sh — RunPod (PyTorch 2.8.0 image) environment initialisation
# The RunPod PyTorch image ships torch pre-installed; we do NOT reinstall it (avoids conflicts
# and wasted time). Only the remaining dependencies are installed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

PY="${PYTHON:-python}"

echo "========================================="
echo " RunPod Environment Setup (PyTorch 2.8)"
echo "========================================="

echo "[1/4] Verifying pre-installed PyTorch..."
$PY - <<'PYCHK'
import torch
print(f"  torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
PYCHK

echo "[2/4] Installing remaining dependencies (torch is provided by the image, skipped)..."
# Install everything in requirements.txt except torch (already present in the image).
grep -viE '^\s*torch(\b|[><=~!])' requirements.txt > /tmp/reqs_no_torch.txt
$PY -m pip install --no-input -r /tmp/reqs_no_torch.txt

echo "[3/4] Downloading experiment models..."
bash cluster/manage_models.sh download

echo "[4/4] Hardware + storage checks..."
$PY -m experiments.runner --check || true
$PY -m experiments.runner --storage-check || true

echo ""
echo "✅ RunPod setup complete."
echo "   Paper-grade run: REALEVAL_DATA__SOURCE=taf28k ./cluster/launch_runpod_h100.sh"
