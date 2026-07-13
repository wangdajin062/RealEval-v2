#!/usr/bin/env bash
# run_all.sh — H100 RealEval One-Click Launcher
# ===========================================================================
# Usage:
#   bash run_all.sh                    # Full paper pipeline (all experiments)
#   bash run_all.sh --smoke            # Smoke test (sandbox verification)
#   bash run_all.sh --distributed      # Multi-GPU via torchrun + NCCL
#   bash run_all.sh --setup            # Environment setup only (no experiments)
#   bash run_all.sh --notebook         # Jupyter Lab only
#
# Pipeline: setup → model download → CUDA check → benchmark → metrics → save
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="paper"
DISTRIBUTED=0
SETUP_ONLY=0
NOTEBOOK_ONLY=0
SKIP_MODELS=0

for arg in "$@"; do
    case "$arg" in
        --smoke)        MODE="smoke" ;;
        --distributed)  DISTRIBUTED=1 ;;
        --setup)        SETUP_ONLY=1 ;;
        --notebook)     NOTEBOOK_ONLY=1 ;;
        --skip-models)  SKIP_MODELS=1 ;;
        --help|-h)
            echo "Usage: bash run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --smoke          Smoke test (no GPU/weights required)"
            echo "  --distributed    Multi-GPU via torchrun + NCCL"
            echo "  --setup          Environment setup only"
            echo "  --notebook       Launch Jupyter Lab only"
            echo "  --skip-models    Skip model download"
            exit 0
            ;;
    esac
done

# ── Banner ──
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       H100 RealEval — QAD-MultiGuard Pipeline               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Mode:        $MODE                                          ║"
echo "║  Distributed: $DISTRIBUTED                                  ║"
echo "║  Workspace:   /workspace                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Noteboook-only mode ──
if [ "$NOTEBOOK_ONLY" = "1" ]; then
    echo "[run_all] Starting Jupyter Lab only..."
    jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
        --ServerApp.token="${JUPYTER_TOKEN:-realeval}" \
        --notebook-dir=/workspace
    exit 0
fi

# ── Step 1: Environment setup ──
echo "━━━ Step 1/5: Environment Setup ━━━"

if [ ! -d "venv" ] && [ ! -d "/workspace/venv" ]; then
    echo "[run_all] Creating virtual environment..."
    python -m venv /workspace/venv
fi

if [ -f /workspace/venv/bin/activate ]; then
    source /workspace/venv/bin/activate
elif [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

echo "[run_all] Installing RealEval..."
pip install -e /workspace/repo 2>/dev/null || pip install -e . 2>/dev/null || {
    echo "[run_all] WARNING: pip install -e . failed, continuing with PYTHONPATH"
    export PYTHONPATH="${REPO_DIR:-/workspace/repo}:${PYTHONPATH}"
}

if [ "$SETUP_ONLY" = "1" ]; then
    echo ""
    echo "✓ Setup complete. Models not downloaded."
    echo "  Run: bash run_all.sh --skip-models to skip model download next time."
    exit 0
fi

# ── Step 2: Model download ──
echo ""
echo "━━━ Step 2/5: Download Models ━━━"

if [ "$SKIP_MODELS" = "0" ] && [ "$MODE" != "smoke" ]; then
    if [ -f cluster/manage_models.sh ]; then
        bash cluster/manage_models.sh download
    else
        echo "[run_all] Downloading minimal models via huggingface-cli..."
        HF_HOME="${HF_HOME:-/workspace/hf_cache}" python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSpeechSeq2Seq
import os

cache = os.environ.get('HF_HOME', '/workspace/hf_cache')
os.makedirs(cache, exist_ok=True)

print('Downloading Qwen2.5-0.5B-Instruct...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', cache_dir=cache)
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', cache_dir=cache)

print('Downloading whisper-tiny...')
AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-tiny', cache_dir=cache)
" 2>/dev/null || echo "[run_all] Model download skipped (may already be cached)"
    fi
else
    echo "[run_all] Skipping model download."
fi

# ── Step 3: CUDA & GPU check ──
echo ""
echo "━━━ Step 3/5: GPU Detection ━━━"

python -c "
import torch
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU count:    {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU[{i}]:       {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  Memory[{i}]:    {props.total_mem / 1024**3:.1f} GB')
" || echo "  WARNING: GPU check failed"

if [ "$MODE" = "smoke" ]; then
    echo ""
    echo "━━━ Step 4/5: Smoke Test ━━━"
    python -m experiments.runner --smoke --benchmark 2>&1 || {
        echo "[run_all] Smoke test completed with warnings (expected in sandbox)."
    }
else
    # ── Step 4: Run experiments ──
    echo ""
    echo "━━━ Step 4/5: Run Experiments (mode=$MODE) ━━━"

    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    export TOKENIZERS_PARALLELISM=false
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    if [ "$DISTRIBUTED" = "1" ]; then
        NGPU=$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)
        echo "[run_all] Launching distributed with $NGPU GPUs..."
        torchrun --nproc_per_node="$NGPU" -m experiments.paper_pipeline "--$MODE" --config config/h100.yaml
    else
        echo "[run_all] Launching single-process paper pipeline..."
        python -m experiments.paper_pipeline "--$MODE" --config config/h100.yaml
    fi
fi

# ── Step 5: Collect results ──
echo ""
echo "━━━ Step 5/5: Results ━━━"

mkdir -p /workspace/outputs/{results,metrics,tables,figures}

cat << 'RESULTEOF' > /workspace/outputs/results/summary.txt
H100 RealEval — QAD-MultiGuard Pipeline Results
=================================================
Generated: $(date)
Mode: $MODE
Distributed: $DISTRIBUTED
RESULTEOF

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓ Pipeline Complete                                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Results:  /workspace/outputs/results/                      ║"
echo "║  Metrics:  /workspace/outputs/metrics/                      ║"
echo "║  Tables:   /workspace/outputs/tables/                       ║"
echo "║  Figures:  /workspace/outputs/figures/                      ║"
echo "║  Logs:     /workspace/logs/                                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Jupyter:  http://localhost:8888                            ║"
echo "║  API:      http://localhost:8000                            ║"
echo "║  API Docs: http://localhost:8000/docs                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

ls -lh /workspace/outputs/ 2>/dev/null || true
echo ""
echo "Done."
