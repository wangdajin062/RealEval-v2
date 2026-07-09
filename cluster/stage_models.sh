#!/bin/bash
# stage_models.sh — Stage experiment models to storage server
# Usage: ./cluster/stage_models.sh
#   STAGE_LARGE=1 ./cluster/stage_models.sh   # Also stage 1.5B/7B for teacher-scale ablation
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

MODELS_ROOT="${REALEVAL_MODELS_ROOT:-/workspace/models}"
mkdir -p "$MODELS_ROOT"

echo "========================================="
echo " Staging Experiment Models"
echo " Target: $MODELS_ROOT"
echo "========================================="

stage_model() {
    local repo="$1"
    local target="$2"
    if [ -f "$target/config.json" ] && ls "$target"/*.safetensors 1>/dev/null 2>&1; then
        echo "  ✅ $repo already staged"
    else
        echo "  Staging $repo ..."
        HF_HOME="$MODELS_ROOT/hf_cache" huggingface-cli download "$repo" --local-dir "$target"
        echo "  ✅ $repo staged"
    fi
}

stage_model "Qwen/Qwen2.5-0.5B-Instruct" "$MODELS_ROOT/Qwen/Qwen2.5-0.5B-Instruct"
stage_model "Qwen/Qwen2.5-0.5B" "$MODELS_ROOT/Qwen/Qwen2.5-0.5B"
stage_model "openai/whisper-tiny" "$MODELS_ROOT/openai/whisper-tiny"

if [ "${STAGE_LARGE:-0}" = "1" ]; then
    echo "  [STAGE_LARGE] Staging 1.5B/7B..."
    stage_model "Qwen/Qwen2.5-1.5B-Instruct" "$MODELS_ROOT/Qwen/Qwen2.5-1.5B-Instruct"
    stage_model "Qwen/Qwen2.5-7B-Instruct" "$MODELS_ROOT/Qwen/Qwen2.5-7B-Instruct"
fi

echo ""
echo "✅ Model staging complete."
echo "   Models root: $MODELS_ROOT"
