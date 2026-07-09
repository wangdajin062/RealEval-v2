#!/bin/bash
# download_assets.sh — Download experiment models (Qwen2.5 series + whisper-tiny)
# Usage: ./cluster/download_assets.sh
#   STAGE_LARGE=1 ./cluster/download_assets.sh   # Also download 1.5B/7B for teacher-scale ablation
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

MODELS_ROOT="${REALEVAL_MODELS_ROOT:-/workspace/models}"
HF_CACHE="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$MODELS_ROOT" "$HF_CACHE"

echo "========================================="
echo " Downloading Experiment Models"
echo " Models root: $MODELS_ROOT"
echo " HF cache:   $HF_CACHE"
echo "========================================="

download_model() {
    local repo="$1"
    local target="$2"
    if [ -f "$target/config.json" ] && ls "$target"/*.safetensors 1>/dev/null 2>&1; then
        echo "  ✅ $repo already exists, skipping"
        return 0
    fi
    echo "  Downloading $repo ..."
    HF_HOME="$HF_CACHE" huggingface-cli download "$repo" --local-dir "$target"
    echo "  ✅ $repo download complete"
}

download_model "Qwen/Qwen2.5-0.5B-Instruct" "$MODELS_ROOT/Qwen/Qwen2.5-0.5B-Instruct"
download_model "Qwen/Qwen2.5-0.5B" "$MODELS_ROOT/Qwen/Qwen2.5-0.5B"
download_model "openai/whisper-tiny" "$MODELS_ROOT/openai/whisper-tiny"

if [ "${STAGE_LARGE:-0}" = "1" ]; then
    echo "  [STAGE_LARGE] Downloading 1.5B/7B for teacher-scale ablation..."
    download_model "Qwen/Qwen2.5-1.5B-Instruct" "$MODELS_ROOT/Qwen/Qwen2.5-1.5B-Instruct"
    download_model "Qwen/Qwen2.5-7B-Instruct" "$MODELS_ROOT/Qwen/Qwen2.5-7B-Instruct"
fi

echo ""
echo "✅ All models downloaded."
echo "   Models root: $MODELS_ROOT"
echo "   HF cache:   $HF_CACHE"
