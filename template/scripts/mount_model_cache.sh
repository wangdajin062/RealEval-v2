#!/bin/bash
# mount_model_cache.sh — Auto-mount HF model cache from host or network storage
# Called by entrypoint.sh at container startup.

set -e

CACHE_SOURCES=(
    "/host_cache/hf_cache"
    "/mnt/models/hf_cache"
    "/workspace/hf_cache"
)

echo "[mount_model_cache] Checking model cache sources..."

for src in "${CACHE_SOURCES[@]}"; do
    if [ -d "$src" ] && [ "$(ls -A "$src" 2>/dev/null)" ]; then
        echo "[mount_model_cache] Found cache at $src"
        export HF_HOME="$src"
        export TRANSFORMERS_CACHE="$src"
        export TORCH_HOME="$src/torch"
        echo "[mount_model_cache] HF_HOME set to $src"
        return 0
    fi
done

echo "[mount_model_cache] No external cache found, using /workspace/hf_cache"
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
