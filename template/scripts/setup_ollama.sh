#!/bin/bash
# setup_ollama.sh — Pull recommended models for RealEval pipeline
# Runs in background after ollama serve starts.

set -e

# Wait for Ollama to be ready
echo "[ollama-setup] Waiting for Ollama to start..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

echo "[ollama-setup] Pulling models..."

# Lightweight models suitable for RealEval distillation experiments
MODELS=(
    "qwen2.5:0.5b"
    "qwen2.5:1.5b"
    "qwen2.5:7b"
)

for model in "${MODELS[@]}"; do
    echo "[ollama-setup] Pulling $model ..."
    ollama pull "$model" 2>/dev/null && echo "[ollama-setup]   $model done" || echo "[ollama-setup]   $model skipped"
done

echo "[ollama-setup] Complete. Models ready."
