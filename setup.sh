#!/bin/bash
# setup.sh — Environment setup for QAD-MultiGuard
# Usage: bash setup.sh

set -e
cd "$(dirname "$0")"

echo "=== QAD-MultiGuard Environment Setup ==="

# Create virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "Created virtual environment"
fi

# Cross-platform virtualenv activation (Linux/macOS: bin/, Windows Git-Bash: Scripts/)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "❌ Could not find venv activation script" >&2; exit 1
fi

# Install dependencies (from pyproject.toml)
pip install --upgrade pip
pip install -e .

# Create output directories
mkdir -p outputs/predictions outputs/metrics outputs/statistics outputs/figures outputs/tables outputs/logs outputs/results

echo "=== Setup complete ==="
echo "Run experiments: python -m experiments.runner --smoke"
