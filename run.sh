#!/bin/bash
# run.sh — Quick run script for QAD-MultiGuard experiments
# Usage: bash run.sh [--smoke | --paper] [--exp 1,3,6] [--benchmark] [--resume]
set -e
cd "$(dirname "$0")"

MODE="--smoke"
PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke)      MODE="--smoke"; shift ;;
        --paper)      MODE="--paper"; shift ;;
        --exp)        PASS+=("--exp" "$2"); shift 2 ;;
        --benchmark)  PASS+=("--benchmark"); shift ;;
        --resume)     PASS+=("--resume"); shift ;;
        --check|--storage-check) PASS+=("$1"); shift ;;
        -h|--help)
            echo "Usage: bash run.sh [--smoke | --paper] [--exp 1,3,6] [--benchmark] [--resume]"
            echo "  --smoke      sandbox verification (default)"
            echo "  --paper      paper-grade run (needs real Qwen + H100)"
            echo "  --exp LIST   comma-separated experiment numbers, or omit for all"
            exit 0 ;;
        *)
            echo "❌ Unknown argument: $1" >&2
            echo "   Run 'bash run.sh --help' for usage." >&2
            exit 2 ;;
    esac
done

python -m experiments.runner "$MODE" "${PASS[@]}"
