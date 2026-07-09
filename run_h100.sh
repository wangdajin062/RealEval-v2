#!/usr/bin/env bash
# One-command H100 paper-validation entry.
#   bash run_h100.sh                 # paper-grade (real Qwen + H100), single process
#   bash run_h100.sh --all           # same, runs all experiment groups (default)
#   bash run_h100.sh --smoke         # sandbox verification (no GPU/weights)
#   bash run_h100.sh --distributed   # 8x H100 via torchrun --nproc_per_node=8 + NCCL
#
# Pipeline: CUDA check -> GPU detect -> env report -> model load -> benchmark -> metrics -> save
# Deliverables: results/{metrics.json,latency.csv,throughput.csv,memory.csv,paper_table.md,
#                        paper_tables/{table1_main,table2_ablation,table3_efficiency}.tex}
set -euo pipefail
cd "$(dirname "$0")"

MODE="--paper"
DISTRIBUTED=0
for a in "$@"; do
  [ "$a" = "--smoke" ] && MODE="--smoke"
  [ "$a" = "--distributed" ] && DISTRIBUTED=1
done

# H100 multi-GPU: expose all 8 cards for NCCL/DDP (harmless if fewer/none present).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "=== H100 paper-validation pipeline ($MODE) ==="
if [ "$DISTRIBUTED" = "1" ]; then
  NGPU="$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
  echo "=== torchrun --nproc_per_node=$NGPU (NCCL) ==="
  torchrun --nproc_per_node="$NGPU" -m experiments.paper_pipeline "$MODE" --config config/h100.yaml
else
  python -m experiments.paper_pipeline "$MODE" --config config/h100.yaml
fi
echo "=== Deliverables in results/ ==="
ls -1 results/ 2>/dev/null || true

