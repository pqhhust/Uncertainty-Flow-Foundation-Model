#!/bin/bash
# Fine-tune Qwen2.5-0.5B on GLUE tasks
# Usage: bash scripts/finetune.sh [task] [mode]
#   task: sst2, cola, mnli, mrpc (default: sst2)
#   mode: full, lora, linear_probe (default: full)

set -e

TASK=${1:-sst2}
MODE=${2:-full}

echo "============================================"
echo "Fine-tuning Qwen2.5-0.5B on ${TASK} (${MODE})"
echo "============================================"

# Activate environment
source activate uffm 2>/dev/null || conda activate uffm 2>/dev/null || true

cd "$(dirname "$0")/.."

python src/train.py \
    experiment=finetune_${TASK} \
    model.finetune_mode=${MODE} \
    tags="[qwen2,${TASK},${MODE}]" \
    "$@"  # Pass any additional Hydra overrides
