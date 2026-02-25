#!/bin/bash
# Run full pipeline: fine-tune on all GLUE tasks, then distill each
# Usage: bash scripts/run_all.sh

set -e

cd "$(dirname "$0")/.."

TASKS=("sst2" "cola" "mrpc")

echo "============================================"
echo "UFFM Full Pipeline: Fine-tune + Distill"
echo "Tasks: ${TASKS[*]}"
echo "============================================"

for TASK in "${TASKS[@]}"; do
    echo ""
    echo ">>> Fine-tuning on ${TASK}..."
    python src/train.py experiment=finetune_${TASK}

    # Find latest checkpoint
    CKPT=$(find logs/train/ -name "*.ckpt" -path "*${TASK}*" | sort -r | head -1)
    if [ -z "$CKPT" ]; then
        echo "Warning: No checkpoint found for ${TASK}, skipping distillation"
        continue
    fi

    echo ""
    echo ">>> Distilling ${TASK} (teacher: ${CKPT})..."
    python src/train.py experiment=distill_flow_${TASK} model.teacher_ckpt_path="${CKPT}"
done

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
