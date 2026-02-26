#!/bin/bash
# Distill fine-tuned Qwen2 into a flow-based model
# Usage: bash scripts/distill.sh [task] [teacher_ckpt]
#   task: sst2, cola (default: sst2)
#   teacher_ckpt: path to fine-tuned teacher checkpoint (REQUIRED)

set -e

TASK=${1:-sst2}
TEACHER_CKPT=${2:-"/home/user01/aiotlab/pqhung/uncertainty/Uncertainty-Flow-Foundation-Model/logs/train/runs/2026-02-26_14-04-20/checkpoints/last.ckpt"}

if [ -z "$TEACHER_CKPT" ]; then
    echo "Error: teacher_ckpt is required"
    echo "Usage: bash scripts/distill.sh [task] [teacher_ckpt_path]"
    echo "Example: bash scripts/distill.sh sst2 logs/train/runs/2026-02-25/checkpoints/best.ckpt"
    exit 1
fi

echo "============================================"
echo "Distilling Qwen2 to Flow Model on ${TASK}"
echo "Teacher: ${TEACHER_CKPT}"
echo "============================================"

# Activate environment
# source activate uffm 2>/dev/null || conda activate uffm 2>/dev/null || true

# cd "$(dirname "$0")/.."

python src/train.py \
    experiment=distill_flow_${TASK} \
    model.teacher_ckpt_path="${TEACHER_CKPT}" \
    tags="[qwen2,${TASK},distill,flow]" \
    logger=wandb \
    logger.wandb.project="uffm" \
    "${@:3}"  # Pass any additional Hydra overrides (args after task and ckpt)
