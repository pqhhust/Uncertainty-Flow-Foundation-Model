#!/bin/bash
# Distill fine-tuned Qwen2 into a flow-based model
# Usage: bash scripts/distill.sh [task] [teacher_ckpt] [mode] [extra overrides...]
#   task: sst2, cola (default: sst2)
#   teacher_ckpt: path to fine-tuned teacher checkpoint (REQUIRED)
#   mode: velocity, meanflow (default: velocity)

set -e

TASK=${1:-sst2}
TEACHER_CKPT=${2:-"/home/user01/aiotlab/pqhung/uncertainty/Uncertainty-Flow-Foundation-Model/logs/train/runs/2026-02-26_14-04-20/checkpoints/last.ckpt"}
MODE=${3:-velocity}

if [ -z "$TEACHER_CKPT" ]; then
    echo "Error: teacher_ckpt is required"
    echo "Usage: bash scripts/distill.sh [task] [teacher_ckpt_path] [mode]"
    echo "  mode: velocity (default) or meanflow"
    echo "Example: bash scripts/distill.sh sst2 logs/.../best.ckpt meanflow"
    exit 1
fi

# Select experiment config based on mode
if [ "$MODE" == "meanflow" ]; then
    EXPERIMENT="meanflow_${TASK}"
else
    EXPERIMENT="distill_flow_${TASK}"
fi

echo "============================================"
echo "Distilling Qwen2 to Flow Model on ${TASK}"
echo "Mode: ${MODE} | Experiment: ${EXPERIMENT}"
echo "Teacher: ${TEACHER_CKPT}"
echo "============================================"

python src/train.py \
    experiment=${EXPERIMENT} \
    model.teacher_ckpt_path="${TEACHER_CKPT}" \
    tags="[qwen2,${TASK},distill,${MODE}]" \
    logger=wandb \
    logger.wandb.project="uffm" \
    "${@:4}"  # Pass any additional Hydra overrides (args after task, ckpt, mode)
