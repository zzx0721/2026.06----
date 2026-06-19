#!/usr/bin/env bash
set -euo pipefail

: "${TA_DATA_ROOT:?Set TA_DATA_ROOT to the downloaded xiaoma26/calvin-lerobot directory}"
WORK_ROOT="${WORK_ROOT:-$PWD/runs/task2_act}"

OUTPUT_ROOT="$WORK_ROOT/outputs"
EVAL_ROOT="$WORK_ROOT/eval_d"

python -m task2_act.train_act \
  --ta-split-root "$TA_DATA_ROOT/splitB" \
  --target-frames 120000 \
  --output-dir "$OUTPUT_ROOT/act_b" \
  --wandb-project cv_hw3_act \
  --wandb-run-name act_b_120k \
  --wandb-mode offline

python -m task2_act.train_act \
  --ta-split-root "$TA_DATA_ROOT/splitA" \
  --ta-split-root "$TA_DATA_ROOT/splitB" \
  --ta-split-root "$TA_DATA_ROOT/splitC" \
  --frames-per-split 40000 \
  --output-dir "$OUTPUT_ROOT/act_abc" \
  --wandb-project cv_hw3_act \
  --wandb-run-name act_abc_120k \
  --wandb-mode offline

python -m task2_act.evaluate_action_l1 \
  --ta-eval-root "$TA_DATA_ROOT/splitD" \
  --target-frames 40000 \
  --model b="$OUTPUT_ROOT/act_b/checkpoints/step_100000" \
  --model abc="$OUTPUT_ROOT/act_abc/checkpoints/step_100000" \
  --output-dir "$EVAL_ROOT"
