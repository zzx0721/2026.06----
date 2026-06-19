#!/usr/bin/env bash
set -euo pipefail

: "${TA_DATA_ROOT:?Set TA_DATA_ROOT to the downloaded xiaoma26/calvin-lerobot directory}"
WORK_ROOT="${WORK_ROOT:-$PWD/runs/task2_act}"

for step in 020000 040000 060000 080000 100000; do
  output="$WORK_ROOT/eval_d_steps/step_$step"
  if [[ -f "$output/metrics.json" ]]; then
    echo "Skip completed step_$step"
    continue
  fi

  python -m task2_act.evaluate_action_l1 \
    --ta-eval-root "$TA_DATA_ROOT/splitD" \
    --target-frames 40000 \
    --model "b=$WORK_ROOT/outputs/act_b/checkpoints/step_$step" \
    --model "abc=$WORK_ROOT/outputs/act_abc/checkpoints/step_$step" \
    --output-dir "$output"
done
