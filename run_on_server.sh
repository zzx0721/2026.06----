#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash run_on_server.sh /path/to/calvin-lerobot [/path/to/work_root]"
  exit 2
fi

export TA_DATA_ROOT="$1"
export WORK_ROOT="${2:-$PWD/runs/task2_act}"

if command -v conda >/dev/null 2>&1; then
  if ! conda env list | grep -q '^cv_hw3_act '; then
    bash scripts/bootstrap_autodl.sh
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate cv_hw3_act
else
  if [ ! -d ".venv" ]; then
    bash scripts/bootstrap_autodl.sh
  fi
  source .venv/bin/activate
fi

if [ ! -f "$TA_DATA_ROOT/download_manifest.json" ] || [ ! -d "$TA_DATA_ROOT/splitA" ] || [ ! -d "$TA_DATA_ROOT/splitD" ]; then
  python -m task2_act.download_ta_dataset --output-root "$TA_DATA_ROOT"
fi

bash scripts/run_full_experiment.sh
