#!/usr/bin/env bash
set -euo pipefail

pick_python_bin() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    echo "$PYTHON_BIN"
    return
  fi
  if command -v python3.12 >/dev/null 2>&1; then
    echo "python3.12"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  echo "python"
}

require_python_312() {
  local py_bin="$1"
  "$py_bin" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(f"Python 3.12+ is required, found {sys.version.split()[0]}")
print(sys.version.split()[0])
PY
}

if command -v conda >/dev/null 2>&1; then
  conda env create -f environment.yml
  conda run -n cv_hw3_act python -m pip install --upgrade pip
  conda run -n cv_hw3_act python -m pip install -e .
  conda run -n cv_hw3_act python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
else
  PYTHON_BIN="$(pick_python_bin)"
  require_python_312 "$PYTHON_BIN"
  "$PYTHON_BIN" -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel build
  python -m pip install --no-build-isolation -r requirements.txt
  python -m pip install -e .
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
fi
