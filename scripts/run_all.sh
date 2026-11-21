#!/usr/bin/env bash
# End-to-end pipeline for one experiment config:
#   JEPA pretraining -> CaloNSF SR training -> CaloChallenge Dataset 2 evaluation -> beyond-native checks
# Usage: scripts/run_all.sh configs/experiments/calochallenge_synthetic.yaml [extra -o overrides...]
set -euo pipefail
CFG=$1; shift || true
PY=${PY:-.venv/bin/python}
export PYTORCH_ENABLE_MPS_FALLBACK=1
OV=("$@")
$PY scripts/pretrain_jepa.py --config "$CFG" -o "${OV[@]+"${OV[@]}"}"
$PY scripts/train_sr.py      --config "$CFG" -o "${OV[@]+"${OV[@]}"}"
$PY scripts/evaluate.py      --config "$CFG" -o "${OV[@]+"${OV[@]}"}"
$PY scripts/beyond_native.py --config "$CFG" -o "${OV[@]+"${OV[@]}"}"
