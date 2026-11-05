#!/usr/bin/env bash
# End-to-end pipeline for one experiment config:
#   JEPA pretraining -> CaloNSF SR training -> baselines -> downstream models -> evaluation -> resolution sweep
# Usage: scripts/run_all.sh configs/experiments/qg_synthetic.yaml [extra -o overrides...]
set -euo pipefail
CFG=$1; shift || true
PY=${PY:-.venv/bin/python}
export PYTORCH_ENABLE_MPS_FALLBACK=1
OV=("$@")
$PY scripts/pretrain_jepa.py    --config "$CFG" -o "${OV[@]+"${OV[@]}"}"
$PY scripts/train_sr.py         --config "$CFG" -o "${OV[@]+"${OV[@]}"}"
$PY scripts/train_baseline.py   --config "$CFG" --name all -o "${OV[@]+"${OV[@]}"}"
$PY scripts/train_downstream.py --config "$CFG" -o "${OV[@]+"${OV[@]}"}"
$PY scripts/evaluate.py         --config "$CFG" -o "${OV[@]+"${OV[@]}"}"
$PY scripts/resolution_sweep.py --config "$CFG" -o "${OV[@]+"${OV[@]}"}"
