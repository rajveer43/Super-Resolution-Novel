#!/usr/bin/env bash
set -euo pipefail
cd /Users/rajveerrathod/Work/gsoc/Super-Resolution-Novel
export PYTORCH_ENABLE_MPS_FALLBACK=1
C=configs/experiments/calogan.yaml
OV=(exp_name=calogan train.epochs=25 train.max_steps_per_epoch=1000 jepa.epochs=15 baseline.epochs=20 downstream.epochs=15 downstream.probe_epochs=10 eval.n_events=2000)
.venv/bin/python scripts/train_sr.py         --config $C -o "${OV[@]}"
.venv/bin/python scripts/train_baseline.py   --config $C --name all -o "${OV[@]}"
.venv/bin/python scripts/train_downstream.py --config $C -o "${OV[@]}"
.venv/bin/python scripts/evaluate.py         --config $C -o "${OV[@]}"
.venv/bin/python scripts/resolution_sweep.py --config $C -o "${OV[@]}"
