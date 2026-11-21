#!/usr/bin/env bash
# Laptop smoke run of the whole framework on toy CaloChallenge Dataset 2 data.
set -euo pipefail
PY=${PY:-.venv/bin/python}
$PY scripts/make_synthetic.py --n-cc 1500
bash scripts/run_all.sh configs/experiments/calochallenge_synthetic.yaml
