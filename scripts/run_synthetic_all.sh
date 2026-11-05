#!/usr/bin/env bash
# Laptop smoke run of the whole framework on toy data for all three datasets.
set -euo pipefail
PY=${PY:-.venv/bin/python}
$PY scripts/make_synthetic.py --n-qg 1500 --n-calogan 1000 --n-cc 1500
for d in qg calogan calochallenge; do
  bash scripts/run_all.sh configs/experiments/${d}_synthetic.yaml
done
