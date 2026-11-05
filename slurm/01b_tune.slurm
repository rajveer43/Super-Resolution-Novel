#!/bin/bash
# Submitted by slurm/submit_all.sh when TUNE=1 in settings.sh, after JEPA pretraining.
# Optuna study of the CaloNSF training hyperparameters (at most 15 trials, scripts/tune.py). Writes
# runs/<experiment>/tune/best_overrides.txt, which the sr, eval and sweep jobs then use.
# The study is stored in SQLite: after a timeout, re-submit with --from tune and it continues.
#SBATCH -J calonsf-tune
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
skip_if_done tune
job_header "tune (Optuna)"
run_py tune.py
mark_done tune
