#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Full evaluation of CaloNSF and every baseline on the test split (metrics.json, report.md, figures).
#SBATCH -J calonsf-eval
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
USE_TUNED=1
skip_if_done eval
job_header "evaluate"
run_py evaluate.py
mark_done eval
