#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Resolution sweep: native -> native x 2^k (time, memory, energy conservation, pooled consistency,
# observable drift). Above-native outputs have no ground truth. Runs after the evaluation, adds the sweep
# table to report.md, then packs the results (and copies them to CFS_COPY if set).
#SBATCH -J calonsf-sweep
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
USE_TUNED=1
skip_if_done sweep
job_header "resolution_sweep"
run_py resolution_sweep.py
mark_done sweep
bash "$REPO/slurm/pack_results.sh"
