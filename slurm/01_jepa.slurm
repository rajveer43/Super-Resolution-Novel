#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Checks the data files (shapes, units, energy closure), then pretrains the cell encoder with JEPA.
#SBATCH -J calonsf-jepa
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
skip_if_done jepa
job_header "check_data + pretrain_jepa"
run_py check_data.py
run_py pretrain_jepa.py
mark_done jepa
