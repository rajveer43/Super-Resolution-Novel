#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# CaloNSF super-resolution training on the JEPA-pretrained encoder.
# Resumes from last.pt after a timeout or preemption.
#SBATCH -J calonsf-sr
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
skip_if_done sr
job_header "train_sr"
run_py train_sr.py
mark_done sr
