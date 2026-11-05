#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Downstream models trained on ground-truth HR only (particle ID / energy / q-g tagger, and the
# attentive probe on the frozen JEPA encoder). They are later applied unchanged to every SR output.
#SBATCH -J calonsf-downstream
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
skip_if_done downstream
job_header "train_downstream"
run_py train_downstream.py
mark_done downstream
