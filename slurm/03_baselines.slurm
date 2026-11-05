#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# One baseline at one LR factor per array task (task i = line i of baseline_tasks in common.sh).
# Resumes from last.pt after a timeout or preemption.
#SBATCH -J calonsf-baseline
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
read -r name factor <<< "$(baseline_task "$SLURM_ARRAY_TASK_ID")"
[[ -n "$name" ]] || die "no baseline for task $SLURM_ARRAY_TASK_ID"
skip_if_done "baseline-${name}-x${factor}"
job_header "train_baseline $name x$factor"
run_py train_baseline.py --name "$name" --factor "$factor"
mark_done "baseline-${name}-x${factor}"
