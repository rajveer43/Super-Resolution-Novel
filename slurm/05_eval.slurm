#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# CaloChallenge Dataset 2 evaluation of CaloNSF (and the nearest / bicubic upsamplers) on the test split:
# metrics.json, report.md, figures; then the beyond-native consistency checks (256^2 ... 2048^2 images).
# Plots the training curves from the CSV logs and packs the results into a tarball at the end.
#SBATCH -J calonsf-eval
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
job_header "evaluate"

# The two steps have their own markers, so a failure in the second does not repeat the first (hours of classifiers).
if [[ -f "$(marker eval)" && "${FORCE:-0}" != 1 ]]; then
    echo "evaluate already finished; skipping"
else
    run_py evaluate.py
    mark_done eval
fi

if [[ -f "$(marker beyond)" && "${FORCE:-0}" != 1 ]]; then
    echo "beyond_native already finished; skipping"
else
    # Memory grows ~3x per doubling (1024^2 and 2048^2 can exceed a 40 GB GPU). A failure here must not lose the
    # evaluation above: fall back to the sizes that fit and say so in the log.
    if ! run_py beyond_native.py; then
        echo "WARNING: beyond_native.py failed with the configured sizes (out of memory?); retrying with 256 and 512"
        run_py beyond_native.py --sizes 256 512 || echo "WARNING: beyond_native.py failed again; see the log above"
    fi
    mark_done beyond
fi
run_py plot_training.py || echo "WARNING: training curves could not be plotted"
bash "$REPO/slurm/pack_results.sh"
