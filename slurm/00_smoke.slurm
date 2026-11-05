#!/bin/bash
# Submitted by "bash slurm/submit_all.sh --smoke" (debug QOS, 1 GPU, 30 min).
# Writes a few hundred synthetic events in the real file schemas, runs every stage with tiny settings
# on the synthetic config of the chosen dataset, then checks that the outputs exist. Prints "SMOKE OK".
#SBATCH -J calonsf-smoke
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh --smoke}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
# calogan_noisy -> calogan_synthetic
CONFIG_NAME="${CONFIG_NAME%%_*}_synthetic"
CONFIG="configs/experiments/${CONFIG_NAME}.yaml"
EXTRA_SET="train.max_steps_per_epoch=40 eval.lr_factors=[2]"
job_header "smoke test"

cd "$REPO"
python scripts/make_synthetic.py --n-qg 400 --n-calogan 400 --n-cc 400
run_py pretrain_jepa.py
run_py train_sr.py
run_py train_baseline.py --name all
run_py train_downstream.py
run_py evaluate.py
run_py resolution_sweep.py

exp="$(experiment_dir)"
missing=0
check() {
    if compgen -G "$1" > /dev/null; then
        echo "  ok      $1"
    else
        echo "  MISSING $1"
        missing=1
    fi
}
echo "checking outputs in $exp"
check "$exp/jepa/encoder.pt"
check "$exp/sr/best.pt"
check "$exp/baselines/edsr_x*/best.pt"
check "$exp/downstream/task/best.pt"
check "$exp/eval/metrics.json"
check "$exp/eval/sweep.json"
check "$exp/eval/report.md"
((missing == 0)) || die "smoke test finished but outputs are missing (see above)"
echo "SMOKE OK"
