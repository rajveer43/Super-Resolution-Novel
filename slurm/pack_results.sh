#!/bin/bash
# Pack one experiment's results (report, metrics, figures, CSV logs, configs, job logs) into a tarball
# to copy home. Model checkpoints (*.pt) are left out unless --with-checkpoints is given.
# Copies the tarball to CFS_COPY if that is set.
#
#   bash slurm/pack_results.sh [<experiment>] [--with-checkpoints]
# Runs automatically at the end of the sweep job.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"

which_run=""
with_ckpt=0
for a in "$@"; do
    case "$a" in
        --with-checkpoints) with_ckpt=1 ;;
        *) which_run="$a" ;;
    esac
done
load_settings
[[ -n "${RUN_ENV:-}" ]] || load_experiment "$which_run"

exp="$(experiment_dir)"
[[ -d "$exp" ]] || die "no results at $exp"
suffix=""
((with_ckpt)) && suffix="_with_checkpoints"
out="$ROOT/results_${EXPERIMENT}${suffix}.tar.gz"
exclude=()
((with_ckpt)) || exclude=(--exclude='*.pt')

# experiment folder + the job logs, under one top-level directory named after the experiment
stage_dir="$(mktemp -d)"
trap 'rm -rf "$stage_dir"' EXIT
mkdir "$stage_dir/$EXPERIMENT"
ln -s "$exp" "$stage_dir/$EXPERIMENT/runs"
[[ -d "$(log_dir)" ]] && ln -s "$(log_dir)" "$stage_dir/$EXPERIMENT/slurm-logs"
tar -czhf "$out" ${exclude[@]+"${exclude[@]}"} -C "$stage_dir" "$EXPERIMENT"
echo "packed $(du -h "$out" | cut -f1)  $out"
echo "copy home with:  scp $USER@perlmutter.nersc.gov:$out ."

if [[ -n "${CFS_COPY:-}" ]]; then
    mkdir -p "$CFS_COPY"
    cp "$out" "$CFS_COPY/"
    echo "copied to $CFS_COPY/$(basename "$out")"
fi
