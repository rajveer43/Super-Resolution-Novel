#!/bin/bash
# Progress of one experiment: queued / running jobs, finished stages, latest training lines, and the
# end of each stage's newest log.
#
#   bash slurm/status.sh                 the most recently submitted experiment
#   bash slurm/status.sh <experiment>    a given one, e.g. 2026-11-05_calogan_full
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
load_experiment "${1:-}"
exp="$(experiment_dir)"
logdir="$(log_dir)"

echo "experiment $EXPERIMENT  ($exp)"
echo
echo "---- jobs in the queue"
ids="$(cut -f3 "$logdir/jobs.tsv" 2> /dev/null | paste -sd, -)"
if [[ -z "$ids" ]]; then
    echo "(no jobs submitted yet)"
elif command -v squeue > /dev/null; then
    squeue -j "$ids" -o '%.14i %.40j %.9T %.10M %.10l %.24R' 2> /dev/null || echo "(none of this experiment's jobs are queued)"
fi

echo
echo "---- finished stages"
for m in jepa sr downstream eval sweep; do
    if [[ -f "$exp/.done_$m" ]]; then echo "done    $m  ($(cat "$exp/.done_$m"))"; else echo "-       $m"; fi
done
n_done=$(compgen -G "$exp/.done_baseline-*" | wc -l | tr -d ' ')
echo "        baselines finished: $n_done  ($(compgen -G "$exp/.done_baseline-*" | sed 's/.*\.done_baseline-//' | paste -sd' ' -))"

echo
echo "---- latest epoch lines"
for f in "$exp"/jepa/*_log.csv "$exp"/sr/*_log.csv; do
    [[ -f "$f" ]] || continue
    echo "== ${f#"$exp"/}: $(($(wc -l < "$f") - 1)) epochs done"
    tail -n 2 "$f" | cut -c1-160
done

echo
echo "---- end of the latest log of each stage ($logdir)"
for stage in jepa sr baselines downstream eval sweep smoke; do
    # shellcheck disable=SC2012
    f="$(ls -t "$logdir/$stage"-*.out 2> /dev/null | head -1)"
    [[ -n "$f" ]] || continue
    echo "== $(basename "$f")"
    tail -n 3 "$f" | cut -c1-200
done
grep -l -i -E "traceback|error:|out of memory|DUE TO TIME LIMIT" "$logdir"/*.out 2> /dev/null \
    | sed 's/^/!! problem reported in /'
[[ -f "$exp/eval/report.md" ]] && echo && echo "report: $exp/eval/report.md"
compgen -G "$ROOT/results_${EXPERIMENT}*.tar.gz" > /dev/null && echo "packed: $(ls "$ROOT"/results_"${EXPERIMENT}"*.tar.gz)"
exit 0
