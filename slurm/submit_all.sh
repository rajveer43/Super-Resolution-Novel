#!/bin/bash
# Submit the whole pipeline as a chain of Slurm jobs (run on a login node).
#
#   bash slurm/submit_all.sh                  everything, from JEPA pretraining
#   bash slurm/submit_all.sh --from sr        restart from a stage (earlier stages are not submitted)
#   bash slurm/submit_all.sh --smoke          the 30-minute check on synthetic data (debug QOS)
#   bash slurm/submit_all.sh --dry-run        print the sbatch commands, submit nothing
#
# Stages: jepa -> [tune] -> sr -----------------------------> eval -> sweep -> pack
#             \-> downstream --------------------/
#         baselines[one per model x factor] -----/
# Finished stages are skipped (a .done_<stage> marker), so re-submitting after a failure or timeout is safe.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"

STAGES=(jepa tune sr baselines downstream eval sweep)
FROM=jepa
DRY=0
SMOKE=0
while (($#)); do
    case "$1" in
        --from) FROM="${2:?--from needs a stage}"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        -h | --help) sed -n '2,14p' "$0"; exit 0 ;;
        *) die "unknown option $1" ;;
    esac
done

stage_index() {
    local i
    for i in "${!STAGES[@]}"; do [[ "${STAGES[$i]}" == "$1" ]] && { echo "$i"; return; }; done
    die "unknown stage '$1' (choose from: ${STAGES[*]})"
}
FROM_I=$(stage_index "$FROM")
wanted() { (($(stage_index "$1") >= FROM_I)); }

load_settings
unset RUN_ENV
if ((!SMOKE)) && [[ "$FROM" != jepa && -z "$RUN_DATE" ]]; then
    # a restart continues the latest experiment with this config and tag, even on a later day
    # shellcheck disable=SC2012
    last="$(ls -td "$ROOT"/logs/*_"${CONFIG_NAME}_${RUN_TAG}" 2> /dev/null | head -1 || true)"
    if [[ -n "$last" ]]; then
        RUN_DATE="$(basename "$last" | cut -c1-10)"
        echo "continuing the experiment from $RUN_DATE (set RUN_DATE in settings.sh to pick another)"
    fi
fi
if ((SMOKE)); then
    resolve_experiment -smoke
else
    resolve_experiment
fi
if [[ "$ACCOUNT" == mXXXX* ]]; then
    ((DRY)) || die "set ACCOUNT in slurm/settings.sh"
    echo "(dry run: ACCOUNT is still the placeholder '$ACCOUNT')"
fi

LOGDIR="$(log_dir)"
ENV_FILE="$LOGDIR/run.env"
echo "experiment : $(experiment_dir)"
echo "logs       : $LOGDIR"

# ---- freeze the settings for the jobs ------------------------------------------------------------
write_env() {
    local v
    for v in REPO ROOT DATA_ROOT EXPERIMENT ACCOUNT QOS GPU_MEM CONFIG_NAME RUN_TAG RUN_DATE TUNE EXTRA_SET CFS_COPY; do
        printf 'export %s=%q\n' "$v" "${!v}"
    done
}
if ((DRY)); then
    echo "--- run.env would be:"
    write_env
else
    mkdir -p "$LOGDIR"
    [[ -f "$ENV_FILE" ]] && cp "$ENV_FILE" "$ENV_FILE.$(date +%Y%m%d-%H%M%S)"
    write_env > "$ENV_FILE"
fi

# ---- sbatch wrapper ------------------------------------------------------------------------------
# sub <stage-name> <time> <script> <dependency or ""> [array range or ""]
sub() {
    local name="$1" time="$2" script="$3" dep="$4" array="${5:-}"
    local -a args=(--parsable -J "calonsf_${CONFIG_NAME}_${name}" -t "$time" --kill-on-invalid-dep=yes
                   --export="ALL,RUN_ENV=$ENV_FILE")
    if ((SMOKE)); then
        args+=(-A "$ACCOUNT" -C gpu -q debug --gpus=1 -c 32)
    else
        local constraint=gpu
        [[ "$GPU_MEM" == 80 ]] && constraint="gpu&hbm80g"
        args+=(-A "$ACCOUNT" -C "$constraint" -q "$QOS" --gpus=1 -c 32)
        [[ "$QOS" == regular ]] && args+=(-N 1)
        [[ "$QOS" == preempt ]] && args+=(--requeue)  # training resumes from last.pt
    fi
    if [[ -n "$array" ]]; then
        args+=(--array="$array" -o "$LOGDIR/${name}-%A_%a.out")
    else
        args+=(-o "$LOGDIR/${name}-%j.out")
    fi
    [[ -n "$dep" ]] && args+=(--dependency="$dep")
    args+=("$REPO/slurm/$script")
    if ((DRY)); then
        echo "sbatch ${args[*]}" >&2
        echo "<$name>"
    else
        local id
        id=$(sbatch "${args[@]}")
        id="${id%%;*}"
        printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$id" "${array:--}" >> "$LOGDIR/jobs.tsv"
        echo "  submitted $name: $id${array:+ (array $array)}" >&2
        echo "$id"
    fi
}

# afterok:<id>,... from "type:id" pairs whose id is not empty
dep() {
    local out="" p
    for p in "$@"; do
        [[ -n "${p#*:}" ]] && out+="${out:+,}$p"
    done
    echo "$out"
}

if ((SMOKE)); then
    sub smoke 00:30:00 00_smoke.slurm "" > /dev/null
    echo "smoke job submitted; wait for 'SMOKE OK' at the end of $LOGDIR/smoke-<jobid>.out"
    exit 0
fi

# one array task per baseline; reading the list needs python, so load the module on the login node too
setup_python > /dev/null 2>&1 || true
n_bl=$(baseline_tasks | wc -l | tr -d ' ')
[[ "$n_bl" =~ ^[0-9]+$ ]] || die "could not read the baseline list from $CONFIG (does 'python -c \"import calonsf\"' work?)"

J="" T="" S="" B="" D=""
if wanted jepa; then
    J=$(sub jepa "$TIME_JEPA" 01_jepa.slurm "")
fi
if wanted tune && [[ "${TUNE:-0}" == 1 ]]; then
    T=$(sub tune "$TIME_TUNE" 01b_tune.slurm "$(dep "afterok:$J")")
fi
if wanted sr; then
    S=$(sub sr "$TIME_SR" 02_sr.slurm "$(dep "afterok:$J" "afterok:$T")")
fi
if wanted baselines && ((n_bl)); then
    # baselines do not need the JEPA encoder, so they start straight away and run in parallel
    B=$(sub baselines "$TIME_BASELINE" 03_baselines.slurm "" "0-$((n_bl - 1))")
fi
if wanted downstream; then
    # the attentive probe sits on the JEPA encoder
    D=$(sub downstream "$TIME_DOWNSTREAM" 04_downstream.slurm "$(dep "afterok:$J")")
fi
E=""
if wanted eval; then
    E=$(sub eval "$TIME_EVAL" 05_eval.slurm "$(dep "afterok:$S" "afterok:$B" "afterok:$D")")
fi
if wanted sweep; then
    # after the evaluation (even a failed one), so the sweep table lands in report.md and the pack is complete
    sub sweep "$TIME_SWEEP" 06_sweep.slurm "$(dep "afterok:$S" "afterany:$E")" > /dev/null
fi

echo
if ((DRY)); then
    echo "dry run: nothing submitted (the commands above would be)"
else
    echo "submitted. Follow with:  bash slurm/status.sh $EXPERIMENT   (job list: $LOGDIR/jobs.tsv)"
    echo "squeue -u \$USER -o '%.18i %.40j %.10T %.10M %.20R'"
fi
