# Shared by every job and by the submit / status scripts. Not run directly.
# Jobs get RUN_ENV (the frozen settings written by submit_all.sh) through sbatch --export.

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- settings
# REPO and the settings: from the frozen run.env inside a job, from settings.sh otherwise.
load_settings() {
    if [[ -n "${RUN_ENV:-}" ]]; then
        # shellcheck disable=SC1090
        source "$RUN_ENV"
    else
        REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        # shellcheck source=settings.sh
        source "$REPO/slurm/settings.sh"
    fi
    [[ -n "${ROOT:-}" && "$ROOT" != "/calonsf" ]] || die "ROOT is empty: is \$SCRATCH set? (see settings.sh)"
    CONFIG="configs/experiments/${CONFIG_NAME}.yaml"
    [[ -f "$REPO/$CONFIG" ]] || die "no config $REPO/$CONFIG (CONFIG_NAME in settings.sh)"
}

# Fix the experiment name once (submit_all.sh): RUN_DATE (today if empty), RUN_TAG (+ suffix) and
# EXPERIMENT = <date>_<config>_<tag>. It is also the exp_name the Python code writes under runs/.
resolve_experiment() {
    RUN_DATE="${RUN_DATE:-$(TZ=Asia/Kolkata date +%Y-%m-%d)}"
    RUN_TAG="${RUN_TAG}${1:-}"
    [[ "$RUN_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "RUN_DATE must look like 2026-11-05, not '$RUN_DATE'"
    [[ "$RUN_TAG" =~ ^[A-Za-z0-9.-]+$ ]] || die "RUN_TAG may only use letters, digits, '-' and '.': '$RUN_TAG'"
    EXPERIMENT="${RUN_DATE}_${CONFIG_NAME}_${RUN_TAG}"
}

experiment_dir() { echo "$ROOT/runs/$EXPERIMENT"; }
log_dir() { echo "$ROOT/logs/$EXPERIMENT"; }

# For scripts run by hand after submission: load the frozen settings of one experiment.
#   load_experiment ""        the most recently submitted one
#   load_experiment <name>    that one, e.g. 2026-11-05_calogan_full
load_experiment() {
    local f
    if [[ -n "${1:-}" ]]; then
        f="$ROOT/logs/$1/run.env"
    else
        # shellcheck disable=SC2012
        f="$(ls -t "$ROOT"/logs/*/run.env 2> /dev/null | head -1 || true)"
    fi
    [[ -f "$f" ]] || die "no submitted experiment found under $ROOT/logs${1:+ named $1}"
    RUN_ENV="$f"
    load_settings
}

# ---------------------------------------------------------------- finished-stage markers
# A training stage that already finished would start again from scratch if re-run, so every job
# writes a marker when it succeeds and skips itself when the marker is there. FORCE=1 re-runs it.
marker() { echo "$(experiment_dir)/.done_$1"; }
skip_if_done() {
    if [[ -f "$(marker "$1")" && "${FORCE:-0}" != 1 ]]; then
        echo "$1 already finished ($(marker "$1")); skipping. Re-run with FORCE=1 to redo it."
        exit 0
    fi
}
mark_done() { mkdir -p "$(experiment_dir)"; date '+%F %T' > "$(marker "$1")"; }

# ---------------------------------------------------------------- environment
setup_python() {
    if ! type module > /dev/null 2>&1 && [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/lmod/lmod/init/bash
    fi
    if type module > /dev/null 2>&1; then
        set +u    # Lmod's shell code reads unset variables
        module load pytorch
        set -u
    fi
    # Nothing goes to $HOME (small quota): extra pip packages, pip/matplotlib/torch caches live under $ROOT.
    # Set after "module load", which points PYTHONUSERBASE at $HOME/.local.
    export PYTHONUSERBASE="$ROOT/python"
    export PATH="$PYTHONUSERBASE/bin:$PATH"
    export PIP_CACHE_DIR="$ROOT/cache/pip"
    export XDG_CACHE_HOME="$ROOT/cache"
    export MPLCONFIGDIR="$ROOT/cache/matplotlib"
    export TORCH_HOME="$ROOT/cache/torch"
    mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$TORCH_HOME"
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1
    export MPLBACKEND=Agg
    export HDF5_USE_FILE_LOCKING=FALSE    # $SCRATCH (Lustre) and several readers of one file
    local cpus="${SLURM_CPUS_PER_TASK:-4}"
    PHYS_CORES=$((cpus > 1 ? cpus / 2 : 1))   # Perlmutter: 2 hardware threads per core
    export OMP_NUM_THREADS="$PHYS_CORES"
}

job_header() {
    echo "==================================================================="
    echo "stage      : $1"
    echo "experiment : $(experiment_dir)"
    echo "job        : ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:--} on $(hostname)"
    echo "started    : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "code       : $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2> /dev/null) $(git -C "$REPO" rev-parse --short HEAD 2> /dev/null)"
    echo "settings   : config=$CONFIG qos=$QOS extra='$EXTRA_SET'"
    python -c 'import sys, torch; print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.cuda.is_available())'
    if command -v nvidia-smi > /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2> /dev/null || echo "(no GPU on this node)"
    fi
    echo "==================================================================="
}

# run_py <script> [script args...]
# Runs scripts/<script> from the repo (the configs' data/ paths resolve through the data link made by
# setup_env.sh), writing to $ROOT/runs/<experiment>. One process per job, started in the batch step
# so it gets all the job's CPUs and its GPU.
run_py() {
    local script="$1"
    shift
    local -a extra=()
    [[ -n "$EXTRA_SET" ]] && read -r -a extra <<< "$EXTRA_SET"
    # stages that build CaloNSF (sr, eval, sweep) set USE_TUNED=1: the Optuna result, if any, wins over EXTRA_SET
    local tuned="$(experiment_dir)/tune/best_overrides.txt"
    if [[ "${USE_TUNED:-0}" == 1 && -f "$tuned" ]]; then
        local line
        while read -r line; do [[ -n "$line" ]] && extra+=("$line"); done < "$tuned"
        echo "using tuned hyperparameters from $tuned"
    fi
    local -a cmd=(python "scripts/$script" --config "$CONFIG" "$@"
                  -o "out_dir=$ROOT/runs" "exp_name=$EXPERIMENT" ${extra[@]+"${extra[@]}"})
    cd "$REPO"
    [[ -e data ]] || die "$REPO/data is missing: run 'bash slurm/setup_env.sh' once on a login node"
    echo "+ ${cmd[*]}"
    local t0=$SECONDS
    "${cmd[@]}"
    echo "-- $script finished in $(((SECONDS - t0) / 60)) min"
}

# Baseline training tasks, one line "<name> <factor>" per array task: every baseline the evaluation
# compares against (eval.baselines) at every evaluated LR factor (eval.lr_factors).
baseline_tasks() {
    local -a extra=()
    [[ -n "$EXTRA_SET" ]] && read -r -a extra <<< "$EXTRA_SET"
    (cd "$REPO" && python -c "import sys; from calonsf.utils.config import load_config; \
c = load_config(sys.argv[1], sys.argv[2:]); \
[print(n, f) for f in c.eval.lr_factors for n in c.eval.baselines]" "$CONFIG" ${extra[@]+"${extra[@]}"})
}
baseline_task() { baseline_tasks | sed -n "$(($1 + 1))p"; }
