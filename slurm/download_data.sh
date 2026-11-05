#!/bin/bash
# Download the datasets from their official hosts into $DATA_ROOT/real (see scripts/download_data.py).
#
#   bash slurm/download_data.sh                    all three (~7 GB), right here on the login node
#   bash slurm/download_data.sh calogan qg         only some: calogan | calochallenge | qg
#   bash slurm/download_data.sh --submit [names]   as a batch job on the data-transfer QOS (xfer, no GPU hours)
#
# Re-running resumes unfinished downloads and skips verified files. The QG parquet files are
# converted to quark-gluon.h5 at the end.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings

submit=0
names=()
for a in "$@"; do
    case "$a" in
        --submit) submit=1 ;;
        calogan | calochallenge | qg) names+=("$a") ;;
        *) die "unknown argument '$a' (calogan | calochallenge | qg | --submit)" ;;
    esac
done

if ((submit)); then
    [[ "$ACCOUNT" == mXXXX* ]] && die "set ACCOUNT in slurm/settings.sh"
    mkdir -p "$ROOT/logs"
    sbatch -A "$ACCOUNT" -q xfer -t 12:00:00 -J calonsf_download -o "$ROOT/logs/download-%j.out" \
        --wrap "bash $REPO/slurm/download_data.sh ${names[*]}"
    echo "log: $ROOT/logs/download-<jobid>.out"
    exit 0
fi

setup_python
args=(--root "$DATA_ROOT/real")
for n in ${names[@]+"${names[@]}"}; do args+=(--dataset "$n"); done
python -c "import pyarrow, h5py" 2> /dev/null || args+=(--no-convert)   # run setup_env.sh to get pyarrow
python "$REPO/scripts/download_data.py" "${args[@]}"
