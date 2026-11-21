#!/bin/bash
# Download the datasets from their official hosts into $DATA_ROOT/real (see scripts/download_data.py).
#
#   bash slurm/download_data.sh            CaloChallenge Dataset 2 (~2.7 GB), right here on the login node
#   bash slurm/download_data.sh --submit   as a batch job on the data-transfer QOS (xfer, no GPU hours)
#
# Re-running resumes unfinished downloads and skips verified files.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings

submit=0
for a in "$@"; do
    case "$a" in
        --submit) submit=1 ;;
        *) die "unknown argument '$a' (only --submit)" ;;
    esac
done

if ((submit)); then
    [[ "$ACCOUNT" == mXXXX* ]] && die "set ACCOUNT in slurm/settings.sh"
    mkdir -p "$ROOT/logs"
    sbatch -A "$ACCOUNT" -q xfer -t 12:00:00 -J calonsf_download -o "$ROOT/logs/download-%j.out" \
        --wrap "bash $REPO/slurm/download_data.sh"
    echo "log: $ROOT/logs/download-<jobid>.out"
    exit 0
fi

setup_python
args=(--root "$DATA_ROOT/real")
python "$REPO/scripts/download_data.py" "${args[@]}"
