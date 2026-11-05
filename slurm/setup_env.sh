#!/bin/bash
# One-time setup, on a Perlmutter login node:  bash slurm/setup_env.sh
# Loads the NERSC PyTorch module, installs the few missing packages into the module's user site
# (kept between sessions), links data/ to $DATA_ROOT and checks that the code imports.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python

echo "pytorch module: $(module list 2>&1 | grep -o 'pytorch/[^ ]*' || echo '?')"
python -c 'import torch; print("torch", torch.__version__)'

missing=()
for spec in numpy scipy h5py yaml:pyyaml matplotlib sklearn:scikit-learn skimage:scikit-image tqdm torchdiffeq pyarrow; do
    mod="${spec%%:*}"
    pkg="${spec#*:}"
    python -c "import $mod" 2> /dev/null || missing+=("$pkg")
done
if ((${#missing[@]})); then
    echo "installing into the user site of this module: ${missing[*]}"
    pip install --user "${missing[@]}"
fi
python -c "import calonsf.models.calonsf, calonsf.eval.evaluate, calonsf.train.trainer; print('calonsf imports ok')"

# The configs name files as data/real/..., relative to the repo. Point data/ at $DATA_ROOT on $SCRATCH.
mkdir -p "$DATA_ROOT/real/qg" "$DATA_ROOT/real/calogan" "$DATA_ROOT/real/calochallenge_v2" "$ROOT/runs" "$ROOT/logs"
if [[ -L "$REPO/data" ]]; then
    ln -sfn "$DATA_ROOT" "$REPO/data"
elif [[ -e "$REPO/data" ]]; then
    echo "note: $REPO/data is a real folder, so it is used as is (DATA_ROOT is ignored)"
else
    ln -s "$DATA_ROOT" "$REPO/data"
fi
echo "data: $REPO/data -> $(readlink -f "$REPO/data")"
for f in real/calogan/eplus.hdf5 real/calogan/gamma.hdf5 real/calogan/piplus.hdf5 real/qg/quark-gluon.h5 \
         real/calochallenge_v2/dataset_2_1.hdf5 real/calochallenge_v2/dataset_2_2.hdf5; do
    if [[ -f "$REPO/data/$f" ]]; then echo "  found   data/$f"; else echo "  missing data/$f"; fi
done
echo "results root: $ROOT  ($(df -h "$ROOT" | awk 'NR == 2 {print $4 " free"}'))"
[[ "$ACCOUNT" == mXXXX* ]] && echo "NEXT: set ACCOUNT in slurm/settings.sh (your projects: run 'iris' or check https://iris.nersc.gov)"
echo "setup done. Next: bash slurm/download_data.sh (if anything is missing above), then bash slurm/submit_all.sh --smoke"
