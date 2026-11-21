# Settings for running CaloNSF on Perlmutter. This is the ONLY file to edit.
# submit_all.sh freezes these values into $ROOT/logs/<experiment>/run.env at submit time, so editing this
# file later does not change jobs that are already queued.

# ---- account and resources ----------------------------------------------------------------------
ACCOUNT=mXXXX                 # NERSC project with GPU hours (some GPU allocations are named mXXXX_g)
QOS=shared                    # shared (1 GPU, charged per GPU: recommended) | regular (whole node) | preempt
GPU_MEM=40                    # 40 | 80 (80 GB A100s are fewer, so they may queue longer)

# ---- what to run ---------------------------------------------------------------------------------
CONFIG_NAME=calochallenge_large   # a file in configs/experiments/ without .yaml:
                              #   calochallenge_large (A100 preset) | calochallenge (laptop-sized)
RUN_TAG=full                  # what this experiment tests, in words joined by '-' (letters, digits, '-', '.')
RUN_DATE=2026-11-21           # fixed = every submit and restart (--from ...) of this run uses the same experiment folder;
                              # empty = today (Asia/Kolkata), i.e. a new experiment each day
                              # -> experiment <RUN_DATE>_<CONFIG_NAME>_<RUN_TAG>, e.g. 2026-11-05_calochallenge_large_full
# Config overrides for every stage, space separated key=value (YAML values, no spaces inside a value).
# calochallenge_large already holds the A100 settings (all events, bigger model, 24 JEPA / 20 SR epochs);
# the values below repeat the preset on purpose, so the run.env of the experiment records them. Fallbacks:
#   SR out of memory:  data.batch_size=8        eval too slow:  eval.classifier.seeds=3
EXTRA_SET="data.batch_size=16 jepa.batch_size=32 eval.classifier.seeds=10"

# ---- where things go -----------------------------------------------------------------------------
ROOT="$SCRATCH/calonsf"       # runs/ logs/ live here ($SCRATCH: fast, purged after 8 weeks unused)
DATA_ROOT="$ROOT/data"        # same layout as the laptop: real/calochallenge_v2/dataset_2_{1,2}.hdf5
CFS_COPY=                     # optional, e.g. /global/cfs/cdirs/mXXXX/calonsf: results are copied here at the end

# ---- wall-time limits (HH:MM:SS, max 48:00:00). Estimates: raise them if a job times out -----------
# Training resumes from last.pt, so a timed-out stage can simply be re-submitted with --from <stage>.
TIME_JEPA=24:00:00            # 24 epochs of the larger model (estimate from CPU timing: up to ~20 h); re-submit if it times out
TIME_SR=47:00:00
TIME_EVAL=24:00:00            # 3 methods x (3 classifiers x 10 seeds) on 100 000 showers: unmeasured, the 3D ResNet dominates.
                              # To shorten: EXTRA_SET="eval.classifier.seeds=3" (charged by used time, not by the limit)
