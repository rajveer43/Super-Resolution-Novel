# Settings for running CaloNSF on Perlmutter. This is the ONLY file to edit.
# submit_all.sh freezes these values into $ROOT/logs/<experiment>/run.env at submit time, so editing this
# file later does not change jobs that are already queued.

# ---- account and resources ----------------------------------------------------------------------
ACCOUNT=mXXXX                 # NERSC project with GPU hours (some GPU allocations are named mXXXX_g)
QOS=shared                    # shared (1 GPU, charged per GPU: recommended) | regular (whole node) | preempt
GPU_MEM=40                    # 40 | 80 (80 GB A100s are fewer, so they may queue longer)

# ---- what to run ---------------------------------------------------------------------------------
CONFIG_NAME=calogan           # a file in configs/experiments/ without .yaml:
                              #   calogan | qg | calochallenge | qg_noisy | calochallenge_noisy
RUN_TAG=full                  # what this experiment tests, in words joined by '-' (letters, digits, '-', '.')
RUN_DATE=2026-11-05           # empty = today (Asia/Kolkata) -> new experiment; a fixed date resumes that one
                              # -> experiment <RUN_DATE>_<CONFIG_NAME>_<RUN_TAG>, e.g. 2026-11-05_calogan_full
TUNE=1                        # 1 = Optuna study (max 15 short trials, scripts/tune.py) after JEPA; SR, eval and
                              # sweep then use the best hyperparameters
# Config overrides for every stage, space separated key=value (YAML values, no spaces inside a value).
# The presets in configs/ are sized for a laptop; these use the A100: all events, bigger batches, longer training.
EXTRA_SET="data.max_events=null data.batch_size=32 data.num_workers=8 train.epochs=30 jepa.epochs=15 baseline.epochs=20 downstream.epochs=15 downstream.probe_epochs=10 eval.n_events=10000 eval.lr_factors=[2,4]"

# ---- where things go -----------------------------------------------------------------------------
ROOT="$SCRATCH/calonsf"       # runs/ logs/ live here ($SCRATCH: fast, purged after 8 weeks unused)
DATA_ROOT="$ROOT/data"        # same layout as the laptop: real/calogan/{eplus,gamma,piplus}.hdf5,
                              # real/qg/quark-gluon.h5, real/calochallenge_v2/dataset_2_{1,2}.hdf5
CFS_COPY=                     # optional, e.g. /global/cfs/cdirs/mXXXX/calonsf: results are copied here at the end

# ---- wall-time limits (HH:MM:SS, max 48:00:00). Estimates: raise them if a job times out -----------
# Training resumes from last.pt, so a timed-out stage can simply be re-submitted with --from <stage>.
TIME_JEPA=06:00:00
TIME_TUNE=16:00:00            # whole study; it continues after a timeout (--from tune)
TIME_SR=24:00:00
TIME_BASELINE=08:00:00        # per baseline (one array task each)
TIME_DOWNSTREAM=08:00:00       # CaloChallenge needs ~5 h (25 epochs x ~12 min); CaloGAN much less
TIME_EVAL=06:00:00
TIME_SWEEP=02:00:00
