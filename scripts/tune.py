"""Optuna study for the CaloNSF training hyperparameters (at most 15 trials).

    python scripts/tune.py --config configs/experiments/calogan.yaml [-o overrides...]

Each trial trains CaloNSF briefly (tune.epochs x tune.steps_per_epoch) on the JEPA-pretrained encoder
(run pretrain_jepa.py first) and is scored on tune.val_events VALIDATION events: the mean W1/sigma of the
dataset's key physics observables (lower is better). Validation loss is not used as the score because
the loss weights are searched too, and they change the size of the loss.

Trial 0 is the current config, so the study always shows whether tuning helped. The study lives in
runs/<exp>/tune/study.db: re-running continues it, and trials already finished count toward the cap.
Outputs in runs/<exp>/tune/:
    best_overrides.txt   one key=value per line; slurm/ passes these to the sr, eval and sweep stages
    trials.csv           every trial with its parameters and score
"""
import os
import shutil

import numpy as np
import optuna
import torch

from calonsf.data.degrade import make_loader
from calonsf.data.geometry import make_geometry
from calonsf.eval.evaluate import collect
from calonsf.physics import stats
from calonsf.physics.observables import KEY_OBS, observables
from calonsf.train.common import parse, setup
from calonsf.train.sr import build_sr
from calonsf.train.trainer import fit
from calonsf.utils.config import load_config
from calonsf.utils.misc import free_cache, seed_everything

MAX_TRIALS = 15

# The encoder's width and depth are fixed by the JEPA checkpoint, so only decoder size, optimisation and
# loss weights are searched. Each entry: config key -> (kind, choices or (low, high)).
SPACE = {
    "train.lr": ("log", (1e-4, 1e-3)),
    "train.warmup_frac": ("float", (0.02, 0.1)),
    "train.encoder_lr_mult": ("log", (0.05, 0.5)),
    "data.batch_size": ("cat", [16, 32, 64]),
    "model.dec_layers": ("cat", [2, 4, 6]),
    "loss.occ": ("log", (0.3, 3.0)),
    "loss.profile": ("log", (0.1, 2.0)),
    "loss.jepa": ("log", (0.01, 0.3)),
}

args = parse(__doc__)
cfg0, base, out, log, device, geom, reader, splits = setup(args, "tune")
tc = cfg0.tune
n_target = min(int(tc.get("trials", MAX_TRIALS)), MAX_TRIALS)
enc = os.path.join(base, "jepa", "encoder.pt")
if not os.path.exists(enc):
    raise SystemExit(f"no JEPA encoder at {enc}: run scripts/pretrain_jepa.py with the same config first")
factor = int(cfg0.eval.lr_factors[0])
val_idx = splits["val"][: int(tc.val_events)]
key_obs = KEY_OBS[geom.name]


def fmt(v):
    return repr(v) if isinstance(v, float) else str(v)


def sample(trial):
    ov = {}
    for k, (kind, r) in SPACE.items():
        if kind == "log":
            ov[k] = trial.suggest_float(k, *r, log=True)
        elif kind == "float":
            ov[k] = trial.suggest_float(k, *r)
        else:
            ov[k] = trial.suggest_categorical(k, r)
    return ov


def objective(trial):
    ov = sample(trial)
    cfg = load_config(args.config, [*args.override, *(f"{k}={fmt(v)}" for k, v in ov.items()),
                                    f"train.epochs={int(tc.epochs)}",
                                    f"train.max_steps_per_epoch={int(tc.steps_per_epoch)}",
                                    "train.pretrained_encoder=auto",
                                    f"eval.n_events={len(val_idx)}", "eval.ensemble=1", f"eval.nfe={int(tc.nfe)}"])
    seed_everything(int(cfg.seed))
    log.info(f"[tune] trial {trial.number}: " + " ".join(f"{k}={fmt(v)}" for k, v in ov.items()))
    tdir = os.path.join(out, f"trial_{trial.number:02d}")
    shutil.rmtree(tdir, ignore_errors=True)  # a trial restarted after a crash starts clean
    g = make_geometry(cfg)
    model, groups, on_epoch = build_sr(cfg, g, base, device, log)
    t = cfg.train
    try:
        fit(model, model.loss, make_loader(cfg, reader, splits["train"], g, True),
            make_loader(cfg, reader, splits["val"], g, False), epochs=int(t.epochs), lr=float(t.lr),
            weight_decay=float(t.weight_decay), device=device, out_dir=tdir, schedule=t.get("schedule", "cosine"),
            warmup_frac=float(t.warmup_frac), grad_clip=float(t.grad_clip),
            max_steps_per_epoch=int(t.max_steps_per_epoch), param_groups=groups, on_epoch=on_epoch,
            tag="tune", log_every=int(t.log_every))
        model.eval()
        data, meta, _ = collect(cfg, g, reader, val_idx, factor, model, {}, device)
        cmp = stats.compare(observables(data["gt"], g, meta), observables(data["calonsf"], g, meta))
        per = {k: float(cmp[k]["w1_norm"]) for k in key_obs if k in cmp}
        score = float(np.mean(list(per.values())))
    finally:
        del model
        free_cache(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        shutil.rmtree(tdir, ignore_errors=True)  # checkpoints of a trial are not needed afterwards
    for k, v in per.items():
        trial.set_user_attr(f"w1_{k}", v)
    log.info(f"[tune] trial {trial.number}: score (mean key-obs W1/sigma) = {score:.4f}")
    if not np.isfinite(score):
        raise optuna.TrialPruned("non-finite score")
    return score


storage = optuna.storages.RDBStorage(
    f"sqlite:///{os.path.join(out, 'study.db')}", heartbeat_interval=60, grace_period=180,
    failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=0))  # a killed job's trial -> FAIL
study = optuna.create_study(study_name="calonsf_sr", storage=storage, direction="minimize", load_if_exists=True,
                            sampler=optuna.samplers.TPESampler(seed=int(cfg0.seed),
                                                               n_startup_trials=int(tc.startup_trials)))
if not study.trials:
    study.enqueue_trial({k: cfg0.get_path(k) for k in SPACE})  # trial 0 = the current defaults
finished = sum(t.state.is_finished() for t in study.trials)
remaining = max(0, n_target - finished)
log.info(f"[tune] {finished} trials finished, running {remaining} more (cap {n_target}); scored on "
         f"{len(val_idx)} validation events at x{factor}")
if remaining:
    # out-of-memory or other runtime errors fail that trial only; they still count toward the cap
    study.optimize(objective, n_trials=remaining, catch=(RuntimeError, ValueError))

study.trials_dataframe().to_csv(os.path.join(out, "trials.csv"), index=False)
done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if not done:
    raise SystemExit("no trial completed; see the log above")
best = study.best_trial
with open(os.path.join(out, "best_overrides.txt"), "w") as f:
    for k, v in best.params.items():
        f.write(f"{k}={fmt(v)}\n")
default = next((t for t in done if t.number == 0), None)
log.info(f"[tune] best trial {best.number}: score {best.value:.4f}"
         + (f" (default config: {default.value:.4f})" if default else ""))
log.info("[tune] best overrides: " + " ".join(f"{k}={fmt(v)}" for k, v in best.params.items()))
log.info(f"[tune] wrote {out}/best_overrides.txt and {out}/trials.csv")
