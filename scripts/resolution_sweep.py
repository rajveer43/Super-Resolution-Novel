"""Resolution sweep: one model, outputs from native up to native * 2^k (e.g. QG 128 -> 2048).
Measures time, memory, exact energy conservation, consistency when pooled back to native, and the
drift of physics observables. Above-native outputs have no ground truth.

    python scripts/resolution_sweep.py --config configs/experiments/qg_synthetic.yaml
"""
import os

import numpy as np
import torch

from calonsf.data.degrade import make_loader
from calonsf.data.ops import sum_pool
from calonsf.eval import plots
from calonsf.eval.plots import _view
from calonsf.eval.report import write_report
from calonsf.physics.observables import STABLE_OBS, observables
from calonsf.train.common import load_calonsf, parse, setup
from calonsf.utils.misc import Timer, device_memory_mb, free_cache, save_json, sync, to_device

args = parse(__doc__)
cfg, base, out, log, device, geom, reader, splits = setup(args, "eval")
model = load_calonsf(cfg, geom, base, device)
factor = int(cfg.eval.lr_factors[0])
n = int(cfg.eval.get("sweep_events", 4))
dl = make_loader(cfg, reader, splits["test"][:n], geom, False, factors=[factor], batch_size=1, shuffle=False)
fig_dir = plots.ensure(os.path.join(base, "eval", "sweep"))
rows = []
native_obs = None
for extra in cfg.eval.sweep_extra_steps:
    extra = int(extra)
    ts, mems, errs, cons, drift = [], [], [], [], []
    shape = None
    for i, batch in enumerate(dl):
        b = to_device(batch, device)
        free_cache(device)
        sync(device)
        with Timer() as t:
            o = model.generate(b, extra_steps=extra, ensemble=1, nfe=int(cfg.eval.nfe), solver=cfg.eval.solver,
                               return_scales=extra > 0)
            sync(device)
        ts.append(t.elapsed)
        mems.append(device_memory_mb(device))
        g = o["grids"]
        shape = {k: tuple(v.shape[-2:]) for k, v in g.items()}
        e_out = sum(float(v.detach().cpu().double().sum()) for v in g.values())
        e_in = sum(float(v.detach().cpu().double().sum()) for v in o["lr_init"].values())
        errs.append(abs(e_out - e_in) / max(e_in, 1e-12))
        if extra > 0:
            nat = o["scales"][o["ladder"].native_index]
            f = 2 ** extra
            pooled = {k: sum_pool(v, f if geom.group(k).sr_axes[0] else 1, f if geom.group(k).sr_axes[1] else 1)
                      for k, v in g.items()}
            cons.append(max(float((pooled[k] - nat[k]).abs().max() / nat[k].abs().sum().clamp(min=1e-12))
                            for k in g))
        ob = observables({k: v.cpu() for k, v in g.items()}, geom, {"e_inc": batch["e_inc"].numpy()})
        if extra == 0:
            native_obs = native_obs or {}
            native_obs[i] = ob
        elif native_obs and i in native_obs:
            ks = [k for k in STABLE_OBS[geom.name] if k in ob]
            drift.append(float(np.mean([np.abs(ob[k][0] - native_obs[i][k][0]).mean() /
                                        (np.abs(native_obs[i][k][0]).mean() + 1e-12) for k in ks])) if ks else float("nan"))
        if i == 0:
            img = _view(geom, {k: v.cpu().numpy() for k, v in g.items()}, 0)
            plots.save_image(os.path.join(fig_dir, f"event0_extra{extra}.png"), img,
                             f"output {'x'.join(map(str, img.shape))}")
    rows.append({"extra_steps": extra, "shape": str(shape), "time_s": float(np.mean(ts)),
                 "memory_mb": float(np.nanmax(mems)) if mems else float("nan"),
                 "energy_err": float(np.max(errs)), "pooled_consistency": float(np.max(cons)) if cons else 0.0,
                 "obs_drift": float(np.mean(drift)) if drift else None})
    log.info(f"[sweep] {rows[-1]}")
    if device.type == "mps":
        torch.mps.empty_cache()
save_json({"factor": factor, "rows": rows}, os.path.join(base, "eval", "sweep.json"))
if os.path.exists(os.path.join(base, "eval", "metrics.json")):
    write_report(base, synthetic="synthetic" in str(cfg.data.files))
