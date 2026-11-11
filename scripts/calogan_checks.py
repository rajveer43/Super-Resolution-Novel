"""Extra CaloGAN checks on a finished run (inference only, no training):

  1. Truth-vs-predicted 2D histograms with binned bias / RMSE, for all events and per particle.
  2. Why the classifier test gives AUC ~ 0.99: the same classifier on separate feature sets
     (observables only, cell energies only, hit/no-hit mask only, cells with tiny deposits removed),
     plus counts of near-zero cells in truth vs each method.
  3. Super-resolution beyond native resolution, shown per particle (e+, gamma, pi+).

    python scripts/calogan_checks.py --config <run>/eval/config.yaml --run-dir <run> --out <dir> --n-events 3000
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from calonsf.data.degrade import make_loader
from calonsf.data.geometry import make_geometry
from calonsf.data.readers import make_reader, make_splits
from calonsf.downstream.tasks import classifier_test
from calonsf.eval import plots
from calonsf.eval.evaluate import SCATTER_OBS, collect, load_baselines
from calonsf.physics import stats
from calonsf.physics.observables import KEY_OBS, observables
from calonsf.train.common import load_calonsf
from calonsf.utils.config import load_config
from calonsf.utils.misc import get_device, seed_everything, to_device

PARTICLES = ["eplus", "gamma", "piplus"]
P_LABEL = {"eplus": "e+", "gamma": "γ", "piplus": "π+"}

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--config", required=True)
p.add_argument("--run-dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n-events", type=int, default=3000)
p.add_argument("--factors", type=int, nargs="+", default=[2, 4])
p.add_argument("--baselines", nargs="+", default=["nearest", "ddpm"])
p.add_argument("--tiny", type=float, default=0.1, help="MeV; cells below this count as near-zero")
p.add_argument("--extra-steps", type=int, default=3)
p.add_argument("--per-particle", type=int, default=2, help="events per particle for the beyond-native figure")
args = p.parse_args()

cfg = load_config(args.config, [f"eval.n_events={args.n_events}", "eval.ensemble=1", "data.num_workers=0",
                                 f"eval.baselines=[{','.join(args.baselines)}]", "device=auto"])
seed_everything(int(cfg.seed))
device = get_device(cfg.device)
geom = make_geometry(cfg)
reader = make_reader(cfg)
test = make_splits(cfg, reader)["test"]
model = load_calonsf(cfg, geom, args.run_dir, device)
os.makedirs(args.out, exist_ok=True)
print(f"device={device}, test events used={min(args.n_events, len(test))}")


def flat_cells(grids):
    return np.concatenate([np.asarray(v).reshape(len(v), -1) for _, v in sorted(grids.items())], 1)


def particle_of(i):
    return PARTICLES[int(np.searchsorted(reader.cum, i, side="right") - 1)]


summary = {}
tvp_keys = list(dict.fromkeys(KEY_OBS["calogan"] + SCATTER_OBS["calogan"]))
for factor in args.factors:
    bl = load_baselines(cfg, geom, args.run_dir, factor, device)
    data, meta, _ = collect(cfg, geom, reader, test, factor, model, bl, device)
    methods = [m for m in ["calonsf", *args.baselines] if m in data]
    gt = data["gt"]
    obs_gt = observables(gt, geom, meta)
    obs = {m: observables(data[m], geom, meta) for m in methods}
    fdir = plots.ensure(os.path.join(args.out, f"x{factor}", "truth_vs_pred"))
    plots.plot_truth_vs_pred(fdir, obs_gt, obs, tvp_keys, methods)
    part = np.asarray(meta["particle"])
    for pi, pn in enumerate(PARTICLES):
        mk = part == pi
        if mk.sum() < 50:
            continue
        sub = lambda o: {k: np.asarray(v)[mk] for k, v in o.items()}  # noqa: E731
        plots.plot_truth_vs_pred(plots.ensure(os.path.join(fdir, pn)), sub(obs_gt), {m: sub(obs[m]) for m in methods},
                                 tvp_keys, methods, suffix=f"_{pn}")

    # ---- classifier test broken down by feature set
    cg = flat_cells(gt)
    hlf_gt = stats.feature_matrix(obs_gt)
    R = {"n_events": int(len(part)), "tiny_MeV": args.tiny, "methods": {}}
    for m in methods:
        cp = flat_cells(data[m])
        hlf = stats.feature_matrix(obs[m])
        lg, lp = np.log1p(cg / geom.energy_scale), np.log1p(np.clip(cp, 0, None) / geom.energy_scale)
        cut = lambda c: np.log1p(np.where(c >= args.tiny, c, 0) / geom.energy_scale)  # noqa: E731
        seed = int(cfg.seed)
        r = {
            "auc_full (as in report)": classifier_test(np.concatenate([hlf_gt, lg], 1), np.concatenate([hlf, lp], 1), seed),
            "auc_observables_only": classifier_test(hlf_gt, hlf, seed),
            "auc_cells_only": classifier_test(lg, lp, seed),
            "auc_hit_mask_only": classifier_test((cg > 0).astype(np.float32), (cp > 0).astype(np.float32), seed),
            f"auc_cells_below_{args.tiny}MeV_zeroed": classifier_test(cut(cg), cut(cp), seed),
            "hits_per_event": float((cp > 0).sum(1).mean()),
            "truth_hits_per_event": float((cg > 0).sum(1).mean()),
            f"tiny_cells_per_event (0<E<{args.tiny})": float(((cp > 0) & (cp < args.tiny)).sum(1).mean()),
            f"truth_tiny_cells_per_event (0<E<{args.tiny})": float(((cg > 0) & (cg < args.tiny)).sum(1).mean()),
            "min_nonzero_MeV": float(cp[cp > 0].min()) if (cp > 0).any() else None,
            "truth_min_nonzero_MeV": float(cg[cg > 0].min()),
            "negative_cells": int((cp < 0).sum()),
        }
        R["methods"][m] = r
        print(f"x{factor} {m}: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    summary[f"x{factor}"] = R

    # near-zero spectrum: where do the smallest deposits sit?
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    fig.patch.set_facecolor(plots.SURFACE)
    plots._style(ax)
    edges = np.logspace(-4, 4, 81)
    for name, c in [("gt", cg)] + [(m, flat_cells(data[m])) for m in methods]:
        v = c[c > 0]
        ax.stairs(np.histogram(v, edges)[0] / len(c), edges, color=plots.color(name), lw=1.5,
                  label=plots.LABEL.get(name, name))
    ax.axvline(args.tiny, color=plots.INK2, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cell energy [MeV]", fontsize=8, color=plots.INK2)
    ax.set_ylabel("cells per event", fontsize=8, color=plots.INK2)
    ax.set_title(f"x{factor}: spectrum of non-zero cells (low end = near-zero hits)", fontsize=9, color=plots.INK)
    ax.legend(frameon=False, fontsize=7)
    plots._save(fig, os.path.join(args.out, f"x{factor}", "near_zero_spectrum.png"))

with open(os.path.join(args.out, "classifier_breakdown.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ---- beyond native, per particle (x2 input, layer 1 shown)
factor = args.factors[0]
picked = []
for pn in PARTICLES:
    picked += [int(i) for i in test if particle_of(i) == pn][: args.per_particle]
dl = make_loader(cfg, reader, np.array(picked), geom, False, factors=[factor], batch_size=1, shuffle=False)
cols = ["LR input", "truth 12×12"] + [f"CaloNSF {12 * 2 ** e}×{12 * 2 ** e}" for e in range(args.extra_steps + 1)]
fig, axes = plt.subplots(len(picked), len(cols), figsize=(2.3 * len(cols), 2.4 * len(picked)), squeeze=False)
fig.patch.set_facecolor(plots.SURFACE)
for r, batch in enumerate(dl):
    b = to_device(batch, device)
    imgs = [np.asarray(batch["lr"]["layer_1"][0, 0]), np.asarray(batch["hr"]["layer_1"][0, 0])]
    for e in range(args.extra_steps + 1):
        o = model.generate(b, extra_steps=e, ensemble=1, nfe=int(cfg.eval.nfe), solver=cfg.eval.solver)
        imgs.append(o["grids"]["layer_1"][0, 0].detach().float().cpu().numpy())
    vmax = max(imgs[1].max(), 1e-9)
    for c, (img, title) in enumerate(zip(imgs, cols)):
        ax = axes[r, c]
        ax.set_xticks([])
        ax.set_yticks([])
        # each finer grid spreads the same energy over 4x more cells, so scale the colour range per image
        scale = (img.size / imgs[1].size) if c >= 2 else (img.size / imgs[1].size if c == 0 else 1)
        vm = vmax / max(scale, 1e-9)
        ax.imshow(np.clip(img, vm * 1e-4, None), cmap=plots.SEQ, norm=LogNorm(vm * 1e-4, vm), interpolation="nearest")
        if r == 0:
            ax.set_title(title, fontsize=9, color=plots.INK)
        if c == 0:
            pn = particle_of(picked[r])
            ax.set_ylabel(f"{P_LABEL[pn]}  {float(batch['e_inc'][0]):.0f} GeV", fontsize=9, color=plots.INK)
plots._save(fig, os.path.join(args.out, f"beyond_native_per_particle_x{factor}.png"))
print(f"wrote {args.out}")
