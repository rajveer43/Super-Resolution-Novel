"""Extra CaloChallenge (dataset 2) checks on a finished run (inference only, no training):

  1. Example showers: LR input, CaloNSF, DDPM, truth, in two views (transverse: angle x radius summed
     over depth; longitudinal: depth x radius summed over angle).
  2. Truth-vs-predicted 2D histograms with binned bias / RMSE, for all events and per incident-energy bin.
  3. Classifier test split by feature set (observables / cells / hit mask / near-zero cells removed),
     and the spectrum of non-zero cell energies.
  4. Super-resolution beyond native resolution for low / mid / high incident energy.

    python scripts/calochallenge_checks.py --config <run>/eval/config.yaml --run-dir <run> --out <dir>
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from calonsf.data.degrade import make_loader
from calonsf.data.geometry import lr_factors_for, make_geometry
from calonsf.data.readers import make_reader, make_splits
from calonsf.downstream.tasks import classifier_test
from calonsf.eval import plots
from calonsf.eval.evaluate import collect, load_baselines
from calonsf.physics import stats
from calonsf.physics.observables import observables
from calonsf.train.common import load_calonsf
from calonsf.utils.config import load_config
from calonsf.utils.misc import get_device, seed_everything, to_device

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--config", required=True)
p.add_argument("--run-dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n-events", type=int, default=2000)
p.add_argument("--factors", type=int, nargs="+", default=[2, 4])
p.add_argument("--baselines", nargs="+", default=["nearest", "ddpm"])
p.add_argument("--tiny", type=float, default=0.1, help="MeV; cells below this count as near-zero")
p.add_argument("--n-examples", type=int, default=4)
p.add_argument("--extra-steps", type=int, default=2)
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
E_BINS = [1e3, 1e4, 1e5, 1e6 + 1]  # MeV
E_NAMES = ["1-10GeV", "10-100GeV", "100-1000GeV"]


def scalars(o):
    """Scalar summaries of the per-layer CaloChallenge observables (energy-weighted over layers)."""
    El, Et = o["E_layer"], o["E_tot"]
    w = El / np.maximum(Et[:, None], 1e-12)
    z = np.arange(El.shape[1])
    r = np.arange(o["radial_profile"].shape[1])
    d = (w * z).sum(1)
    out = {"E_ratio_inc": o["E_ratio_inc"], "shower_depth": d,
           "depth_width": np.sqrt(np.clip((w * z ** 2).sum(1) - d ** 2, 0, None)),
           "width_x_mean": (w * o["width_x"]).sum(1), "width_y_mean": (w * o["width_y"]).sum(1),
           "radial_mean": (o["radial_profile"] * r).sum(1), "sparsity_mean": (w * o["sparsity"]).sum(1),
           "E_first_10_layers": w[:, :10].sum(1)}
    return out


def flat(g):
    return np.asarray(g["shower"]).reshape(len(g["shower"]), -1)


def transverse(x):  # (Z, A, R) -> (A, R)
    return x.sum(0)


def longitudinal(x):  # (Z, A, R) -> (Z, R)
    return x.sum(1)


def draw_polar(ax, img, vmin, vmax):
    A, R = img.shape
    th = np.linspace(0, 2 * np.pi, A + 1)
    rr = np.arange(R + 1)
    ax.pcolormesh(th, rr, np.clip(img, vmin, None).T, cmap=plots.SEQ, norm=LogNorm(vmin, vmax), shading="flat")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_color(plots.GRID)


def plot_examples(path, data, meta, methods, n):
    cols = ["nearest"] + [m for m in methods if m != "nearest"] + ["gt"]
    title = {"nearest": "LR input", "gt": "Truth"}
    fig = plt.figure(figsize=(2.4 * len(cols), 2.5 * 2 * n))
    fig.patch.set_facecolor(plots.SURFACE)
    # examples spread over incident energy (log-spaced quantiles of the test events)
    e = np.asarray(meta["e_inc"])
    order = np.argsort(e)
    rows = [int(order[int(q * (len(e) - 1))]) for q in np.linspace(0.05, 0.95, n)]
    for r, i in enumerate(rows):
        g = data["gt"]["shower"][i]
        zmax = int(g.sum((1, 2)).argmax())  # layer with the most energy in truth
        tv, lv = g[zmax], longitudinal(g)
        for c, m in enumerate(cols):
            x = data[m]["shower"][i]
            ax = fig.add_subplot(2 * n, len(cols), 2 * r * len(cols) + c + 1, projection="polar")
            draw_polar(ax, x[zmax], max(tv.max() * 1e-3, 1e-6), max(tv.max(), 1e-6))
            if r == 0:
                ax.set_title(title.get(m, plots.LABEL.get(m, m)), fontsize=9, color=plots.INK)
            if c == 0:
                ax.text(-0.3, 0.5, f"{e[i] / 1e3:.0f} GeV\nlayer {zmax}", transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=8, color=plots.INK)
            ax = fig.add_subplot(2 * n, len(cols), (2 * r + 1) * len(cols) + c + 1)
            ax.imshow(np.clip(longitudinal(x), lv.max() * 1e-3, None).T, origin="lower", cmap=plots.SEQ,
                      norm=LogNorm(max(lv.max() * 1e-3, 1e-6), max(lv.max(), 1e-6)), aspect="auto",
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel("radius ↑ / depth →", fontsize=7, color=plots.INK2)
    plots._save(fig, path)


summary = {}
for factor in args.factors:
    fac = lr_factors_for(geom, factor)
    bl = load_baselines(cfg, geom, args.run_dir, factor, device)
    data, meta, times = collect(cfg, geom, reader, test, factor, model, bl, device)
    methods = [m for m in ["calonsf", *args.baselines] if m in data]
    fdir = plots.ensure(os.path.join(args.out, f"x{factor}"))
    plot_examples(os.path.join(fdir, "examples.png"), data, meta, methods, args.n_examples)

    obs_gt = observables(data["gt"], geom, meta)
    obs = {m: observables(data[m], geom, meta) for m in methods}
    s_gt = scalars(obs_gt)
    s = {m: scalars(obs[m]) for m in methods}
    tdir = plots.ensure(os.path.join(fdir, "truth_vs_pred"))
    plots.plot_truth_vs_pred(tdir, s_gt, s, list(s_gt), methods)
    e = np.asarray(meta["e_inc"])
    for lo, hi, nm in zip(E_BINS[:-1], E_BINS[1:], E_NAMES):
        mk = (e >= lo) & (e < hi)
        if mk.sum() < 50:
            continue
        sub = lambda o: {k: v[mk] for k, v in o.items()}  # noqa: E731
        plots.plot_truth_vs_pred(plots.ensure(os.path.join(tdir, nm)), sub(s_gt), {m: sub(s[m]) for m in methods},
                                 list(s_gt), methods, suffix=f"_{nm}")
    # standard 1D histograms with ratio panels and per-layer profiles, as in the main eval
    plots.plot_hist_ratio(plots.ensure(os.path.join(fdir, "hist")), s_gt, s, methods)
    plots.plot_profiles(plots.ensure(os.path.join(fdir, "profile")), obs_gt, obs, methods)

    # classifier test by feature set + near-zero cells
    cg = flat(data["gt"])
    hlf_gt = stats.feature_matrix(obs_gt)
    R = {"lr_shape_factors": fac[geom.groups[0].name], "n_events": int(len(e)), "methods": {}}
    seed = int(cfg.seed)
    for m in methods:
        cp = np.clip(flat(data[m]), 0, None)
        hlf = stats.feature_matrix(obs[m])
        lg, lp = np.log1p(cg / geom.energy_scale), np.log1p(cp / geom.energy_scale)
        cut = lambda c: np.log1p(np.where(c >= args.tiny, c, 0) / geom.energy_scale)  # noqa: E731
        r = {"auc_full (as in report)": classifier_test(np.concatenate([hlf_gt, lg], 1), np.concatenate([hlf, lp], 1), seed),
             "auc_observables_only": classifier_test(hlf_gt, hlf, seed),
             "auc_cells_only": classifier_test(lg, lp, seed),
             "auc_hit_mask_only": classifier_test((cg > 0).astype(np.float32), (cp > 0).astype(np.float32), seed),
             f"auc_cells_below_{args.tiny}MeV_zeroed": classifier_test(cut(cg), cut(cp), seed),
             "hits_per_event": float((cp > 0).sum(1).mean()), "truth_hits_per_event": float((cg > 0).sum(1).mean()),
             f"tiny_cells_per_event (0<E<{args.tiny})": float(((cp > 0) & (cp < args.tiny)).sum(1).mean()),
             f"truth_tiny_cells_per_event (0<E<{args.tiny})": float(((cg > 0) & (cg < args.tiny)).sum(1).mean()),
             "energy_pred_over_truth": float(cp.sum() / cg.sum()),
             "time_per_event_ms": 1e3 * float(times.get(m, float("nan")))}
        for k in s_gt:
            d = s[m][k] - s_gt[k]
            r[f"{k}: bias / RMSE"] = [float(d.mean()), float(np.sqrt((d ** 2).mean()))]
        R["methods"][m] = r
        print(f"x{factor} {m}: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in r.items() if not k.endswith("RMSE")))
    summary[f"x{factor}"] = R

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    fig.patch.set_facecolor(plots.SURFACE)
    plots._style(ax)
    edges = np.logspace(-4, 5, 91)
    for name, c in [("gt", cg)] + [(m, np.clip(flat(data[m]), 0, None)) for m in methods]:
        ax.stairs(np.histogram(c[c > 0], edges)[0] / len(c), edges, color=plots.color(name), lw=1.5,
                  label=plots.LABEL.get(name, name))
    ax.axvline(args.tiny, color=plots.INK2, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cell energy [MeV]", fontsize=8, color=plots.INK2)
    ax.set_ylabel("cells per event", fontsize=8, color=plots.INK2)
    ax.set_title(f"x{factor}: spectrum of non-zero cells", fontsize=9, color=plots.INK)
    ax.legend(frameon=False, fontsize=7)
    plots._save(fig, os.path.join(fdir, "cell_spectrum.png"))

with open(os.path.join(args.out, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ---- beyond native: one event per incident-energy bin, transverse view (x2 input)
factor = args.factors[0]
e_all = np.array([reader.get(int(i))[1]["e_inc"] for i in test[:3000]])
picked = []
for lo, hi in zip(E_BINS[:-1], E_BINS[1:]):
    w = np.where((e_all >= lo) & (e_all < hi))[0]
    if len(w):
        picked.append(int(test[w[0]]))
dl = make_loader(cfg, reader, np.array(picked), geom, False, factors=[factor], batch_size=1, shuffle=False)
A, Rn = geom.groups[0].shape
cols = ["LR input", f"Truth {A}×{Rn}"] + [f"CaloNSF {A * 2 ** k}×{Rn * 2 ** k}" for k in range(args.extra_steps + 1)]
fig = plt.figure(figsize=(2.5 * len(cols), 2.6 * len(picked)))
fig.patch.set_facecolor(plots.SURFACE)
for r, batch in enumerate(dl):
    b = to_device(batch, device)
    imgs = [transverse(np.asarray(batch["lr"]["shower"][0])), transverse(np.asarray(batch["hr"]["shower"][0]))]
    for k in range(args.extra_steps + 1):
        o = model.generate(b, extra_steps=k, ensemble=1, nfe=int(cfg.eval.nfe), solver=cfg.eval.solver)
        imgs.append(transverse(o["grids"]["shower"][0].detach().float().cpu().numpy()))
    for c, (img, t) in enumerate(zip(imgs, cols)):
        ax = fig.add_subplot(len(picked), len(cols), r * len(cols) + c + 1, projection="polar")
        vmax = img.max()  # each grid has its own cell size, so scale the colours per image
        draw_polar(ax, img, max(vmax * 1e-3, 1e-6), max(vmax, 1e-6))
        if r == 0:
            ax.set_title(t, fontsize=9, color=plots.INK)
        if c == 0:
            ax.text(-0.25, 0.5, f"{float(batch['e_inc'][0]) / 1e3:.0f} GeV", transform=ax.transAxes, rotation=90,
                    va="center", fontsize=9, color=plots.INK)
plots._save(fig, os.path.join(args.out, f"beyond_native_x{factor}.png"))
print(f"wrote {args.out}")
