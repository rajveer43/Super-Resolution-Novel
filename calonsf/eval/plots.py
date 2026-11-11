"""Static figures (matplotlib, PNG). Colours: fixed categorical slots assigned by METHOD (never by rank),
truth in neutral ink; one-hue sequential map for energy; two-hue diverging map with a grey
midpoint for residuals."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402

INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_SLOT = {"calonsf": 0, "calonsf_mean": 0, "nearest": 1, "bicubic": 2, "edsr": 3, "rcan": 4, "swinir": 5, "liif": 6,
               "ddpm": 7, "graphfm": 2, "varvq": 3}  # only <=4 series are ever drawn together
LABEL = {"calonsf": "CaloNSF", "calonsf_mean": "CaloNSF (ensemble mean)", "nearest": "LR (nearest)", "bicubic": "Bicubic", "edsr": "EDSR", "rcan": "RCAN",
         "swinir": "SwinIR", "liif": "LIIF", "ddpm": "DDPM (DiffLense)", "graphfm": "Graph-FM", "varvq": "VAR-VQ",
         "gt": "Truth (HR)"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#d9d8d4", "#f2a27f", "#c4471a"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def color(method: str) -> str:
    return INK2 if method == "gt" else SLOTS[METHOD_SLOT.get(method, 0) % len(SLOTS)]


def _view(geom, grids: dict, i: int) -> np.ndarray:
    """2D image for display: QG = sum of channels; CaloGAN = layer_1; CaloChallenge = sum over depth."""
    if geom.name == "qg":
        return np.asarray(grids["jet"][i]).sum(0)
    if geom.name == "calogan":
        return np.asarray(grids["layer_1"][i, 0])
    return np.asarray(grids["shower"][i]).sum(0)


def plot_examples(path, geom, gt, lr, preds: dict, std=None, n: int = 4, methods=("calonsf",)):
    cols = ["lr"] + list(methods) + ["gt", "residual"] + (["std"] if std is not None else [])
    fig, axes = plt.subplots(n, len(cols), figsize=(2.3 * len(cols), 2.3 * n), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r in range(n):
        g = _view(geom, gt, r)
        vmax = max(g.max(), 1e-9)
        vmin = max(vmax * 1e-4, g[g > 0].min() if (g > 0).any() else 1e-9)
        for c, name in enumerate(cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            if name == "lr":
                img = _view(geom, lr, r)
            elif name == "gt":
                img = g
            elif name == "residual":
                res = _view(geom, preds["calonsf"], r) - g
                m = max(np.abs(res).max(), 1e-9)
                ax.imshow(res, cmap=DIV, norm=TwoSlopeNorm(0, -m, m), interpolation="nearest", aspect="auto")
                if r == 0:
                    ax.set_title("CaloNSF - truth", fontsize=9, color=INK)
                continue
            elif name == "std":
                ax.imshow(_view(geom, std, r), cmap=SEQ, interpolation="nearest", aspect="auto")
                if r == 0:
                    ax.set_title("ensemble σ", fontsize=9, color=INK)
                continue
            else:
                img = _view(geom, preds[name], r)
            ax.imshow(np.clip(img, vmin, None), cmap=SEQ, norm=LogNorm(vmin, vmax), interpolation="nearest",
                      aspect="auto")
            if r == 0:
                ax.set_title({"lr": "LR input"}.get(name, LABEL.get(name, name)), fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_observables(path, obs_gt: dict, obs: dict, keys: list[str], methods: list[str]):
    """Histograms (scalar observables) or mean profiles (vector observables). <=4 series per panel."""
    keys = [k for k in keys if k in obs_gt]
    n = len(keys)
    nc = min(3, n)
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.1 * nr), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    series = [("gt", obs_gt)] + [(m, obs[m]) for m in methods if m in obs]
    for ax, k in zip(axes.ravel(), keys):
        _style(ax)
        a = np.asarray(obs_gt[k])
        if a.ndim == 1:
            lo, hi = np.percentile(a, [0.5, 99.5])
            if hi <= lo:
                hi = lo + 1e-6
            bins = np.linspace(lo, hi, 40)
            for m, o in series:
                ax.hist(np.clip(np.asarray(o[k]), lo, hi), bins=bins, histtype="step", lw=2, color=color(m),
                        label=LABEL.get(m, m), density=True, ls="-" if m != "gt" else "-",
                        fill=False)
            if series:
                ax.hist(np.clip(a, lo, hi), bins=bins, density=True, color=INK2, alpha=0.12)
        else:
            x = np.arange(a.shape[1])
            for m, o in series:
                ax.plot(x, np.asarray(o[k]).mean(0), lw=2, color=color(m), label=LABEL.get(m, m),
                        marker="o" if a.shape[1] < 12 else None, ms=4)
            ax.set_xlabel("bin", fontsize=8, color=INK2)
        ax.set_title(k, fontsize=9, color=INK)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(series), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_curve(path, x, ys: dict, xlabel: str, ylabel: str, title: str, logx: bool = False):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for m, y in ys.items():
        ax.plot(x, y, lw=2, marker="o", ms=5, color=color(m), label=LABEL.get(m, m))
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.set_title(title, fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_roc(path, curves: dict):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for m, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=2, color=color(m), label=f"{LABEL.get(m, m)}  AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    ax.set_xlabel("false positive rate", fontsize=9, color=INK2)
    ax.set_ylabel("true positive rate", fontsize=9, color=INK2)
    ax.set_title("Quark/gluon tagger on SR inputs", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def save_image(path, img: np.ndarray, title: str = ""):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(SURFACE)
    vmax = max(img.max(), 1e-9)
    ax.imshow(np.clip(img, vmax * 1e-5, None), cmap=SEQ, norm=LogNorm(vmax * 1e-5, vmax), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def ensure(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------------ detailed calorimeter diagnostics
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _range(arrs, lo_q=0.5, hi_q=99.5):
    v = np.concatenate([np.asarray(a, float).ravel() for a in arrs])
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, [lo_q, hi_q]) if len(v) else (0.0, 1.0)
    return (lo, hi) if hi > lo else (lo - 0.5, hi + 0.5)


def _ratio_panel(rx, edges, ref_counts, ratios: dict):
    for m, r in ratios.items():
        rx.stairs(np.clip(r, 0.5, 1.5), edges, baseline=None, color=color(m), lw=1.4)
    rx.axhline(1, color=INK2, lw=0.8)
    rx.set_ylim(0.5, 1.5)
    rx.set_ylabel("ratio to truth", fontsize=8, color=INK2)


def plot_hist_ratio(out_dir, obs_gt: dict, obs: dict, methods: list[str], bins: int = 40, min_ref: int = 10):
    """One figure per scalar observable: normalised histograms + ratio-to-truth panel (bins with < min_ref truth
    events get no ratio; ratios outside 0.5-1.5 are drawn on the edge so large shifts stay visible)."""
    for k, a in obs_gt.items():
        a = np.asarray(a, float)
        if a.ndim != 1 or a.std() == 0:
            continue
        series = {m: np.asarray(obs[m][k], float) for m in methods if m in obs and k in obs[m]}
        lo, hi = _range([a, *series.values()])
        edges = np.linspace(lo, hi, bins + 1)
        fig, (ax, rx) = plt.subplots(2, 1, figsize=(5, 4.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        fig.patch.set_facecolor(SURFACE)
        _style(ax)
        _style(rx)
        n_ref, _ = np.histogram(a, edges)
        h_ref = n_ref / len(a) / np.diff(edges)
        ax.stairs(h_ref, edges, fill=True, color=INK2, alpha=0.15)
        ax.stairs(h_ref, edges, color=INK2, lw=1.6, label=LABEL["gt"])
        ratios = {}
        for m, v in series.items():
            h = np.histogram(v, edges)[0] / len(v) / np.diff(edges)
            ax.stairs(h, edges, color=color(m), lw=1.6, label=LABEL.get(m, m))
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios[m] = np.where(n_ref >= min_ref, h / h_ref, np.nan)
        _ratio_panel(rx, edges, n_ref, ratios)
        ax.set_ylabel("normalised", fontsize=8, color=INK2)
        ax.set_title(k, fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=7)
        rx.set_xlabel(k, fontsize=8, color=INK2)
        _save(fig, os.path.join(out_dir, f"hist_{k}.png"))


def plot_profiles(out_dir, obs_gt: dict, obs: dict, methods: list[str]):
    """Vector observables (per layer / per radial bin): mean with a 16-84 % band of the per-event values,
    and the ratio of the means to truth. The band shows whether the event-to-event spread is right too."""
    for k, a in obs_gt.items():
        a = np.asarray(a, float)
        if a.ndim != 2 or a.shape[1] < 2:
            continue
        x = np.arange(a.shape[1])
        fig, (ax, rx) = plt.subplots(2, 1, figsize=(5.6, 4.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        fig.patch.set_facecolor(SURFACE)
        _style(ax)
        _style(rx)
        mref = a.mean(0)
        for m, v in [("gt", a)] + [(m, np.asarray(obs[m][k], float)) for m in methods if m in obs and k in obs[m]]:
            mu = v.mean(0)
            lo, hi = np.percentile(v, [16, 84], axis=0)
            ax.plot(x, mu, lw=1.8, color=color(m), label=LABEL.get(m, m))
            ax.fill_between(x, lo, hi, color=color(m), alpha=0.12 if m == "gt" else 0.08, lw=0)
            if m != "gt":
                with np.errstate(divide="ignore", invalid="ignore"):
                    rx.plot(x, np.clip(np.where(np.abs(mref) > 0, mu / mref, np.nan), 0.5, 1.5), lw=1.4,
                            color=color(m))
        rx.axhline(1, color=INK2, lw=0.8)
        rx.set_ylim(0.5, 1.5)
        rx.set_ylabel("mean / truth", fontsize=8, color=INK2)
        rx.set_xlabel("layer" if a.shape[1] > 12 else "bin", fontsize=8, color=INK2)
        ax.set_title(f"{k}: mean and 16-84 % band", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=7)
        _save(fig, os.path.join(out_dir, f"profile_{k}.png"))


def _paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = abs(a.mean()) + 1e-12
    r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
    return (b - a).mean() / s, (b - a).std() / s, r


def plot_scatter(path, obs_gt: dict, obs: dict, keys: list[str], methods: list[str], max_points: int = 3000):
    """Same events: each method's value against the truth value, with y = x, bias, resolution and Pearson r."""
    keys = [k for k in keys if k in obs_gt and np.asarray(obs_gt[k]).ndim == 1]
    methods = [m for m in methods if m in obs]
    if not keys or not methods:
        return
    fig, axes = plt.subplots(len(keys), len(methods), figsize=(3.2 * len(methods), 3.0 * len(keys)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, k in enumerate(keys):
        a = np.asarray(obs_gt[k], float)[:max_points]
        lo, hi = _range([a] + [np.asarray(obs[m][k], float)[:max_points] for m in methods], 0.1, 99.9)
        for j, m in enumerate(methods):
            ax = axes[i, j]
            _style(ax)
            b = np.asarray(obs[m][k], float)[:max_points]
            bias, res, r = _paired(a, b)
            ax.scatter(a, b, s=2, alpha=0.25, color=color(m), rasterized=True)
            ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_title(f"{LABEL.get(m, m)}: bias {100 * bias:+.1f}%  res {100 * res:.1f}%  r {r:.3f}",
                         fontsize=7.5, color=INK)
            if j == 0:
                ax.set_ylabel(f"{k} (method)", fontsize=8, color=INK2)
            if i == len(keys) - 1:
                ax.set_xlabel(f"{k} (truth)", fontsize=8, color=INK2)
    _save(fig, path)


def plot_truth_vs_pred(out_dir, obs_gt: dict, obs: dict, keys: list[str], methods: list[str], bins: int = 20,
                       min_count: int = 20, suffix: str = ""):
    """Event-by-event check (one figure per observable). Top row: 2D histogram of truth (x) vs method (y)
    for each method, with y = x; a perfect model puts all density on the diagonal. Bottom row: in bins of
    the truth value, the mean of (method - truth) (bias) and the RMSE around truth (spread), all methods
    overlaid. Bins with < min_count events are left out."""
    methods = [m for m in methods if m in obs]
    for k in keys:
        if k not in obs_gt or np.asarray(obs_gt[k]).ndim != 1:
            continue
        a = np.asarray(obs_gt[k], float)
        series = {m: np.asarray(obs[m][k], float) for m in methods if k in obs[m]}
        if not series or a.std() == 0:
            continue
        lo, hi = _range([a], 0.5, 99.5)  # axes follow truth; predictions outside are drawn on the edge
        pad = 0.1 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        edges = np.linspace(lo, hi, 61)
        bedges = np.percentile(a[(a >= lo) & (a <= hi)], np.linspace(0, 100, bins + 1))
        bedges = np.unique(bedges)
        centres = 0.5 * (bedges[1:] + bedges[:-1])
        n = len(series)
        fig = plt.figure(figsize=(3.2 * max(n, 2), 6.4))
        fig.patch.set_facecolor(SURFACE)
        gs = fig.add_gridspec(2, 2 * max(n, 2), height_ratios=[1.1, 1])
        span = 2 * max(n, 2) // n
        bx = fig.add_subplot(gs[1, : max(n, 2)])
        rx = fig.add_subplot(gs[1, max(n, 2):])
        for ax in (bx, rx):
            _style(ax)
        for j, (m, b) in enumerate(series.items()):
            ax = fig.add_subplot(gs[0, j * span:(j + 1) * span])
            _style(ax)
            ax.grid(False)
            h, _, _ = np.histogram2d(np.clip(a, lo, hi), np.clip(b, lo, hi), [edges, edges])
            ax.imshow(np.where(h > 0, h, np.nan).T, origin="lower", extent=(lo, hi, lo, hi), cmap=SEQ,
                      norm=LogNorm(1, max(h.max(), 2)), aspect="auto", interpolation="nearest")
            ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, ls="--")
            d = b - a
            ax.set_title(f"{LABEL.get(m, m)}\nbias {d.mean():+.3g}  RMSE {np.sqrt((d ** 2).mean()):.3g}",
                         fontsize=8, color=INK)
            ax.set_xlabel(f"truth {k}", fontsize=8, color=INK2)
            if j == 0:
                ax.set_ylabel(f"predicted {k}", fontsize=8, color=INK2)
            idx = np.digitize(a, bedges[1:-1])
            ok = (a >= bedges[0]) & (a <= bedges[-1])
            mu, rm = np.full(len(centres), np.nan), np.full(len(centres), np.nan)
            for i in range(len(centres)):
                s = ok & (idx == i)
                if s.sum() >= min_count:
                    mu[i] = d[s].mean()
                    rm[i] = np.sqrt((d[s] ** 2).mean())
            bx.plot(centres, mu, "-o", ms=3, lw=1.6, color=color(m), label=LABEL.get(m, m))
            rx.plot(centres, rm, "-o", ms=3, lw=1.6, color=color(m), label=LABEL.get(m, m))
        bx.axhline(0, color=INK2, lw=0.8)
        bx.set_title("mean(pred − truth) in bins of truth", fontsize=9, color=INK)
        rx.set_title("RMSE in bins of truth", fontsize=9, color=INK)
        rx.set_ylim(bottom=0)
        for ax in (bx, rx):
            ax.set_xlabel(f"truth {k}", fontsize=8, color=INK2)
        bx.legend(frameon=False, fontsize=7)
        _save(fig, os.path.join(out_dir, f"tvp_{k}{suffix}.png"))


def plot_w1_heatmap(path, results: dict, methods: list[str], title: str):
    """W1/sigma per observable (rows) and method (columns), log colour scale, value printed in each cell."""
    methods = [m for m in methods if m in results]
    keys = sorted({k for m in methods for k in results[m]["observables"]})
    if not keys:
        return
    M = np.array([[results[m]["observables"].get(k, {}).get("w1_norm", np.nan) for m in methods] for k in keys])
    fig, ax = plt.subplots(figsize=(1.05 * len(methods) + 2.6, 0.32 * len(keys) + 1.6))
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(np.log10(np.clip(M, 1e-4, None)), cmap=SEQ, aspect="auto", vmin=-4, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABEL.get(m, m) for m in methods], rotation=30, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=7.5)
    for i in range(len(keys)):
        for j in range(len(methods)):
            v = M[i, j]
            ax.text(j, i, "–" if not np.isfinite(v) else (f"{v:.2g}" if v >= 1e-3 else "<1e-3"), ha="center",
                    va="center", fontsize=6.5, color="white" if np.isfinite(v) and v > 0.1 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04).set_label("log10(W1/σ), lower is better", fontsize=8)
    ax.set_title(title, fontsize=10, color=INK)
    _save(fig, path)


def _corr_features(obs: dict) -> tuple[list[str], np.ndarray]:
    """Scalar observables, plus each vector observable averaged over its columns (CaloChallenge per-layer ones)."""
    names, cols = [], []
    for k in sorted(obs):
        v = np.asarray(obs[k], float)
        names.append(k if v.ndim == 1 else f"{k} (mean)")
        cols.append(v if v.ndim == 1 else v.reshape(len(v), -1).mean(1))
    return names, np.stack(cols, 1)


def plot_corr_diff(path, obs_gt: dict, obs: dict, methods: list[str]):
    """corr(method) - corr(truth) between observables: checks the joint structure, not only the marginals."""
    names, G = _corr_features(obs_gt)
    keep = G.std(0) > 0
    names = [n for n, k in zip(names, keep) if k]
    with np.errstate(invalid="ignore"):
        cg = np.corrcoef(G[:, keep], rowvar=False)
    methods = [m for m in methods if m in obs]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.4 * len(methods) + 0.8, 4.4), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    im = None
    for ax, m in zip(axes[0], methods):
        _, P = _corr_features(obs[m])
        with np.errstate(invalid="ignore"):
            d = np.nan_to_num(np.corrcoef(P[:, keep], rowvar=False) - cg)
        im = ax.imshow(d, cmap=DIV, norm=TwoSlopeNorm(0, -0.5, 0.5), interpolation="nearest")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        ax.set_title(f"{LABEL.get(m, m)}  max |Δ| {np.abs(d).max():.2f}", fontsize=9, color=INK)
    if im is not None:
        fig.colorbar(im, ax=list(axes[0]), fraction=0.03).set_label("corr(method) − corr(truth)", fontsize=8)
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _residual_views(geom, x: np.ndarray) -> dict:
    """CaloChallenge mean shower (Z, A, R) as 2D maps: depth x radius (summed over angle) and
    angle x radius (summed over depth)."""
    return {"depth × radius": x.sum(1), "angle × radius": x.sum(0)}


def plot_mean_residual(path, geom, gt: dict, preds: dict, methods: list[str]):
    """Mean (method − truth) energy per cell: where a method systematically adds or removes energy."""
    methods = [m for m in methods if m in preds]
    if geom.name == "calochallenge":
        ref = _residual_views(geom, np.asarray(gt["shower"]).mean(0))
        res = {m: _residual_views(geom, np.asarray(preds[m]["shower"]).mean(0) - np.asarray(gt["shower"]).mean(0))
               for m in methods}
    else:
        ref = {g: np.asarray(gt[g]).mean(0)[0] for g in sorted(gt)}
        res = {m: {g: np.asarray(preds[m][g]).mean(0)[0] - ref[g] for g in sorted(gt)} for m in methods}
    groups = list(ref)
    fig, axes = plt.subplots(len(groups), len(methods) + 1, figsize=(2.9 * (len(methods) + 1), 2.6 * len(groups)),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, g in enumerate(groups):
        r = ref[g]
        ax = axes[i, 0]
        ax.imshow(np.clip(r, max(r.max(), 1e-9) * 1e-4, None), cmap=SEQ,
                  norm=LogNorm(max(r.max(), 1e-9) * 1e-4, max(r.max(), 1e-9)), aspect="auto", interpolation="nearest")
        ax.set_ylabel(g, fontsize=8, color=INK2)
        if i == 0:
            ax.set_title("Truth: mean energy", fontsize=8.5, color=INK)
        lim = max(max(np.abs(res[m][g]).max() for m in methods), 1e-12)
        for j, m in enumerate(methods, 1):
            ax = axes[i, j]
            im = ax.imshow(res[m][g], cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", interpolation="nearest")
            if i == 0:
                ax.set_title(f"{LABEL.get(m, m)} − truth", fontsize=8.5, color=INK)
        fig.colorbar(im, ax=list(axes[i, 1:]), fraction=0.02, pad=0.01).set_label("MeV / cell", fontsize=7)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _cells(grids: dict, thr: float = 0.0, max_cells: int = 400000, seed: int = 0) -> np.ndarray:
    v = np.concatenate([np.asarray(x).ravel() for x in grids.values()])
    v = v[v > thr]
    if len(v) > max_cells:
        v = np.random.default_rng(seed).choice(v, max_cells, replace=False)
    return v


def plot_cell_spectrum(path, gt: dict, preds: dict, methods: list[str], bins: int = 60):
    """Energy spectrum of the non-empty cells (log10 MeV) with a ratio panel; low-energy speckle and missing
    hot cores show up at the two ends."""
    a = np.log10(_cells(gt))
    series = {m: np.log10(_cells(preds[m])) for m in methods if m in preds}
    lo, hi = _range([a, *series.values()], 0.1, 99.9)
    edges = np.linspace(lo, hi, bins + 1)
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(5.4, 4.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    _style(rx)
    n_ref = np.histogram(a, edges)[0]
    h_ref = n_ref / len(a) / np.diff(edges)
    ax.stairs(h_ref, edges, fill=True, color=INK2, alpha=0.15)
    ax.stairs(h_ref, edges, color=INK2, lw=1.6, label=LABEL["gt"])
    ratios = {}
    for m, v in series.items():
        h = np.histogram(v, edges)[0] / len(v) / np.diff(edges)
        ax.stairs(h, edges, color=color(m), lw=1.6, label=LABEL.get(m, m))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios[m] = np.where(n_ref >= 10, h / h_ref, np.nan)
    _ratio_panel(rx, edges, n_ref, ratios)
    ax.set_yscale("log")
    ax.set_title("Cell energy spectrum (non-empty cells)", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7)
    rx.set_xlabel("log10(cell energy / MeV)", fontsize=8, color=INK2)
    _save(fig, path)


def plot_occupancy(path, gt: dict, preds: dict, methods: list[str], thresholds=None):
    """Hit precision and recall against truth vs the readout threshold, plus the number of hit cells relative to truth."""
    g = np.concatenate([np.asarray(x).ravel() for x in gt.values()])
    thresholds = thresholds if thresholds is not None else np.logspace(-3, 2, 16)  # MeV
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    fig.patch.set_facecolor(SURFACE)
    for m in methods:
        if m not in preds:
            continue
        p = np.concatenate([np.asarray(x).ravel() for x in preds[m].values()])
        prec, rec, occ = [], [], []
        for t in thresholds:
            hg, hp = g > t, p > t
            tp = (hg & hp).sum()
            prec.append(tp / max(hp.sum(), 1))
            rec.append(tp / max(hg.sum(), 1))
            occ.append(hp.sum() / max(hg.sum(), 1))
        for ax, y in zip(axes, (prec, rec, occ)):
            ax.plot(thresholds, y, lw=1.8, marker="o", ms=3, color=color(m), label=LABEL.get(m, m))
    for ax, t in zip(axes, ("hit precision", "hit recall", "hit cells / truth hit cells")):
        _style(ax)
        ax.set_xscale("log")
        ax.set_xlabel("threshold [MeV]", fontsize=8, color=INK2)
        ax.set_title(t, fontsize=9.5, color=INK)
    axes[2].axhline(1, color=INK2, lw=0.8)
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, path)


def plot_classifier_scores(path, scores: dict):
    """Classifier two-sample test: score histograms of held-out truth vs method events (overlap = hard to tell)."""
    ms = list(scores)
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(3.6 * len(ms), 3.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    edges = np.linspace(0, 1, 41)
    for ax, m in zip(axes[0], ms):
        _style(ax)
        p, y, auc = scores[m]
        ax.hist(p[y == 0], edges, density=True, histtype="stepfilled", color=INK2, alpha=0.25, label=LABEL["gt"])
        ax.hist(p[y == 1], edges, density=True, histtype="step", lw=1.8, color=color(m), label=LABEL.get(m, m))
        ax.set_yscale("log")
        ax.set_title(f"{LABEL.get(m, m)}: AUC {auc:.3f}", fontsize=9, color=INK)
        ax.set_xlabel("classifier P(method)", fontsize=8, color=INK2)
        ax.legend(frameon=False, fontsize=7)
    _save(fig, path)


def plot_calibration(path, gt: dict, mean: dict, std: dict, ks=None):
    """Ensemble coverage: fraction of non-empty truth cells within mean ± kσ vs k, against the Gaussian ideal."""
    from scipy.special import erf
    ks = np.linspace(0.25, 3, 12) if ks is None else ks
    a = np.concatenate([np.asarray(gt[g]).ravel() for g in gt])
    m = np.concatenate([np.asarray(mean[g]).ravel() for g in gt])
    s = np.concatenate([np.asarray(std[g]).ravel() for g in gt])
    sel = a > 0
    if not sel.any() or s.max() == 0:
        return
    cov = [(np.abs(m[sel] - a[sel]) <= k * s[sel] + 1e-12).mean() for k in ks]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot(ks, erf(ks / np.sqrt(2)), color=INK2, lw=1.2, ls="--", label="Gaussian ideal")
    ax.plot(ks, cov, lw=2, marker="o", ms=4, color=color("calonsf"), label=LABEL["calonsf"])
    ax.set_xlabel("k (interval = mean ± kσ)", fontsize=8, color=INK2)
    ax.set_ylabel("coverage of truth cells", fontsize=8, color=INK2)
    ax.set_title("Ensemble calibration (below the line = overconfident)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7)
    _save(fig, path)


def plot_downstream_vs_energy(path, results: dict, methods: list[str], gt_metrics: dict | None = None):
    """Downstream metrics (energy bias / resolution, accuracy) per incident-energy bin, one line per method."""
    def bins_of(r):
        return {k: v for k, v in (r or {}).items() if k.startswith("E[")}
    methods = [m for m in methods if bins_of(results.get(m, {}).get("downstream_task_by_subset"))]
    if not methods:
        return
    labels = list(bins_of(results[methods[0]]["downstream_task_by_subset"]))
    metrics = [k for k in ("rel_bias", "rel_resolution", "accuracy")
               if k in results[methods[0]]["downstream_task_by_subset"][labels[0]]]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(labels))
    for ax, k in zip(axes[0], metrics):
        _style(ax)
        if gt_metrics:
            gb = bins_of(gt_metrics)
            ax.plot(x, [gb.get(l, {}).get(k, np.nan) for l in labels], color=INK2, lw=1.4, ls="--",
                    label="model on truth")
        for m in methods:
            b = bins_of(results[m]["downstream_task_by_subset"])
            ax.plot(x, [b.get(l, {}).get(k, np.nan) for l in labels], lw=1.8, marker="o", ms=4, color=color(m),
                    label=LABEL.get(m, m))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=20)
        ax.set_title(k, fontsize=9.5, color=INK)
    axes[0, 0].legend(frameon=False, fontsize=7)
    _save(fig, path)


def plot_w1_vs_factor(path, factors: dict, methods: list[str]):
    """Mean key-observable W1/sigma against the LR factor, one line per method (log y)."""
    fs = sorted(factors, key=int)
    if len(fs) < 2:
        return
    fig, ax = plt.subplots(figsize=(5, 3.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for m in methods:
        y = [factors[f]["methods"].get(m, {}).get("key_obs_mean_w1_norm") for f in fs]
        if any(v is not None for v in y):
            ax.plot([int(f) for f in fs], [np.nan if v is None else v for v in y], lw=1.8, marker="o", ms=5,
                    color=color(m), label=LABEL.get(m, m))
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks([int(f) for f in fs])
    ax.set_xticklabels([f"×{f}" for f in fs])
    ax.set_xlabel("LR factor", fontsize=8, color=INK2)
    ax.set_ylabel("key-observable W1/σ (lower is better)", fontsize=8, color=INK2)
    ax.set_title("Physics fidelity vs down-sampling factor", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7)
    _save(fig, path)
