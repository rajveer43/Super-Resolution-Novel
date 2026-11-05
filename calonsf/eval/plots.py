"""Static figures (matplotlib, PNG). Colours: fixed categorical slots assigned by METHOD (never by rank),
ground truth in neutral ink; one-hue sequential map for energy; two-hue diverging map with a grey
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
METHOD_SLOT = {"calonsf": 0, "nearest": 1, "bicubic": 2, "edsr": 3, "rcan": 4, "swinir": 5, "liif": 6,
               "ddpm": 7, "graphfm": 2, "varvq": 3}  # only <=4 series are ever drawn together
LABEL = {"calonsf": "CaloNSF", "nearest": "LR (nearest)", "bicubic": "Bicubic", "edsr": "EDSR", "rcan": "RCAN",
         "swinir": "SwinIR", "liif": "LIIF", "ddpm": "DDPM (DiffLense)", "graphfm": "Graph-FM", "varvq": "VAR-VQ",
         "gt": "Ground truth"}
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
                    ax.set_title("CaloNSF - GT", fontsize=9, color=INK)
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
