"""Training curves of the two stages (JEPA pretraining, super-resolution) from the CSV logs the trainer writes
(`jepa/jepa_log.csv`, `sr/sr_log.csv`), in the same publication style as the evaluation figures: epoch on the x axis,
train (dashed) and validation (solid) for every loss, the checkpoint kept as `best.pt` marked, and for the JEPA the
standard deviation of the target features (it falls to ~0 if the representation collapses)."""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from .plots import GRID, INK, INK2, _letter, _save, _style  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2", "#c0a020"]
NOTE = "Dashed: training set (running mean over the epoch). Solid: validation set. Triangle: epoch kept as best.pt."


def read_log(path: str) -> dict:
    """{column: float array} of a trainer CSV log; empty if the file is missing."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
        except (TypeError, ValueError):
            pass
    return out


def best_epoch(log: dict) -> int:
    """Row index of the epoch the trainer kept: lowest val_metric if it was logged, else lowest val_loss."""
    for key in ("val_metric", "val_loss"):
        v = log.get(key)
        if v is not None and np.isfinite(v).any():
            return int(np.nanargmin(v))
    return -1


def _curves(ax, log, names, labels, log_y=True):
    """Train (dashed) and validation (solid) curve for every name; one colour per name."""
    ep = log["epoch"] + 1
    for i, (n, lab) in enumerate(zip(names, labels)):
        c = PALETTE[i % len(PALETTE)]
        if f"train_{n}" in log:
            ax.plot(ep, log[f"train_{n}"], color=c, ls="--", lw=1.2)
        if f"val_{n}" in log:
            ax.plot(ep, log[f"val_{n}"], color=c, ls="-", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(y=0.12)
    if log_y:
        ax.set_yscale("log")
    _style(ax)
    return [Line2D([0], [0], color=PALETTE[i % len(PALETTE)], lw=1.5, label=lab) for i, lab in enumerate(labels)]


def _mark_best(ax, log, key):
    b = best_epoch(log)
    if b >= 0 and key in log and np.isfinite(log[key][b]):
        ax.plot(log["epoch"][b] + 1, log[key][b], marker="v", ms=8, color=INK, mec="white", zorder=5, ls="none")


def plot_jepa(log: dict, path: str, info: str = ""):
    if not log or "epoch" not in log:
        return None
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6))
    _curves(axs[0], log, ["loss"], ["Total"])
    axs[0].set_ylabel(r"JEPA loss $\mathcal{L}_{\mathrm{JEPA}}$")
    _mark_best(axs[0], log, "val_loss")
    h = _curves(axs[1], log, ["jepa_in", "jepa_cross"], ["In-scale", "Cross-scale"])
    axs[1].set_ylabel(r"Loss component")
    _curves(axs[2], log, ["target_std"], ["Target-feature std"], log_y=False)
    axs[2].set_ylabel(r"Std of target features")
    axs[2].set_ylim(bottom=0)
    for i, a in enumerate(axs):
        _letter(a, i)
    _save(fig, path, info or "JEPA pretraining, CaloChallenge Dataset 2", NOTE, legend=h, ncol=2)
    return path


def plot_sr(log: dict, path: str, info: str = ""):
    if not log or "epoch" not in log:
        return None
    has_metric = "val_metric" in log and np.isfinite(log["val_metric"]).any()
    fig, axs = plt.subplots(1, 3 if has_metric else 2, figsize=(11.5 if has_metric else 8, 3.6))
    _curves(axs[0], log, ["loss"], ["Total"])
    axs[0].set_ylabel(r"Training objective $\mathcal{L}$")
    _mark_best(axs[0], log, "val_loss")
    comp = [(n, lab) for n, lab in (("fm", "Flow matching"), ("occ", "Occupancy (BCE)"), ("profile", "Profile"),
                                    ("jepa", "JEPA")) if f"val_{n}" in log or f"train_{n}" in log]
    h = _curves(axs[1], log, [n for n, _ in comp], [lab for _, lab in comp])
    axs[1].set_ylabel("Loss component")
    if has_metric:
        axs[2].plot(log["epoch"] + 1, log["val_metric"], color=PALETTE[0], lw=1.5, marker="o", ms=3)
        _mark_best(axs[2], log, "val_metric")
        axs[2].set_xlabel("Epoch")
        axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[2].margins(y=0.12)
        axs[2].set_ylabel("Mean separation power (validation)")
        _style(axs[2])
    for i, a in enumerate(axs):
        _letter(a, i)
    note = NOTE + (" Panel (c) is the criterion used to choose best.pt (lower is better)." if has_metric else "")
    _save(fig, path, info or "CaloNSF super-resolution training, CaloChallenge Dataset 2", note, legend=h, ncol=4)
    return path


def plot_training(exp_dir: str, out_dir: str | None = None) -> list:
    """Write training_jepa.{png,pdf} and training_sr.{png,pdf} into out_dir (default <exp_dir>/eval); returns the paths."""
    out_dir = out_dir or os.path.join(exp_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    made = [plot_jepa(read_log(os.path.join(exp_dir, "jepa", "jepa_log.csv")), os.path.join(out_dir, "training_jepa.png")),
            plot_sr(read_log(os.path.join(exp_dir, "sr", "sr_log.csv")), os.path.join(out_dir, "training_sr.png"))]
    return [p for p in made if p]
