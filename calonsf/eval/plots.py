"""Figures for the CaloChallenge evaluation, in publication style: serif font with proper maths, labelled axes with
symbols and units, panel letters, a legend naming every series, a header with the dataset / LR factor / number
of events, and a footnote saying what error bars and bands show. Each figure is written as a PNG (300 dpi) and as
a vector PDF. Colours are fixed per method; line style and marker differ as well so the figures survive greyscale."""
from __future__ import annotations

import os
import string

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .features import LAYERED, LOG_SCALE  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.8,
})

INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c3"
COLORS = {"calonsf": "#2a78d6", "nearest": "#eb6834", "bicubic": "#1baf7a"}
LINES = {"calonsf": ("-", "o"), "nearest": ("--", "s"), "bicubic": (":", "^")}  # (line style, marker)
LABEL = {"calonsf": "CaloNSF", "nearest": "Nearest", "bicubic": "Bicubic", "geant4": "Geant4"}
KIND_LABEL = {"dnn_low": "DNN, low-level", "dnn_high": "DNN, high-level", "cnn_low": "3D ResNet, low-level"}

# axis text per feature: (panel title, x-axis label of its histogram); units: energies in MeV, positions in mm
TITLE = {"E_layer": r"Layer energy $E_i$", "CoE_eta": r"Centre of energy in $\eta$", "CoE_phi": r"Centre of energy in $\phi$",
         "CoE_r": r"Centre of energy in $r$", "width_eta": r"Width of the centre of energy in $\eta$",
         "width_phi": r"Width of the centre of energy in $\phi$", "width_r": r"Width of the centre of energy in $r$",
         "sparsity": "Sparsity", "E_tot_over_Einc": r"Total deposited energy", "voxel_energy": "Voxel energy"}
SHORT = {**TITLE, "CoE_eta": r"CoE in $\eta$", "CoE_phi": r"CoE in $\phi$", "CoE_r": r"CoE in $r$",
         "width_eta": r"Width of CoE in $\eta$", "width_phi": r"Width of CoE in $\phi$", "width_r": r"Width of CoE in $r$"}
XLABEL = {"E_layer": r"$\log_{10}(E_i\,/\,\mathrm{MeV})$", "CoE_eta": r"$\langle\eta\rangle_i$ [mm]",
          "CoE_phi": r"$\langle\phi\rangle_i$ [mm]", "CoE_r": r"$\langle r\rangle_i$ [mm]",
          "width_eta": r"$\sigma_{\langle\eta\rangle,i}$ [mm]", "width_phi": r"$\sigma_{\langle\phi\rangle,i}$ [mm]",
          "width_r": r"$\sigma_{\langle r\rangle,i}$ [mm]", "sparsity": r"Sparsity $1-$activity$_i$",
          "E_tot_over_Einc": r"$E_{\mathrm{dep}}\,/\,E_{\mathrm{inc}}$",
          "voxel_energy": r"$\log_{10}(I_{ia}\,/\,\mathrm{MeV})$"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def _letter(ax, i):
    ax.text(0.0, 1.04, f"({string.ascii_lowercase[i]})", transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left")


def _col(m):
    return COLORS.get(m, INK2)


def _ls(m):
    return LINES.get(m, ("-", "o"))


def _save(fig, path, info: str = "", note: str = "", legend=None, ncol: int = 4):
    """Header (info) on top, legend and footnote at the bottom; writes path (.png) and the same name as .pdf."""
    top, bottom = (0.965 if info else 1.0), 0.0
    if legend:
        fig.legend(handles=legend, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.025 if note else 0.0))
        bottom = 0.075 if note else 0.05
    elif note:
        bottom = 0.04
    if info:
        fig.suptitle(info, fontsize=10, color=INK2, y=0.995)
    if note:
        fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", facecolor="white")
    plt.close(fig)


def _method_handles(methods, band=False, ref=False, ref_label="Geant4"):
    h = []
    if ref:
        h.append(Patch(facecolor=BAND, edgecolor="none", label=ref_label))
    h += [Line2D([0], [0], color=_col(m), ls=_ls(m)[0], marker=_ls(m)[1], ms=4, lw=1.4, label=LABEL.get(m, m)) for m in methods]
    return h


# ---------------------------------------------------------------------------------------------------- figures
def plot_separation(path, sep: dict, band: dict, info: str):
    """Separation power S of each method's feature histograms with respect to Geant4, per layer."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.6))
    for i, (ax, k) in enumerate(zip(axes.flat, LAYERED)):
        _style(ax)
        _letter(ax, i)
        x = np.arange(len(band[k]["min"]))
        ax.fill_between(x, band[k]["min"], band[k]["max"], color=BAND, lw=0)
        for m, s in sep.items():
            ax.plot(x, s[k], color=_col(m), ls=_ls(m)[0], marker=_ls(m)[1], ms=3, markevery=4, lw=1.3)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(TITLE[k])
        ax.set_xlabel(r"Layer index $i$")
        ax.set_ylabel(r"Separation power $S$ (symlog)")
    ax = axes.flat[-1]
    _style(ax)
    _letter(ax, len(LAYERED))
    names = ["E_tot_over_Einc", "voxel_energy"]
    w = 0.8 / max(len(sep), 1)
    for i, (m, s) in enumerate(sep.items()):
        ax.bar(np.arange(2) + i * w, [s[k][0] for k in names], w, color=_col(m), edgecolor="none")
    for j, k in enumerate(names):
        hi = band[k]["max"][0]
        ax.hlines(hi, j - 0.05, j + 0.85, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(np.arange(2) + 0.4 - w / 2)
    ax.set_xticklabels([TITLE[k] for k in names])
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylabel(r"Separation power $S$ (symlog)")
    ax.set_title("Single histograms")
    _save(fig, path, info, "Grey band: Geant4 vs Geant4 (min-max over random half-splits of the test set). "
                           "Dashed line: upper edge of that band. S = 0: identical histograms, S = 1: no overlap.",
          _method_handles(sep, ref=True, ref_label="Geant4 vs Geant4 (band)"), ncol=4)


def _centers(e):
    return 0.5 * (e[1:] + e[:-1])


def plot_histograms(path, edges: dict, vedges: np.ndarray, h_ref: dict, h: dict, layer: int, info: str):
    """Normalised histograms of the high-level features (a mid-detector layer for the per-layer ones)."""
    items = [("voxel_energy", 0), ("E_tot_over_Einc", 0)] + [(k, layer) for k in LAYERED]
    fig, axes = plt.subplots(2, 5, figsize=(17, 7))
    for i, (ax, (k, j)) in enumerate(zip(axes.flat, items)):
        _style(ax)
        _letter(ax, i)
        e = vedges if k == "voxel_energy" else edges[k][j][0]
        x = _centers(e)
        ref = np.asarray(h_ref[k][j], float)
        ax.fill_between(x, ref / max(ref.sum(), 1), step="mid", color=BAND, lw=0)
        for m, hm in h.items():
            v = np.asarray(hm[k][j], float)
            ax.step(x, v / max(v.sum(), 1), where="mid", color=_col(m), ls=_ls(m)[0], lw=1.3)
        ax.set_yscale("log")
        ax.set_title(SHORT[k] + (rf" ($i={j}$)" if k in LAYERED else ""), fontsize=9)
        ax.set_xlabel(XLABEL[k])
        ax.set_ylabel("Fraction of voxels per bin" if k == "voxel_energy" else "Fraction of showers per bin")
    _save(fig, path, info, "Equal-width bins over the Geant4 range; values outside it fall in the edge bins. CoE: centre of energy; $i$: layer index.",
          _method_handles(h, ref=True, ref_label="Geant4 (reference)"), ncol=4)


def plot_pcc(path, pcc_ref: np.ndarray, pcc: dict, info: str):
    n = 1 + len(pcc)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6))
    im = axes[0].imshow(pcc_ref, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
    axes[0].set_title("Geant4")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    cb.set_label(r"Pearson correlation coefficient $\mathrm{PCC}(E_i, E_j)$")
    _letter(axes[0], 0)
    for i, (ax, (m, c)) in enumerate(zip(axes[1:], pcc.items()), start=1):
        d = c - pcc_ref
        lim = max(float(np.abs(d).max()), 1e-3)
        im2 = ax.imshow(d, vmin=-lim, vmax=lim, cmap="RdBu_r", origin="lower")
        ax.set_title(rf"{LABEL.get(m, m)} $-$ Geant4")
        cb = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"$\Delta\mathrm{PCC}$")
        _letter(ax, i)
    for ax in axes:
        ax.set_xlabel(r"Layer index $j$")
        ax.set_ylabel(r"Layer index $i$")
    _save(fig, path, info, "Pearson correlation of the layer energies over all test showers; the right-hand panels show "
                           "the difference to Geant4 on their own symmetric colour scale.")


def plot_summary(path, R: dict, info: str):
    """Classifier AUC and the distance metrics at a glance."""
    methods = list(R["methods"])
    have = {k for m in methods for k in R["methods"][m]["classifier"]}
    kinds = [k for k in ("dnn_low", "dnn_high", "cnn_low") if k in have]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    xs = np.arange(len(methods))

    ax = axes[0]
    w = 0.8 / max(len(kinds), 1)
    hatches = ["", "//", ".."]
    for i, k in enumerate(kinds):
        ax.bar(xs + i * w, [R["methods"][m]["classifier"][k]["auc_mean"] for m in methods], w,
               yerr=[R["methods"][m]["classifier"][k]["auc_std"] for m in methods], capsize=2, color="#8fb4e3",
               edgecolor=INK, hatch=hatches[i % 3], lw=0.6, label=KIND_LABEL[k])
    ax.axhline(0.5, color=INK, lw=0.9, ls="--", label="AUC = 0.5 (indistinguishable)")
    ax.set_ylabel(r"Classifier AUC")
    ax.set_title("Binary classifier test")
    ax.set_xticks(xs + 0.4 - w / 2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    singles = [(r"Mean separation power $\bar{S}$", [R["methods"][m]["separation_power"]["summary"]["mean_all"] for m in methods], None,
                r"$\bar{S}$ (mean over features)"),
               ("Fréchet physics distance", [R["methods"][m]["fpd"] for m in methods], None, "FPD"),
               ("Kernel physics distance", [R["methods"][m]["kpd"] for m in methods], [R["methods"][m]["kpd_std"] for m in methods], "KPD")]
    for ax, (title, vals, err, yl) in zip(axes[1:], singles):
        ax.bar(xs, vals, 0.6, yerr=err, capsize=2, color=[_col(m) for m in methods], edgecolor=INK, lw=0.6)
        ax.set_title(title)
        ax.set_ylabel(yl)
    for i, ax in enumerate(axes):
        _style(ax)
        _letter(ax, i)
        ax.set_xticks(xs + (0.4 - w / 2 if ax is axes[0] else 0))
        ax.set_xticklabels([LABEL.get(m, m) for m in methods])
        ax.set_xlabel("Method")
    _save(fig, path, info, "Lower is better except the AUC, which is ideal at 0.5. Error bars: standard deviation over classifier "
                           "seeds (AUC) and over random batches (KPD).")


def make_all(fdir: str, edges, vedges, h_ref, band, pcc_ref, plot_data: dict, R: dict, info: str = ""):
    """Every figure of one LR factor; a figure failure must never lose the metrics."""
    n_layers = len(h_ref["E_layer"])
    jobs = [("separation_power.png", lambda p: plot_separation(p, plot_data["sep"], band, info)),
            ("histograms.png", lambda p: plot_histograms(p, edges, vedges, h_ref, plot_data["h"], n_layers // 2, info)),
            ("pcc.png", lambda p: plot_pcc(p, pcc_ref, plot_data["pcc"], info)),
            ("summary.png", lambda p: plot_summary(p, R, info))]
    for name, fn in jobs:
        try:
            fn(os.path.join(fdir, name))
        except Exception as e:  # noqa: BLE001
            print(f"[eval] figure {name} failed: {e}")
