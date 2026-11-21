"""Minimal image output for super-resolved CaloChallenge showers (energy summed over depth, log scale)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "axes.labelsize": 10, "axes.titlesize": 10, "pdf.fonttype": 42})
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def view(grids: dict, i: int) -> np.ndarray:
    """2D image of event i: the (alpha, r) grid summed over the 45 layers."""
    return np.asarray(grids["shower"][i]).sum(0)


def save_image(path: str, img: np.ndarray, title: str = "") -> None:
    """Rows are the angular bins alpha, columns the radial bins r; written as PNG and PDF."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    vmax = max(float(img.max()), 1e-9)
    im = ax.imshow(np.clip(img, vmax * 1e-5, None), cmap=SEQ, norm=LogNorm(vmax * 1e-5, vmax), interpolation="nearest",
                   origin="lower", aspect="auto")
    ax.set_xlabel(r"Radial bin index $a_r$")
    ax.set_ylabel(r"Angular bin index $a_\alpha$")
    ax.set_title(title)
    fig.colorbar(im, ax=ax).set_label(r"Energy summed over layers [MeV] (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", facecolor="white")
    plt.close(fig)
