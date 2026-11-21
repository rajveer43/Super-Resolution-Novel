"""Sample images for the beyond-native checks: the shower as a Cartesian x-y image (x = eta, y = phi, energy summed
over the 45 layers) at every resolution, in the same publication style as `plots.py`.

All panels show the energy DENSITY [MeV mm^-2] (pixel energy / pixel area) on one shared logarithmic colour scale,
so a finer image is directly comparable to a coarser one; empty pixels are white."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from ..utils.viz import SEQ  # noqa: E402
from .features import refine_r_edges  # noqa: E402
from .plots import INK2  # noqa: E402  (also sets the serif / maths rcParams)


def render_polar_density(energy: np.ndarray, r_edges, size: int, half: float) -> np.ndarray:
    """(A, R) polar cell energies -> (size, size) energy density [MeV mm^-2] by looking up the cell under each pixel
    centre (for display of the native-resolution grids). Rows = y (phi), columns = x (eta); 0 outside the disc."""
    A, R = energy.shape
    e = np.asarray(r_edges, np.float64)
    e = e if len(e) == R + 1 else refine_r_edges(e, R)
    c = -half + (np.arange(size) + 0.5) * (2 * half / size)
    X, Y = np.meshgrid(c, c)
    rr, aa = np.hypot(X, Y), np.mod(np.arctan2(Y, X), 2 * np.pi)
    ri = np.clip(np.searchsorted(e, rr, side="right") - 1, 0, R - 1)
    ai = np.clip((aa / (2 * np.pi) * A).astype(int), 0, A - 1)
    area = 0.5 * (e[1:] ** 2 - e[:-1] ** 2) * (2 * np.pi / A)
    return np.where(rr < e[-1], energy[ai, ri] / area[ri], 0.0)


def _panel(ax, dens, half, norm, title, extent=None, xlabel=True, ylabel=True, circle=True):
    ext = extent or (-half, half, -half, half)
    im = ax.imshow(np.ma.masked_less_equal(dens, 0), cmap=SEQ.with_extremes(bad="white"), norm=norm, extent=ext,
                   origin="lower", interpolation="nearest")
    if circle:
        ax.add_patch(plt.Circle((0, 0), half, fill=False, ls="--", lw=0.6, color=INK2))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$x=\eta$ [mm]" if xlabel else "")
    ax.set_ylabel(r"$y=\phi$ [mm]" if ylabel else "")
    return im


def _finish(fig, path, info, note, im, axes):
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(r"Energy density [MeV mm$^{-2}$] (log scale)")
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.94))  # room for the header and the footnote
    fig.suptitle(info, fontsize=10, color=INK2)
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=8, color=INK2)
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", facecolor="white")
    plt.close(fig)


def plot_ladder(path, panels: list, half: float, info: str, ncols: int = 4):
    """panels: [(title, density image)]; full view of the detector at every resolution."""
    vmax = max(float(d.max()) for _, d in panels)
    norm = LogNorm(vmax * 1e-5, vmax)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols + 1.0, 4.0 * nrows + 0.8), squeeze=False,
                             constrained_layout=True)
    for i, (ax, (t, d)) in enumerate(zip(axes.flat, panels)):
        im = _panel(ax, d, half, norm, t, xlabel=i >= (nrows - 1) * ncols, ylabel=i % ncols == 0)
        ax.text(0.0, 1.07, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=10, fontweight="bold")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    _finish(fig, path, info, "Cartesian x-y image of the whole detector, energy summed over the 45 layers; shared logarithmic "
                             "colour scale; the dashed circle is the outer radius.", im, axes)


def plot_zoom(path, images: dict, pixels: dict, half: float, centre, win: float, info: str):
    """The same window (+-win mm around `centre`) cut out of each image size, to show the pixel-level detail."""
    crops = []
    for N, img in images.items():
        p = pixels[N]
        x0, x1 = centre[0] - win, centre[0] + win
        y0, y1 = centre[1] - win, centre[1] + win
        i0, i1 = [int(np.clip(np.floor((v + half) / p), 0, N)) for v in (x0, x1)]
        j0, j1 = [int(np.clip(np.floor((v + half) / p), 0, N)) for v in (y0, y1)]
        crops.append((N, img[j0:j1, i0:i1] / (p * p), (-half + i0 * p, -half + i1 * p, -half + j0 * p, -half + j1 * p)))
    vmax = max(float(c.max()) for _, c, _ in crops)
    norm = LogNorm(vmax * 1e-5, vmax)
    fig, axes = plt.subplots(1, len(crops), figsize=(4.2 * len(crops) + 1.0, 4.4), squeeze=False, constrained_layout=True)
    for i, (ax, (N, c, ext)) in enumerate(zip(axes[0], crops)):
        im = _panel(ax, c, half, norm, f"{N} $\\times$ {N} (pixel {pixels[N]:.3g} mm)", extent=ext, ylabel=i == 0, circle=False)
        ax.text(0.0, 1.07, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=10, fontweight="bold")
    _finish(fig, path, info, f"Zoom of $\\pm${win:g} mm around the centre of energy of the shower; same logarithmic colour scale in all panels.",
            im, axes)
