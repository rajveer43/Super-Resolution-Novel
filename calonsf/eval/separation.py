"""Histogram comparison of high-level features with the CaloChallenge separation power

    S(h1, h2) = 1/2 sum_i (h1_i - h2_i)^2 / (h1_i + h2_i),   sum_i h_j,i = 1,

so S = 0 iff h1 = h2 and S = 1 iff the histograms do not overlap. Bin edges come from the reference (Geant4)
set: `n_bins` equal-width bins over its range (log10 for quantities that span orders of magnitude), and values
outside that range are counted in the edge bins so no mass is lost. The statistical spread of S between two
independent Geant4 samples is estimated by splitting the reference set into random halves (the "Geant4 band")."""
from __future__ import annotations

import warnings

import numpy as np

from .features import LAYERED, LOG_SCALE, SCALAR


def separation_power(h1: np.ndarray, h2: np.ndarray) -> float:
    h1 = np.asarray(h1, np.float64)
    h2 = np.asarray(h2, np.float64)
    if h1.sum() == 0 or h2.sum() == 0:
        return float("nan")
    h1, h2 = h1 / h1.sum(), h2 / h2.sum()
    den = h1 + h2
    ok = den > 0
    return float(0.5 * (((h1 - h2) ** 2)[ok] / den[ok]).sum())


def _tf(name: str, v: np.ndarray, floor: float) -> np.ndarray:
    return np.log10(np.maximum(v, floor)) if name in LOG_SCALE else v


def make_edges(ref: dict, n_bins: int = 50) -> dict:
    """{feature: (edges (n_bins+1,), per-layer edges for layered features)} from the reference features."""
    edges = {}
    for k in [*SCALAR, *LAYERED]:
        v = ref[k]
        cols = [v] if v.ndim == 1 else [v[:, j] for j in range(v.shape[1])]
        e = []
        for c in cols:
            c = c[np.isfinite(c)]
            floor = 1e-12
            if k in LOG_SCALE:
                pos = c[c > 0]
                floor = float(pos.min()) if len(pos) else 1.0
            t = _tf(k, c, floor) if len(c) else np.zeros(1)
            lo, hi = float(t.min()), float(t.max())
            if hi <= lo:
                hi = lo + 1.0
            e.append((np.linspace(lo, hi, n_bins + 1), floor))
        edges[k] = e
    return edges


def _hist(name: str, v: np.ndarray, edge_floor) -> np.ndarray:
    e, floor = edge_floor
    v = v[np.isfinite(v)]
    t = np.clip(_tf(name, v, floor), e[0], e[-1])
    return np.histogram(t, bins=e)[0]


def feature_hists(feats: dict, edges: dict, idx: np.ndarray | None = None) -> dict:
    """{feature: list over layers of histogram counts} (a single entry for scalar features)."""
    out = {}
    for k, ef in edges.items():
        v = feats[k] if idx is None else feats[k][idx]
        if v.ndim == 1:
            out[k] = [_hist(k, v, ef[0])]
        else:
            out[k] = [_hist(k, v[:, j], ef[j]) for j in range(v.shape[1])]
    return out


def sep_from_hists(h_ref: dict, h_pred: dict) -> dict:
    return {k: [separation_power(a, b) for a, b in zip(h_ref[k], h_pred[k])] for k in h_ref}


def voxel_edges(ref_shower: np.ndarray, n_bins: int = 50, chunk: int = 500) -> np.ndarray:
    """Log10 bin edges for the positive voxel energies of the reference set."""
    lo, hi = np.inf, -np.inf
    for s in range(0, len(ref_shower), chunk):
        x = ref_shower[s:s + chunk]
        pos = x[x > 0]
        if len(pos):
            lo, hi = min(lo, float(pos.min())), max(hi, float(pos.max()))
    if not np.isfinite(lo):
        lo, hi = 1e-3, 1.0
    lo, hi = np.log10(lo), np.log10(max(hi, lo * 10))
    return np.linspace(lo, hi, n_bins + 1)


def voxel_event_hists(shower: np.ndarray, edges: np.ndarray, chunk: int = 500) -> np.ndarray:
    """(N, n_bins) counts of the positive voxel energies of every event, so any subset of events can be
    histogrammed by summing rows (used for the Geant4 band)."""
    nb = len(edges) - 1
    out = np.zeros((len(shower), nb), np.int64)
    for s in range(0, len(shower), chunk):
        x = shower[s:s + chunk].reshape(min(chunk, len(shower) - s), -1)
        m = x > 0
        b = np.clip(np.searchsorted(edges, np.log10(np.where(m, x, 1.0)), side="right") - 1, 0, nb - 1)
        flat = (b + (np.arange(len(x)) * nb)[:, None])[m]
        out[s:s + len(x)] = np.bincount(flat, minlength=len(x) * nb).reshape(len(x), nb)
    return out


def geant4_band(ref_feats: dict, ref_vox_hist: np.ndarray, edges: dict, n_pairs: int = 10, seed: int = 0) -> dict:
    """Min / max separation power over `n_pairs` random half-splits of the reference set."""
    rng = np.random.default_rng(seed)
    n = len(ref_vox_hist)
    runs = []
    for _ in range(n_pairs):
        p = rng.permutation(n)
        a, b = p[: n // 2], p[n // 2: 2 * (n // 2)]
        ha, hb = feature_hists(ref_feats, edges, a), feature_hists(ref_feats, edges, b)
        ha["voxel_energy"] = [ref_vox_hist[a].sum(0)]
        hb["voxel_energy"] = [ref_vox_hist[b].sum(0)]
        runs.append(sep_from_hists(ha, hb))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # layers that are empty in every event have no separation power
        return {k: {"min": np.nanmin(np.array([r[k] for r in runs]), 0).tolist(),
                    "max": np.nanmax(np.array([r[k] for r in runs]), 0).tolist()} for k in runs[0]}


def summarize(sep: dict) -> dict:
    """Mean over layers per feature, plus the mean over all features (equal weight per feature)."""
    per = {k: float(np.nanmean(v)) for k, v in sep.items()}
    per["mean_all"] = float(np.nanmean(list(per.values())))
    return per
