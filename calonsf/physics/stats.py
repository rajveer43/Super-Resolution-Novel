"""Distribution comparisons: 1D Wasserstein, KS, per-event residuals, Frechet (FPD) and kernel (KPD)
physics distances (JetNet-style, implemented here to avoid extra dependencies)."""
from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.stats import ks_2samp, wasserstein_distance


def _cols(a: np.ndarray) -> np.ndarray:
    return a.reshape(a.shape[0], -1)


def compare(gt: dict, pred: dict, mask: np.ndarray | None = None) -> dict:
    """Per observable: W1 (raw and divided by GT std), KS statistic, mean/std of the per-event relative
    residual (pred-gt)/|gt|. Vector observables are averaged over columns."""
    out = {}
    for k in gt:
        if k not in pred:
            continue
        a, b = _cols(gt[k]), _cols(pred[k])
        if mask is not None:
            a, b = a[mask], b[mask]
        if len(a) < 2:
            continue
        w1, w1n, ks, mr, sr = [], [], [], [], []
        for j in range(a.shape[1]):
            x, y = a[:, j], b[:, j]
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                continue
            s = x.std()
            w = wasserstein_distance(x, y)
            w1.append(w)
            w1n.append(w / s if s > 0 else 0.0)
            ks.append(ks_2samp(x, y).statistic if s > 0 or y.std() > 0 else 0.0)
            den = np.abs(x) + 1e-8
            nz = np.abs(x) > 1e-8 * (np.abs(x).max() + 1e-30)
            rel = ((y - x) / den)[nz] if nz.any() else np.zeros(1)
            mr.append(float(np.mean(rel)))
            sr.append(float(np.std(rel)))
        if w1:
            out[k] = {"w1": float(np.mean(w1)), "w1_norm": float(np.mean(w1n)), "ks": float(np.mean(ks)),
                      "rel_bias": float(np.mean(mr)), "rel_spread": float(np.mean(sr))}
    return out


def feature_matrix(obs: dict, keys: list[str] | None = None) -> np.ndarray:
    keys = keys or sorted(obs)
    return np.concatenate([_cols(obs[k]) for k in keys], axis=1)


def _standardize(a: np.ndarray, b: np.ndarray):
    mu, sd = a.mean(0), a.std(0)
    keep = sd > 1e-12
    return (a[:, keep] - mu[keep]) / sd[keep], (b[:, keep] - mu[keep]) / sd[keep]


def fpd(gt_feats: np.ndarray, pred_feats: np.ndarray) -> float:
    """Frechet physics distance on standardized high-level features."""
    a, b = _standardize(gt_feats, pred_feats)
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    s1, s2 = np.atleast_2d(s1), np.atleast_2d(s2)
    covmean = linalg.sqrtm(s1 @ s2)
    if isinstance(covmean, tuple):  # older scipy returns (sqrtm, errest) with disp=False
        covmean = covmean[0]
    covmean = np.real(covmean)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def kpd(gt_feats: np.ndarray, pred_feats: np.ndarray, n_batches: int = 10, batch: int = 500, seed: int = 0):
    """Unbiased MMD^2 with a cubic polynomial kernel; returns (mean, std) over batches."""
    a, b = _standardize(gt_feats, pred_feats)
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    m = min(batch, len(a), len(b))
    vals = []
    for _ in range(n_batches):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        k = lambda p, q: (p @ q.T / d + 1) ** 3
        kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
        np.fill_diagonal(kxx, 0)
        np.fill_diagonal(kyy, 0)
        vals.append(kxx.sum() / (m * (m - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def subsets(meta: dict, cfg, geom_name: str) -> dict[str, np.ndarray]:
    """Masks for per-class / per-particle / per-energy-bin analysis."""
    out = {"all": np.ones(len(meta["label"]), dtype=bool)}
    names = cfg.analysis.get("class_names", None)
    lab = np.asarray(meta["label"])
    if names:
        for i, n in enumerate(names):
            if (lab == i).any():
                out[f"class={n}"] = lab == i
    bins = cfg.analysis.get("energy_bins", None)
    e = np.asarray(meta["e_inc"])
    if bins and (e > 0).any():
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (e >= lo) & (e < hi)
            if m.sum() >= 5:
                out[f"E[{lo},{hi})"] = m
    return out
