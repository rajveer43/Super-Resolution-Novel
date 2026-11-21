"""Computer-science-inspired metrics (CaloChallenge section 8.4): the Frechet physics distance (FPD) and the
kernel physics distance (KPD), both on the standardised high-level features of section 8.1 (the features are
standardised with the reference mean / std). The same definitions as the JetNet implementation the challenge
used: FPD is the 2-Wasserstein distance between Gaussian fits, KPD an unbiased MMD^2 with a cubic polynomial
kernel, averaged over batches."""
from __future__ import annotations

import numpy as np
from scipy import linalg


def _standardize(a: np.ndarray, b: np.ndarray):
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu, sd = a.mean(0), a.std(0)
    keep = sd > 1e-8 * np.maximum(np.abs(mu), 1.0)  # drop (numerically) constant features: z-scores would explode
    return (a[:, keep] - mu[keep]) / sd[keep], (b[:, keep] - mu[keep]) / sd[keep]


def fpd(ref: np.ndarray, pred: np.ndarray) -> float:
    a, b = _standardize(ref, pred)
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.atleast_2d(np.cov(a, rowvar=False)), np.atleast_2d(np.cov(b, rowvar=False))
    covmean = linalg.sqrtm(s1 @ s2)
    covmean = np.real(covmean[0] if isinstance(covmean, tuple) else covmean)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def _kernel_sum(p: np.ndarray, q: np.ndarray, d: int) -> float:
    return float(((p @ q.T / d + 1) ** 3).sum())  # float64: the cube overflows float32 for outlying features


def kpd(ref: np.ndarray, pred: np.ndarray, batch: int = 10000, n_batches: int = 10, seed: int = 0):
    """(mean, std) over `n_batches` random batches of `batch` events (fewer if the sets are smaller)."""
    a, b = _standardize(ref, pred)
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    m = min(batch, len(a), len(b))
    vals = []
    for _ in range(n_batches):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        diag_x = (((x * x).sum(1) / d + 1) ** 3).sum()  # the unbiased estimate leaves out the i == j terms
        diag_y = (((y * y).sum(1) / d + 1) ** 3).sum()
        vals.append((_kernel_sum(x, x, d) - diag_x) / (m * (m - 1)) + (_kernel_sum(y, y, d) - diag_y) / (m * (m - 1))
                    - 2 * _kernel_sum(x, y, d) / (m * m))
    return float(np.mean(vals)), float(np.std(vals))
