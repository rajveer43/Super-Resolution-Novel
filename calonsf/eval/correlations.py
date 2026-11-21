"""Pearson correlation coefficients between the layer energies (CaloChallenge section 8.2):

    PCC(E_i, E_j) = sum_k (E_i,k - mean E_i)(E_j,k - mean E_j) / (sqrt(sum_k (E_i,k - mean E_i)^2) sqrt(...)),

k running over the showers. A layer with constant energy (e.g. always empty) has no defined PCC; its row and
column are set to 0 (and 1 on the diagonal)."""
from __future__ import annotations

import numpy as np


def layer_pcc(e_layer: np.ndarray) -> np.ndarray:
    """(N, Z) layer energies -> (Z, Z) correlation matrix."""
    x = np.asarray(e_layer, np.float64)
    x = x - x.mean(0, keepdims=True)
    norm = np.sqrt((x ** 2).sum(0))
    ok = norm > 0
    c = np.zeros((x.shape[1], x.shape[1]))
    xo = x[:, ok] / norm[ok]
    c[np.ix_(ok, ok)] = xo.T @ xo
    np.fill_diagonal(c, 1.0)
    return c


def pcc_difference(ref: np.ndarray, pred: np.ndarray) -> dict:
    """Mean / max absolute difference and RMS difference of two PCC matrices (off-diagonal entries)."""
    off = ~np.eye(len(ref), dtype=bool)
    d = (pred - ref)[off]
    return {"mean_abs_diff": float(np.abs(d).mean()), "max_abs_diff": float(np.abs(d).max()),
            "rms_diff": float(np.sqrt((d ** 2).mean()))}
