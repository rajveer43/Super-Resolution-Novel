"""Toy data written in the EXACT file schemas of the real datasets, so the real readers are exercised.

The physics is only qualitative (CaloChallenge-like longitudinal Gamma profile). It exists
so the whole pipeline runs on a laptop without downloads. Never report results on it as physics.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter


def make_calochallenge(dirpath: str, n: int = 3000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    nz, na, nr = 45, 16, 9
    for fi, name in enumerate(["dataset_2_1.hdf5", "dataset_2_2.hdf5"]):
        E = np.exp(rng.uniform(np.log(1e3), np.log(1e6), (n, 1))).astype(np.float32)
        S = np.zeros((n, nz * na * nr), np.float32)
        for e in range(n):
            lg = np.log(E[e, 0] / 1e3)
            nh = int(200 + 60 * lg)
            a_shape = 2.0 + 0.6 * lg
            z = rng.gamma(a_shape, 1.6, nh) + rng.uniform(0, 2)
            r = np.abs(rng.normal(0, 0.8, nh)) + rng.exponential(0.6, nh) * (rng.random(nh) < 0.4) * 4
            al = rng.uniform(0, 2 * np.pi, nh)
            w = rng.lognormal(0, 0.6, nh)
            w = w / w.sum() * E[e, 0] * 0.9
            keep = (z < nz) & (r < nr)
            h, _ = np.histogramdd(np.stack([z[keep], al[keep], r[keep]], 1), bins=[nz, na, nr],
                                  range=[[0, nz], [0, 2 * np.pi], [0, nr]], weights=w[keep])
            S[e] = h.astype(np.float32).reshape(-1)  # (z, alpha, r) flattening
        with h5py.File(os.path.join(dirpath, name), "w") as f:
            f.create_dataset("showers", data=S, compression="gzip")
            f.create_dataset("incident_energies", data=E)


def make_all(root: str, n_cc: int = 3000, seed: int = 0) -> None:
    os.makedirs(root, exist_ok=True)
    make_calochallenge(root, n_cc, seed)
