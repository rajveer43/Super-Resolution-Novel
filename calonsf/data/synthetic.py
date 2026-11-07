"""Toy data written in the EXACT file schemas of the real datasets, so the real readers are exercised.

The physics is only qualitative (quark jets narrower/fewer constituents than gluon jets; EM showers
narrow and early, pion showers wide and deep; CaloChallenge-like longitudinal Gamma profile). It exists
so the whole pipeline runs on a laptop without downloads. Never report results on it as physics.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter


def make_qg(path: str, n: int = 3000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 125, 125, 3), dtype=np.float32)
    y = rng.integers(0, 2, n).astype(np.float32)
    pt = rng.uniform(70, 320, n).astype(np.float32)
    m0 = np.zeros(n, dtype=np.float32)
    for e in range(n):
        g = y[e] == 1
        nc = rng.poisson(22 if g else 12) + 3
        sig = 0.16 if g else 0.10
        pos = rng.normal(0, sig, (nc, 2))
        frac = rng.dirichlet(np.full(nc, 1.0 if g else 0.6))
        ene = pt[e] * frac
        kind = rng.choice(3, nc, p=[0.6, 0.25, 0.15])  # charged, photon, neutral hadron
        pix = np.clip(((pos + 1) / 2 * 125).astype(int), 0, 124)
        trk, ecal, hcal = (np.zeros((125, 125), np.float32) for _ in range(3))
        for (a, b), k, en in zip(pix, kind, ene):
            if k == 0:
                trk[a, b] += en
                hcal[a, b] += 0.7 * en
                ecal[a, b] += 0.1 * en
            elif k == 1:
                ecal[a, b] += en
            else:
                hcal[a, b] += en
        ecal = gaussian_filter(ecal, 1.0)
        hcal = gaussian_filter(hcal, 2.5)
        img = np.stack([trk, ecal, hcal], -1)
        img[img < 1e-3 * img.max()] = 0
        X[e] = img
        w = ene / ene.sum()
        m0[e] = pt[e] * np.sqrt(max((w * (pos ** 2).sum(1)).sum() - ((w[:, None] * pos).sum(0) ** 2).sum(), 0))
    with h5py.File(path, "w") as f:
        f.create_dataset("X_jets", data=X, compression="gzip", chunks=(1, 125, 125, 3))
        f.create_dataset("y", data=y)
        f.create_dataset("pt", data=pt)
        f.create_dataset("m0", data=m0)


def _hist_layer(xy: np.ndarray, w: np.ndarray, shape: tuple[int, int], extent: float = 240.0) -> np.ndarray:
    H, W = shape
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[H, W], range=[[-extent, extent], [-extent, extent]],
                             weights=w)
    return h.astype(np.float32)


def make_calogan(dirpath: str, n: int = 2000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    specs = {
        "eplus": dict(frac=[2, 12, 2], dep=0.95, sig=12.0),
        "gamma": dict(frac=[1, 12, 3], dep=0.95, sig=12.0),
        "piplus": dict(frac=[1, 4, 6], dep=0.60, sig=35.0),
    }
    shapes = [(3, 96), (12, 12), (12, 6)]
    for name, s in specs.items():
        E = rng.uniform(1, 100, (n, 1)).astype(np.float32)
        L = [np.zeros((n, *sh), np.float32) for sh in shapes]
        ov = np.zeros((n, 3), np.float32)
        for e in range(n):
            fr = rng.dirichlet(s["frac"]) * s["dep"] * rng.uniform(0.85, 1.0)
            c = rng.normal(0, 5.0, 2)
            for k, sh in enumerate(shapes):
                nh = 300
                core = rng.random(nh) < 0.9
                sig = s["sig"] * (1 + 0.5 * k) * np.where(core, 1.0, 4.0)
                xy = c + rng.normal(0, 1, (nh, 2)) * sig[:, None]
                w = rng.lognormal(0, 0.5, nh)
                w = w / w.sum() * E[e, 0] * 1000 * fr[k]
                inside = (np.abs(xy) < 240).all(1)
                ov[e, k] = w[~inside].sum()
                L[k][e] = _hist_layer(xy[inside], w[inside], sh)
        with h5py.File(os.path.join(dirpath, f"{name}.hdf5"), "w") as f:
            f.create_dataset("energy", data=E)
            for k in range(3):
                f.create_dataset(f"layer_{k}", data=L[k])
            f.create_dataset("overflow", data=ov)


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


def make_all(root: str, n_qg: int = 3000, n_calogan: int = 2000, n_cc: int = 3000, seed: int = 0) -> None:
    os.makedirs(root, exist_ok=True)
    make_qg(os.path.join(root, "qg.h5"), n_qg, seed)
    make_calogan(root, n_calogan, seed + 1)
    make_calochallenge(root, n_cc, seed + 2)
