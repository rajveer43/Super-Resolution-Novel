"""Lazy HDF5 reader for CaloChallenge Dataset 2. The reader returns, per event,
    groups: {group_name: float32 array (n_layers, H, W)}  energies in native units
    meta:   {"label": int, "e_inc": float, "particle": int, ...}

Every dataset-specific assumption is asserted or logged.
"""
from __future__ import annotations

import os

import h5py
import numpy as np

from ..utils.misc import get_logger

log = get_logger()


class _H5:
    """Re-open the file lazily in each process (safe with DataLoader workers)."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Set data.files in the config "
                                    f"(or run scripts/make_synthetic.py for the toy data).")
        self.path = path
        self._f = None
        self._pid = None

    @property
    def f(self) -> h5py.File:
        if self._f is None or self._pid != os.getpid():
            self._f = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._f


class CaloChallengeReader:
    """CaloChallenge Dataset 2: showers (N, 6480) MeV, incident_energies (N, 1) MeV.
    flatten_order 'zar' means voxel = z*(16*9) + alpha*9 + r. Run `verify_flatten_order` to check."""

    def __init__(self, files: list[str], flatten_order: str = "zar", n_z: int = 45, n_a: int = 16, n_r: int = 9):
        self.h5 = [_H5(p) for p in files]
        self.lengths = [h.f["showers"].shape[0] for h in self.h5]
        self.cum = np.concatenate([[0], np.cumsum(self.lengths)])
        self.order = flatten_order
        self.nz, self.na, self.nr = n_z, n_a, n_r
        assert self.h5[0].f["showers"].shape[1] == n_z * n_a * n_r
        assert flatten_order in ("zar", "zra"), flatten_order
        log.info(f"[CaloChallenge] {self.lengths} events; flatten order '{flatten_order}' "
                 f"-> (z={n_z}, alpha={n_a}, r={n_r}); energies in MeV")

    def __len__(self) -> int:
        return int(self.cum[-1])

    def _reshape(self, v: np.ndarray) -> np.ndarray:
        if self.order == "zar":
            return v.reshape(self.nz, self.na, self.nr)
        return v.reshape(self.nz, self.nr, self.na).transpose(0, 2, 1)

    def get(self, i: int):
        fi = int(np.searchsorted(self.cum, i, side="right") - 1)
        j = i - int(self.cum[fi])
        f = self.h5[fi].f
        x = np.clip(np.asarray(f["showers"][j], dtype=np.float32), 0, None)
        e = float(np.asarray(f["incident_energies"][j]).reshape(-1)[0])
        return {"shower": self._reshape(x)}, {"label": -1, "particle": -1, "e_inc": e, "file": fi}


def verify_flatten_order(reader: CaloChallengeReader, n: int = 500) -> dict:
    """Heuristic check: for EM showers the mean radial profile decreases with r, and the mean
    angular profile is ~flat in alpha. Returns diagnostics and logs a warning if it looks wrong."""
    idx = np.linspace(0, len(reader) - 1, min(n, len(reader))).astype(int)
    acc = np.zeros((reader.nz, reader.na, reader.nr))
    for i in idx:
        acc += reader.get(int(i))[0]["shower"]
    radial = acc.sum(axis=(0, 1))
    angular = acc.sum(axis=(0, 2))
    dec_frac = float(np.mean(np.diff(radial) < 0))
    ratio = float(radial[0] / max(radial[-1], 1e-12))
    ang_cv = float(angular.std() / (angular.mean() + 1e-12))
    # wrong orders mix r into the alpha axis: the "radial" profile becomes flat-ish (small core/edge ratio)
    ok = dec_frac >= 0.75 and ratio >= 5.0 and ang_cv < 0.5
    msg = f"[CaloChallenge] flatten-order check: radial-decreasing={dec_frac:.2f}, core/edge={ratio:.1f}, " \
          f"alpha-CV={ang_cv:.2f} -> " \
          f"{'OK' if ok else 'SUSPICIOUS: try data.flatten_order=zra and check the official binning XML'}"
    (log.info if ok else log.warning)(msg)
    return dict(radial=radial.tolist(), angular=angular.tolist(), ok=ok)


def make_reader(cfg):
    d = cfg.data
    if d.name == "calochallenge":
        return CaloChallengeReader(list(d.files), d.get("flatten_order", "zar"))
    raise ValueError(d.name)


def make_splits(cfg, reader) -> dict[str, np.ndarray]:
    """Fixed-seed splits. For CaloChallenge with 2 files, file 2 is the official evaluation set."""
    d = cfg.data
    rng = np.random.default_rng(int(cfg.seed))
    n = len(reader)
    if d.name == "calochallenge" and len(reader.lengths) >= 2 and d.get("official_split", True):
        tr_all = np.arange(reader.lengths[0])
        te = np.arange(reader.lengths[0], n)
        rng.shuffle(tr_all)
        nv = int(len(tr_all) * float(d.split.val) / (float(d.split.train) + float(d.split.val)))
        out = {"train": tr_all[nv:], "val": tr_all[:nv], "test": te}
    else:
        perm = rng.permutation(n)
        ntr = int(n * float(d.split.train))
        nv = int(n * float(d.split.val))
        out = {"train": perm[:ntr], "val": perm[ntr:ntr + nv], "test": perm[ntr + nv:]}
    me = d.get("max_events", None)
    if me:
        frac = {"train": 1.0, "val": 0.15, "test": 0.15}
        out = {k: v[: max(1, int(int(me) * frac[k]))] for k, v in out.items()}
    log.info("splits: " + ", ".join(f"{k}={len(v)}" for k, v in out.items()))
    return out
