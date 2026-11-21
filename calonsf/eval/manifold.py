"""Manifold-based metrics (CaloChallenge section 8.5): precision, recall, density and coverage.

The manifold of a sample set is the union of the balls B(x_i, NND_k(x_i)) around its points, NND_k being the
distance to the k-th nearest neighbour (k = 5). With real = Geant4 reference x_i and fake = generated y_j:

    precision = 1/m sum_j 1[y_j in manifold(real)]            recall   = 1/n sum_i 1[x_i in manifold(fake)]
    density   = 1/(k m) sum_j sum_i 1[y_j in B(x_i, NND_k)]   coverage = 1/n sum_i 1[exists j: y_j in B(x_i, NND_k)]

Voxel energies are preprocessed as log10 I, with voxels below 0.1 MeV set to log10(0.1 MeV) = -1."""
from __future__ import annotations

import numpy as np
import torch


def manifold_features(shower: np.ndarray) -> np.ndarray:
    """(N, Z, A, R) MeV -> (N, Z*A*R) log10 voxel energies, floored at log10(0.1 MeV) = -1."""
    x = shower.reshape(len(shower), -1)
    return np.log10(np.maximum(x, 0.1)).astype(np.float32)


def _kth_nn_dist(x: torch.Tensor, k: int, block: int) -> torch.Tensor:
    out = []
    for s in range(0, len(x), block):
        d = torch.cdist(x[s:s + block], x)
        out.append(d.topk(k + 1, dim=1, largest=False).values[:, -1])  # [:, 0] is the point itself
    return torch.cat(out)


@torch.no_grad()
def prdc(real: np.ndarray, fake: np.ndarray, k: int = 5, device: str | torch.device = "cpu", block: int = 1024) -> dict:
    r = torch.as_tensor(real, device=device)
    f = torch.as_tensor(fake, device=device)
    n, m = len(r), len(f)
    r_nnd = _kth_nn_dist(r, k, block)
    f_nnd = _kth_nn_dist(f, k, block)
    fake_in = torch.zeros(m, dtype=torch.bool, device=r.device)  # fake point inside some real ball
    fake_cnt = torch.zeros(m, dtype=torch.float32, device=r.device)  # number of real balls containing it
    recall_hits = cover_hits = 0
    for s in range(0, n, block):
        d = torch.cdist(r[s:s + block], f)  # (b, m)
        inside = d < r_nnd[s:s + block, None]
        fake_in |= inside.any(0)
        fake_cnt += inside.sum(0)
        cover_hits += int(inside.any(1).sum())
        recall_hits += int((d < f_nnd[None, :]).any(1).sum())
    return {"precision": float(fake_in.float().mean()), "recall": recall_hits / n,
            "density": float(fake_cnt.sum() / (k * m)), "coverage": cover_hits / n}
