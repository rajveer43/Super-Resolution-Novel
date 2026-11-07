"""Sparse cell-set tokens (Graph Denoising paper: calorimeter cells as graph nodes).

Only non-zero cells (plus an optional random sample of empty cells) become tokens. Tokens are ordered
along a Morton curve so that windowed attention behaves like local attention.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import ScaleSpec


@dataclass
class Cells:
    idx: torch.Tensor  # (B, N) dense index at `spec`
    valid: torch.Tensor  # (B, N) bool
    energy: torch.Tensor  # (B, N) raw energy (0 for padding)
    coords: torch.Tensor  # (B, N, 3)
    level: torch.Tensor  # (B, N, 2)
    layer: torch.Tensor  # (B, N) global layer id
    group: torch.Tensor  # (B, N)
    spec: ScaleSpec

    @property
    def B(self) -> int:
        return self.idx.shape[0]

    @property
    def N(self) -> int:
        return self.idx.shape[1]


def pack_mask(mask: torch.Tensor, rank: torch.Tensor, max_tokens: int | None = None,
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,T) bool mask -> (idx (B,N), valid (B,N)) with tokens sorted by `rank` within each row.
    If a row has more than max_tokens selected cells, a random subset is kept."""
    B, T = mask.shape
    b, idx = mask.nonzero(as_tuple=True)
    if b.numel() == 0:
        z = torch.zeros(B, 1, dtype=torch.long, device=mask.device)
        return z, torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
    if max_tokens is not None:
        counts = torch.bincount(b, minlength=B)
        if int(counts.max()) > max_tokens:
            pri = torch.rand(b.shape[0], device=mask.device, generator=generator)
            order = torch.argsort(b.float() * 2.0 + pri)  # group by row, random within row
            b, idx = b[order], idx[order]
            start = torch.cumsum(counts, 0) - counts
            pos = torch.arange(b.shape[0], device=mask.device) - start[b]
            keep = pos < max_tokens
            b, idx = b[keep], idx[keep]
    key = b * (T + 1) + rank[idx]
    order = torch.argsort(key)
    b, idx = b[order], idx[order]
    counts = torch.bincount(b, minlength=B)
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(b.shape[0], device=mask.device) - start[b]
    N = max(int(counts.max()), 1)
    out = torch.zeros(B, N, dtype=torch.long, device=mask.device)
    valid = torch.zeros(B, N, dtype=torch.bool, device=mask.device)
    out[b, pos] = idx
    valid[b, pos] = True
    return out, valid


def cells_from_dense(dense: torch.Tensor, spec: ScaleSpec, threshold: float = 0.0,
                     zero_frac: float = 0.0, max_tokens: int | None = None,
                     generator: torch.Generator | None = None, select_all: bool = False) -> Cells:
    """Build tokens from a dense (B,T) energy vector at resolution `spec`."""
    if select_all:
        mask = torch.ones_like(dense, dtype=torch.bool)
    else:
        mask = dense > threshold
        if zero_frac > 0:
            mask = mask | (torch.rand(dense.shape, device=dense.device, generator=generator) < zero_frac)
    idx, valid = pack_mask(mask, spec.rank(dense.device), max_tokens, generator)
    return cells_from_idx(idx, valid, dense, spec)


def cells_from_idx(idx: torch.Tensor, valid: torch.Tensor, dense: torch.Tensor | None, spec: ScaleSpec,
                   energy: torch.Tensor | None = None) -> Cells:
    info = spec.cell_info(idx)
    if energy is None:
        energy = torch.gather(dense, 1, idx) * valid
    return Cells(idx=idx, valid=valid, energy=energy, coords=info["coords"], level=info["level"],
                 layer=info["layer"], group=info["group"], spec=spec)


def scatter_to_dense(idx: torch.Tensor, values: torch.Tensor, valid: torch.Tensor, T: int) -> torch.Tensor:
    """(B,N[,C]) idx/values -> dense (B,T) with summation. Differentiable w.r.t. values."""
    B = idx.shape[0]
    idx = idx.reshape(B, -1)
    values = (values * valid).reshape(B, -1)
    out = torch.zeros(B, T, device=values.device, dtype=values.dtype)
    return out.scatter_add(1, idx, values)
