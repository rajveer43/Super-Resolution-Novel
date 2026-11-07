"""Energy-conserving grid operations. All grids are (B, n_layers, H, W) energy tensors."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sum_pool(x: torch.Tensor, fu: int, fv: int) -> torch.Tensor:
    """Sum-pool (NOT average / bicubic) so the LR total energy equals the HR total energy."""
    if fu == 1 and fv == 1:
        return x
    B, n, H, W = x.shape
    assert H % fu == 0 and W % fv == 0, (x.shape, fu, fv)
    return x.reshape(B, n, H // fu, fu, W // fv, fv).sum(dim=(3, 5))


def pool_grids(grids: dict, factors: dict) -> dict:
    return {g: sum_pool(x, *factors[g]) for g, x in grids.items()}


def repeat_upsample(x: torch.Tensor, fu: int, fv: int, conserve: bool = True) -> torch.Tensor:
    """Nearest-neighbour upsampling ("upsampling by repetition", Graph Denoising paper).
    With conserve=True each child gets parent/(fu*fv) so the total energy is unchanged."""
    y = x.repeat_interleave(fu, dim=-2).repeat_interleave(fv, dim=-1)
    return y / (fu * fv) if conserve else y


def overlap_matrix(n_in: int, n_out: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """A[o, i] = fraction of input cell i that falls in output cell o (columns sum to 1)."""
    # built in float64 on CPU (exact), then moved (MPS has no float64)
    e_in = torch.linspace(0, 1, n_in + 1, dtype=torch.float64)
    e_out = torch.linspace(0, 1, n_out + 1, dtype=torch.float64)
    lo = torch.maximum(e_out[:-1, None], e_in[None, :-1])
    hi = torch.minimum(e_out[1:, None], e_in[None, 1:])
    ov = (hi - lo).clamp(min=0) / (e_in[1:] - e_in[:-1])[None, :]
    return ov.to(device=device, dtype=dtype)


def resample_energy(x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Exact area-overlap resampling to an arbitrary (also non-integer-factor) shape.
    Conserves total energy exactly; used to deliver e.g. 125x125 or 300x300 outputs."""
    H, W = x.shape[-2:]
    Ho, Wo = out_hw
    if (H, W) == (Ho, Wo):
        return x
    Au = overlap_matrix(H, Ho, x.device, x.dtype)
    Av = overlap_matrix(W, Wo, x.device, x.dtype)
    return torch.einsum("oh,bnhw,pw->bnop", Au, x, Av)


def shift_energy(x: torch.Tensor, shift: float, axis: int) -> torch.Tensor:
    """Shift a grid by a fractional number of cells along `axis` (-2 or -1); energy leaving the
    grid is lost. Used for the two-shower overlay test."""
    n = x.shape[axis]
    pos = torch.arange(n, device=x.device, dtype=x.dtype)
    src_lo, src_hi = pos, pos + 1
    dst_lo, dst_hi = pos[:, None] + 0 * pos[None], pos[:, None] + 1
    # A[o, i] = overlap between shifted input cell i and output cell o
    lo = torch.maximum(dst_lo, (src_lo + shift)[None, :])
    hi = torch.minimum(dst_hi, (src_hi + shift)[None, :])
    A = (hi - lo).clamp(min=0)
    if axis in (-2, 2):
        return torch.einsum("oh,bnhw->bnow", A, x)
    return torch.einsum("ow,bnhw->bnho", A, x)


def pad_to(x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Zero-pad (B,n,h,w) to (H,W), roughly centred. Returns padding (left,right,top,bottom)."""
    h, w = x.shape[-2:]
    ph, pw = H - h, W - w
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
    return F.pad(x, pad), pad


def crop_pad(x: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    l, r, t, b = pad
    H, W = x.shape[-2:]
    return x[..., t: H - b, l: W - r]
