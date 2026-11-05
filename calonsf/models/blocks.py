"""Transformer building blocks with continuous axial RoPE.

RoPE: V-JEPA 2 splits the head dim into (t, h, w) segments and rotates each with its axis index.
We do the same with CONTINUOUS cell coordinates (depth, u, v) instead of integer patch indices,
so the same weights apply to any grid resolution. Periodic axes (CaloChallenge alpha) use integer
frequencies, which makes attention exactly invariant to alpha -> alpha + 2pi.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _even(n: int) -> int:
    return max(2, n - n % 2)


class ContinuousRoPE(nn.Module):
    def __init__(self, head_dim: int, periodic=(False, False), max_freq: float = 256.0,
                 periodic_max_freq: int = 64, depth_max_freq: float = 16.0):
        super().__init__()
        d_depth = _even(head_dim // 4)
        d_rest = head_dim - d_depth
        d_u = _even(d_rest // 2)
        d_v = head_dim - d_depth - d_u
        assert d_v % 2 == 0 and d_v > 0, (head_dim, d_depth, d_u, d_v)
        self.dims = (d_depth, d_u, d_v)
        freqs = [math.pi * torch.logspace(math.log10(0.5), math.log10(depth_max_freq), d_depth // 2)]
        for d, per in ((d_u, periodic[0]), (d_v, periodic[1])):
            n = d // 2
            if per:
                f = torch.round(torch.logspace(0, math.log10(periodic_max_freq), n)).clamp(min=1)
            else:
                f = math.pi * torch.logspace(math.log10(0.5), math.log10(max_freq), n)
            freqs.append(f)
        self.register_buffer("f_depth", freqs[0], persistent=False)
        self.register_buffer("f_u", freqs[1], persistent=False)
        self.register_buffer("f_v", freqs[2], persistent=False)

    def cos_sin(self, coords: torch.Tensor):
        """coords (B, N, 3) -> cos, sin of shape (B, 1, N, head_dim)."""
        ang = torch.cat([coords[..., 0:1] * self.f_depth, coords[..., 1:2] * self.f_u,
                         coords[..., 2:3] * self.f_v], dim=-1)
        ang = torch.cat([ang, ang], dim=-1)
        return ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)

    @staticmethod
    def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
        return x * cos + rot * sin


def _sdpa(q, k, v, key_mask: torch.Tensor) -> torch.Tensor:
    """q (B,h,Nq,hd), key_mask (B,Nk) bool. Rows with no valid key get a dummy key (output unused)."""
    empty = ~key_mask.any(dim=-1, keepdim=True)
    first = torch.zeros_like(key_mask)
    first[:, 0] = True
    km = key_mask | (empty & first)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=km[:, None, None, :])


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, rope: ContinuousRoPE | None, cross: bool = False):
        super().__init__()
        assert d % heads == 0
        self.h, self.hd = heads, d // heads
        self.rope = rope
        self.cross = cross
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def _split(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.hd).transpose(1, 2)

    def _full(self, x, coords, ctx, ctx_coords, ctx_mask):
        q = self._split(self.q(x))
        k, v = self.kv(ctx).chunk(2, dim=-1)
        k, v = self._split(k), self._split(v)
        if self.rope is not None:
            cq, sq = self.rope.cos_sin(coords)
            ck, sk = self.rope.cos_sin(ctx_coords)
            q, k = ContinuousRoPE.apply(q, cq, sq), ContinuousRoPE.apply(k, ck, sk)
        out = _sdpa(q, k, v, ctx_mask)
        B, _, N, _ = out.shape
        return self.o(out.transpose(1, 2).reshape(B, N, -1))

    def forward(self, x, coords, mask, ctx=None, ctx_coords=None, ctx_mask=None, window: int | None = None,
                q_chunk: int = 8192):
        if self.cross:
            outs = [self._full(x[:, s:s + q_chunk], coords[:, s:s + q_chunk], ctx, ctx_coords, ctx_mask)
                    for s in range(0, x.shape[1], q_chunk)]
            return torch.cat(outs, dim=1)
        B, N, D = x.shape
        if window is None or N <= window:
            return self._full(x, coords, x, coords, mask)
        # local attention inside Morton-ordered windows
        nw = math.ceil(N / window)
        pad = nw * window - N
        xp = F.pad(x, (0, 0, 0, pad))
        cp = F.pad(coords, (0, 0, 0, pad))
        mp = F.pad(mask, (0, pad), value=False)
        xw = xp.reshape(B * nw, window, D)
        cw = cp.reshape(B * nw, window, 3)
        mw = mp.reshape(B * nw, window)
        out = []
        chunk = max(1, (1 << 16) // window)  # windows processed per call (memory)
        for s in range(0, B * nw, chunk):
            out.append(self._full(xw[s:s + chunk], cw[s:s + chunk], xw[s:s + chunk], cw[s:s + chunk],
                                  mw[s:s + chunk]))
        return torch.cat(out, 0).reshape(B, nw * window, D)[:, :N]


class Block(nn.Module):
    """Pre-LN transformer block: self-attn [+ cross-attn] + MLP, with optional AdaLN-Zero conditioning."""

    def __init__(self, d: int, heads: int, rope: ContinuousRoPE | None, mlp_ratio: float = 4.0,
                 cross: bool = False, cond_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.cond = cond_dim is not None
        self.has_cross = cross
        aff = not self.cond
        self.n1 = nn.LayerNorm(d, elementwise_affine=aff)
        self.attn = Attention(d, heads, rope)
        if cross:
            self.nc = nn.LayerNorm(d, elementwise_affine=aff)
            self.xattn = Attention(d, heads, rope, cross=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=aff)
        hid = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Dropout(dropout), nn.Linear(hid, d))
        self.drop = nn.Dropout(dropout)
        if self.cond:
            n_mod = 9 if cross else 6
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, n_mod * d))
            nn.init.zeros_(self.ada[1].weight)
            nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, coords, mask, ctx=None, ctx_coords=None, ctx_mask=None, cond=None, window=None):
        if self.cond:
            assert cond is not None
            mods = self.ada(cond).unsqueeze(1).chunk(9 if self.has_cross else 6, dim=-1)
        else:
            mods = None

        def mod(h, i):
            if mods is None:
                return h, 1.0
            sh, sc, g = mods[3 * i], mods[3 * i + 1], mods[3 * i + 2]
            return h * (1 + sc) + sh, (1.0 + g)

        h, g = mod(self.n1(x), 0)
        x = x + g * self.drop(self.attn(h, coords, mask, window=window))
        i = 1
        if self.has_cross:
            h, g = mod(self.nc(x), 1)
            x = x + g * self.drop(self.xattn(h, coords, mask, ctx=ctx, ctx_coords=ctx_coords, ctx_mask=ctx_mask))
            i = 2
        h, g = mod(self.n2(x), i)
        x = x + g * self.drop(self.mlp(h))
        return x * mask.unsqueeze(-1)


def fourier(x: torch.Tensor, n: int = 8, max_freq: float = 64.0) -> torch.Tensor:
    f = torch.logspace(0, math.log10(max_freq), n, device=x.device) * math.pi
    a = x.unsqueeze(-1) * f
    return torch.cat([a.sin(), a.cos()], dim=-1).flatten(-2)


class TimeEmbed(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(32, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = 16
        f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * 1000.0 * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=-1))
