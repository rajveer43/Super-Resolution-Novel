"""Interpolation, EDSR-lite, RCAN-lite, SwinIR-lite and LIIF baselines."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.ops import repeat_upsample
from .base import SRBaseline, pad_mult


class Nearest(SRBaseline):
    trainable = False
    name = "nearest"

    def predict_group(self, g, lr_t, batch):
        lr = batch["lr"][g]
        fu, fv = self.factors[g]
        from .base import to_t
        return to_t(repeat_upsample(lr, fu, fv, conserve=True), self.es)


class Bicubic(SRBaseline):
    trainable = False
    name = "bicubic"

    def predict_group(self, g, lr_t, batch):
        lr = batch["lr"][g]
        fu, fv = self.factors[g]
        H, W = lr.shape[-2:]
        y = F.interpolate(lr / (fu * fv), size=(H * fu, W * fv), mode="bicubic", align_corners=False).clamp(min=0)
        from .base import to_t
        return to_t(y, self.es)


class ResBlock(nn.Module):
    def __init__(self, c: int, ca: bool = False):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1))
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(4, c // 8), 1), nn.ReLU(),
                                nn.Conv2d(max(4, c // 8), c, 1), nn.Sigmoid()) if ca else None

    def forward(self, x):
        r = self.body(x)
        if self.ca is not None:
            r = r * self.ca(r)  # channel attention (RCAN)
        return x + 0.2 * r


class _ConvSR(nn.Module):
    def __init__(self, n: int, c: int, blocks: int, fu: int, fv: int, ca: bool):
        super().__init__()
        self.fu, self.fv = fu, fv
        self.head = nn.Conv2d(n, c, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(c, ca) for _ in range(blocks)], nn.Conv2d(c, c, 3, padding=1))
        self.tail = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, n, 3, padding=1))

    def features(self, x):
        h = self.head(x)
        return h + self.body(h)

    def forward(self, x):
        f = self.features(x)
        up = f.repeat_interleave(self.fu, -2).repeat_interleave(self.fv, -1)
        return self.tail(up) + x.repeat_interleave(self.fu, -2).repeat_interleave(self.fv, -1)


class EDSR(SRBaseline):
    name = "edsr"
    ca = False

    def __init__(self, cfg, geom, factor):
        super().__init__(cfg, geom, factor)
        c, b = int(cfg.baseline.channels), int(cfg.baseline.blocks)
        self.nets = nn.ModuleDict({g: _ConvSR(geom.group(g).n_layers, c, b, *self.factors[g], self.ca)
                                   for g in geom.group_names})

    def predict_group(self, g, lr_t, batch):
        return self.nets[g](lr_t)


class RCAN(EDSR):
    name = "rcan"
    ca = True


class WindowAttn(nn.Module):
    """Swin block: (shifted) window self-attention + MLP on a feature map."""

    def __init__(self, c: int, heads: int, ws: int, shift: bool):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.n1, self.n2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(c, 2 * c), nn.GELU(), nn.Linear(2 * c, c))

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.ws
        xp, (h0, w0) = pad_mult(x, ws)
        Hp, Wp = xp.shape[-2:]
        s = ws // 2 if self.shift else 0
        if s:
            xp = torch.roll(xp, (-s, -s), (-2, -1))
        t = xp.permute(0, 2, 3, 1).reshape(B, Hp // ws, ws, Wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
        t = t.reshape(-1, ws * ws, C)
        h = self.n1(t)
        t = t + self.attn(h, h, h, need_weights=False)[0]
        t = t + self.mlp(self.n2(t))
        t = t.reshape(B, Hp // ws, Wp // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        y = t.permute(0, 3, 1, 2)
        if s:
            y = torch.roll(y, (s, s), (-2, -1))
        return y[..., :h0, :w0]


class _Swin(nn.Module):
    def __init__(self, n, c, blocks, fu, fv):
        super().__init__()
        self.fu, self.fv = fu, fv
        self.head = nn.Conv2d(n, c, 3, padding=1)
        self.body = nn.ModuleList([WindowAttn(c, 4, 8, i % 2 == 1) for i in range(blocks)])
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.tail = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, n, 3, padding=1))

    def forward(self, x):
        h = self.head(x)
        f = h
        for b in self.body:
            f = b(f)
        f = h + self.conv(f)
        up = f.repeat_interleave(self.fu, -2).repeat_interleave(self.fv, -1)
        return self.tail(up) + x.repeat_interleave(self.fu, -2).repeat_interleave(self.fv, -1)


class SwinIR(SRBaseline):
    name = "swinir"

    def __init__(self, cfg, geom, factor):
        super().__init__(cfg, geom, factor)
        c, b = int(cfg.baseline.channels), int(cfg.baseline.blocks)
        c = c - c % 4
        self.nets = nn.ModuleDict({g: _Swin(geom.group(g).n_layers, c, b, *self.factors[g]) for g in geom.group_names})

    def predict_group(self, g, lr_t, batch):
        return self.nets[g](lr_t)


class _LIIF(nn.Module):
    """Local implicit image function: MLP(feature of nearest LR cell (3x3 unfolded), rel. coord, cell size)."""

    def __init__(self, n, c, blocks):
        super().__init__()
        self.enc = _ConvSR(n, c, blocks, 1, 1, False)
        self.mlp = nn.Sequential(nn.Linear(9 * c + 4, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, n))

    def forward(self, x, out_hw):
        B, n, h, w = x.shape
        f = self.enc.features(x)
        c = f.shape[1]
        fu = F.unfold(f, 3, padding=1).view(B, 9 * c, h, w)
        Ho, Wo = out_hw
        yi = (torch.arange(Ho, device=x.device).float() + 0.5) / Ho * h
        xj = (torch.arange(Wo, device=x.device).float() + 0.5) / Wo * w
        iy = yi.floor().long().clamp(0, h - 1)
        ix = xj.floor().long().clamp(0, w - 1)
        feat = fu[:, :, iy][:, :, :, ix]  # (B, 9c, Ho, Wo)
        ry = (yi - iy.float() - 0.5) * 2
        rx = (xj - ix.float() - 0.5) * 2
        rel = torch.stack(torch.meshgrid(ry, rx, indexing="ij"), 0)
        cell = torch.tensor([2 * h / Ho, 2 * w / Wo], device=x.device).view(2, 1, 1).expand(2, Ho, Wo)
        q = torch.cat([feat, rel.unsqueeze(0).expand(B, -1, -1, -1), cell.unsqueeze(0).expand(B, -1, -1, -1)], 1)
        out = self.mlp(q.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        base = x[:, :, iy][:, :, :, ix]
        return out + base


class LIIF(SRBaseline):
    name = "liif"

    def __init__(self, cfg, geom, factor):
        super().__init__(cfg, geom, factor)
        c, b = int(cfg.baseline.channels) // 2, max(2, int(cfg.baseline.blocks) // 2)
        self.nets = nn.ModuleDict({g: _LIIF(geom.group(g).n_layers, c, b) for g in geom.group_names})

    def predict_group(self, g, lr_t, batch, out_hw=None):
        fu, fv = self.factors[g]
        return self.nets[g](lr_t, out_hw or (lr_t.shape[-2] * fu, lr_t.shape[-1] * fv))
