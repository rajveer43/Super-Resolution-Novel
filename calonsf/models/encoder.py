"""Cell tokenizer + resolution-agnostic transformer encoder + physics conditioning."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..data.tokens import Cells
from .blocks import Block, ContinuousRoPE, fourier


def energy_features(energy: torch.Tensor, total: torch.Tensor, energy_scale: float) -> torch.Tensor:
    """[log1p(E/E_scale), log(E/E_event)/10] - both scale-free w.r.t. grid resolution."""
    et = torch.log1p(energy.clamp(min=0) / energy_scale)
    lf = (torch.log(energy.clamp(min=0) / total.clamp(min=1e-12).unsqueeze(-1) + 1e-12).clamp(-30, 0)) / 10.0
    return torch.stack([et, lf], dim=-1)


def make_rope(cfg_m, geom) -> ContinuousRoPE | None:
    if cfg_m.get("pos_encoding", "rope") != "rope":
        return None
    r = cfg_m.get("rope", {})
    return ContinuousRoPE(cfg_m.d_model // cfg_m.heads, geom.periodic, float(r.get("max_freq", 256)),
                          int(r.get("periodic_max_freq", 64)), float(r.get("depth_max_freq", 16)))


class CellEmbed(nn.Module):
    def __init__(self, d: int, n_layers: int, energy_scale: float, extra_in: int = 0,
                 abs_level: bool = True, absolute_pos: bool = False):
        super().__init__()
        self.energy_scale = energy_scale
        self.abs_level = abs_level
        self.absolute_pos = absolute_pos
        n_in = 4 + extra_in + (3 * 16 if absolute_pos else 0)
        self.mlp = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))
        self.layer = nn.Embedding(n_layers, d)

    def forward(self, cells: Cells, total: torch.Tensor, extra: torch.Tensor | None = None,
                energy: torch.Tensor | None = None) -> torch.Tensor:
        e = cells.energy if energy is None else energy
        feats = [energy_features(e, total, self.energy_scale),
                 cells.level if self.abs_level else torch.zeros_like(cells.level)]
        if extra is not None:
            feats.append(extra)
        if self.absolute_pos:  # ablation: learned absolute (Fourier) positions instead of RoPE
            c = cells.coords.clone()
            feats.append(fourier(c, 8, 64.0))
        x = self.mlp(torch.cat(feats, dim=-1)) + self.layer(cells.layer)
        return x * cells.valid.unsqueeze(-1)


class Conditioner(nn.Module):
    """Incident-energy / particle-type conditioning vector (B, d). Never uses the QG class label."""

    def __init__(self, d: int, cfg_c, n_particles: int = 3):
        super().__init__()
        self.use_e = bool(cfg_c.get("incident_energy", False))
        self.use_p = bool(cfg_c.get("particle", False))
        self.lo = math.log(float(cfg_c.get("e_min", 1.0)))
        self.hi = math.log(float(cfg_c.get("e_max", 100.0)))
        self.base = nn.Parameter(torch.zeros(d))
        if self.use_e:
            self.e_mlp = nn.Sequential(nn.Linear(16, d), nn.SiLU(), nn.Linear(d, d))
        if self.use_p:
            self.p_emb = nn.Embedding(n_particles, d)

    def forward(self, batch: dict, B: int) -> torch.Tensor:
        c = self.base.unsqueeze(0).expand(B, -1)
        if self.use_e:
            e = batch["e_inc"].clamp(min=1e-6)
            x = (torch.log(e) - self.lo) / (self.hi - self.lo) * 2 - 1
            c = c + self.e_mlp(fourier(x.unsqueeze(-1), 8, 8.0))
        if self.use_p:
            c = c + self.p_emb(batch["particle"].clamp(min=0))
        return c


class CellEncoder(nn.Module):
    """Transformer over LR cell tokens. Pretrained with JEPA, then reused by the SR model."""

    def __init__(self, cfg_m, geom):
        super().__init__()
        d = int(cfg_m.d_model)
        self.d = d
        self.window = cfg_m.get("window", 1024)
        self.rope = make_rope(cfg_m, geom)
        self.embed = CellEmbed(d, geom.n_layers_total, geom.energy_scale,
                               abs_level=bool(cfg_m.get("abs_level_feature", True)),
                               absolute_pos=self.rope is None)
        self.blocks = nn.ModuleList([Block(d, int(cfg_m.heads), self.rope, float(cfg_m.get("mlp_ratio", 4)),
                                           cond_dim=d, dropout=float(cfg_m.get("dropout", 0.0)))
                                     for _ in range(int(cfg_m.enc_layers))])
        self.norm = nn.LayerNorm(d)

    def forward(self, cells: Cells, total: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor | None = None,
                energy: torch.Tensor | None = None) -> torch.Tensor:
        m = cells.valid if mask is None else mask
        x = self.embed(cells, total, energy=energy)
        for blk in self.blocks:
            x = blk(x, cells.coords, m, cond=cond, window=self.window)
        return self.norm(x) * m.unsqueeze(-1)
