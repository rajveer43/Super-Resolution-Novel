"""Common interface for grid baselines. Each layer group gets its own network (channels = layers).
All baselines work in t = log1p(E / E_scale) space and train for ONE fixed LR factor."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.geometry import Geometry, lr_factors_for
from ...data.ops import repeat_upsample, sum_pool


def to_t(x: torch.Tensor, es: float) -> torch.Tensor:
    return torch.log1p(x.clamp(min=0) / es)


def from_t(y: torch.Tensor, es: float) -> torch.Tensor:
    return es * torch.expm1(F.relu(y))


def project_energy(hr: torch.Tensor, lr: torch.Tensor, fu: int, fv: int) -> torch.Tensor:
    """Rescale each LR cell's children so they sum to the LR cell energy (uniform if all zero)."""
    s = sum_pool(hr, fu, fv)
    scale = torch.where(s > 0, lr / s.clamp(min=1e-20), torch.zeros_like(s))
    out = hr * repeat_upsample(scale, fu, fv, conserve=False)
    empty = (s <= 0) & (lr > 0)
    uni = repeat_upsample(torch.where(empty, lr, torch.zeros_like(lr)), fu, fv, conserve=True)
    return out + uni


class SRBaseline(nn.Module):
    trainable = True
    name = "base"

    def __init__(self, cfg, geom: Geometry, factor: int):
        super().__init__()
        self.cfg, self.geom = cfg, geom
        self.factor = factor
        self.factors = lr_factors_for(geom, factor, cfg.data.get("factor_overrides", None))
        self.es = geom.energy_scale
        self.project = bool(cfg.baseline.get("project_energy", True))

    def _check(self, batch):
        assert batch["factors"] == self.factors or all(tuple(batch["factors"][g]) == tuple(self.factors[g])
                                                       for g in self.factors), "baseline trained for another factor"

    def predict_group(self, g: str, lr_t: torch.Tensor, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    def loss_group(self, g: str, lr_t: torch.Tensor, hr_t: torch.Tensor, batch: dict) -> torch.Tensor:
        return (self.predict_group(g, lr_t, batch) - hr_t).abs().mean()

    def loss(self, batch: dict) -> dict:
        self._check(batch)
        tot = 0.0
        for g in self.geom.group_names:
            tot = tot + self.loss_group(g, to_t(batch["lr"][g], self.es), to_t(batch["hr"][g], self.es), batch)
        return {"loss": tot / len(self.geom.group_names)}

    @torch.no_grad()
    def predict(self, batch: dict) -> dict:
        self._check(batch)
        out = {}
        for g in self.geom.group_names:
            y = from_t(self.predict_group(g, to_t(batch["lr"][g], self.es), batch), self.es)
            if self.project:
                y = project_energy(y, batch["lr"][g], *self.factors[g])
            out[g] = y
        return out


def pad_mult(x: torch.Tensor, m: int):
    H, W = x.shape[-2:]
    ph, pw = (-H) % m, (-W) % m
    return F.pad(x, (0, pw, 0, ph)), (H, W)
