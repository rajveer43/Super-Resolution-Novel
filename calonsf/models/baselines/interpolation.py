"""Non-trainable reference upsamplers. Both conserve every LR cell's energy exactly."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ...data.geometry import Geometry, lr_factors_for
from ...data.ops import repeat_upsample, sum_pool


def project_energy(hr: torch.Tensor, lr: torch.Tensor, fu: int, fv: int) -> torch.Tensor:
    """Rescale each LR cell's children so they sum to the LR cell energy (uniform if all zero)."""
    s = sum_pool(hr, fu, fv)
    scale = torch.where(s > 0, lr / s.clamp(min=1e-20), torch.zeros_like(s))
    out = hr * repeat_upsample(scale, fu, fv, conserve=False)
    empty = (s <= 0) & (lr > 0)
    uni = repeat_upsample(torch.where(empty, lr, torch.zeros_like(lr)), fu, fv, conserve=True)
    return out + uni


class SRBaseline:
    """predict(batch) -> {group: HR energy grid}, for one fixed LR factor."""
    name = "base"

    def __init__(self, cfg, geom: Geometry, factor: int):
        self.geom = geom
        self.factor = factor
        self.factors = lr_factors_for(geom, factor, cfg.data.get("factor_overrides", None))

    def predict_group(self, g: str, lr: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, batch: dict) -> dict:
        out = {}
        for g in self.geom.group_names:
            lr = batch["lr"][g]
            fu, fv = self.factors[g]
            out[g] = project_energy(self.predict_group(g, lr), lr, fu, fv)
        return out


class Nearest(SRBaseline):
    """Upsampling by repetition: each LR cell is copied into its fu x fv children, divided by fu*fv."""
    name = "nearest"

    def predict_group(self, g, lr):
        fu, fv = self.factors[g]
        return repeat_upsample(lr, fu, fv, conserve=True)


class Bicubic(SRBaseline):
    name = "bicubic"

    def predict_group(self, g, lr):
        fu, fv = self.factors[g]
        H, W = lr.shape[-2:]
        return F.interpolate(lr / (fu * fv), size=(H * fu, W * fv), mode="bicubic", align_corners=False).clamp(min=0)
