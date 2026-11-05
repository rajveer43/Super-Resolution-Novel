"""JEPA pretraining of the cell encoder (V-JEPA 2, Sec. 2.1), adapted to calorimeter cell sets.

Objective: L1( P(context features, mask tokens), sg(EMA-encoder(full view)) ) on masked tokens only.
Two prediction targets share one predictor pass:
  1. in-scale  : multi-block-masked LR cells, target = EMA encoder on the full LR view;
  2. cross-scale (new): HR cells at HR coordinates, target = EMA encoder on the HR view.
     The context is LR, so the encoder learns what finer structure looks like in feature space,
     without any pixel reconstruction.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.geometry import Geometry, TWO_PI
from ..data.tokens import Cells, cells_from_dense
from .blocks import Block, ContinuousRoPE
from .encoder import CellEncoder, Conditioner


def multiblock_mask(cells: Cells, n_blocks: int, scale: tuple[float, float], periodic, gen=None) -> torch.Tensor:
    """Spatial multi-block mask over (u, v), shared across depth (masks whole towers)."""
    B, N = cells.idx.shape
    dev = cells.idx.device
    valid = cells.valid
    mask = torch.zeros(B, N, dtype=torch.bool, device=dev)
    u, v = cells.coords[..., 1], cells.coords[..., 2]
    rng_u = TWO_PI if periodic[0] else 2.0
    rng_v = TWO_PI if periodic[1] else 2.0
    nvalid = valid.sum(1).clamp(min=1)
    for _ in range(n_blocks):
        pick = (torch.rand(B, device=dev, generator=gen) * nvalid).long().clamp(max=N - 1)
        # pick-th valid token per row (valid tokens are packed first)
        cu = u.gather(1, pick[:, None])
        cv = v.gather(1, pick[:, None])
        hu = (scale[0] + (scale[1] - scale[0]) * torch.rand(B, 1, device=dev, generator=gen)) * rng_u / 2
        hv = (scale[0] + (scale[1] - scale[0]) * torch.rand(B, 1, device=dev, generator=gen)) * rng_v / 2
        du = (u - cu).abs()
        dv = (v - cv).abs()
        if periodic[0]:
            du = torch.minimum(du, TWO_PI - du)
        if periodic[1]:
            dv = torch.minimum(dv, TWO_PI - dv)
        mask |= (du <= hu) & (dv <= hv)
    mask &= valid
    # guarantee >= 1 masked and >= 1 context token per row
    first = torch.zeros_like(mask)
    first[:, 0] = True
    none = ~mask.any(1, keepdim=True)
    mask = mask | (none & first & valid)
    allm = (mask == valid).all(1, keepdim=True)
    mask = mask & ~(allm & first)
    return mask


class Predictor(nn.Module):
    def __init__(self, d: int, dp: int, heads: int, layers: int, n_layers_total: int, rope_cfg, periodic):
        super().__init__()
        r = rope_cfg or {}
        self.rope = ContinuousRoPE(dp // heads, periodic, float(r.get("max_freq", 256)),
                                   int(r.get("periodic_max_freq", 64)), float(r.get("depth_max_freq", 16)))
        self.inp = nn.Linear(d, dp)
        self.mask_token = nn.Parameter(torch.zeros(dp))
        self.q_level = nn.Linear(2, dp)
        self.q_layer = nn.Embedding(n_layers_total, dp)
        self.blocks = nn.ModuleList([Block(dp, heads, self.rope) for _ in range(layers)])
        self.norm = nn.LayerNorm(dp)
        self.out = nn.Linear(dp, d)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, ctx, ctx_coords, ctx_valid, q_coords, q_level, q_layer, q_valid):
        xq = self.mask_token + self.q_level(q_level) + self.q_layer(q_layer)
        x = torch.cat([self.inp(ctx), xq], dim=1)
        c = torch.cat([ctx_coords, q_coords], dim=1)
        m = torch.cat([ctx_valid, q_valid], dim=1)
        for blk in self.blocks:
            x = blk(x, c, m)
        return self.out(self.norm(x[:, ctx.shape[1]:]))


class JEPA(nn.Module):
    def __init__(self, cfg, geom: Geometry):
        super().__init__()
        m, j = cfg.model, cfg.jepa
        self.geom = geom
        self.cfg = cfg
        self.encoder = CellEncoder(m, geom)
        self.conditioner = Conditioner(int(m.d_model), m.get("conditioning", {}))
        self.target = copy.deepcopy(self.encoder)
        self.target_cond = copy.deepcopy(self.conditioner)
        for p in list(self.target.parameters()) + list(self.target_cond.parameters()):
            p.requires_grad_(False)
        self.predictor = Predictor(int(m.d_model), int(j.get("d_pred", 128)), int(j.get("pred_heads", 4)),
                                   int(j.get("pred_layers", 3)), geom.n_layers_total, m.get("rope", {}),
                                   geom.periodic)

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        for src, dst in ((self.encoder, self.target), (self.conditioner, self.target_cond)):
            for ps, pt in zip(src.parameters(), dst.parameters()):
                pt.mul_(momentum).add_(ps.detach(), alpha=1 - momentum)

    def loss(self, batch: dict) -> dict:
        j, m = self.cfg.jepa, self.cfg.model
        geom = self.geom
        factors = batch["factors"]
        spec_lr = geom.spec({g: (geom.group(g).shape[0] // factors[g][0], geom.group(g).shape[1] // factors[g][1])
                             for g in geom.group_names})
        lr = spec_lr.to_dense(batch["lr"])
        B = lr.shape[0]
        total = lr.sum(1)
        cells = cells_from_dense(lr, spec_lr, zero_frac=float(m.get("zero_token_frac", 0.0)),
                                 max_tokens=int(m.get("max_enc_tokens", 2048)))
        cond = self.conditioner(batch, B)
        masked = multiblock_mask(cells, int(j.get("n_blocks", 4)), tuple(j.get("block_scale", [0.15, 0.4])),
                                 geom.periodic)
        ctx_valid = cells.valid & ~masked
        ctx = self.encoder(cells, total, cond, mask=ctx_valid)
        with torch.no_grad():
            tcond = self.target_cond(batch, B)
            t_lr = F.layer_norm(self.target(cells, total, tcond), (ctx.shape[-1],))

        q_coords, q_level, q_layer, q_valid, targets, weights = \
            [cells.coords], [cells.level], [cells.layer], [masked], [t_lr], [torch.ones_like(total)]
        out = {}
        if j.get("cross_scale", True):
            spec_hr = geom.native_spec()
            hr = spec_hr.to_dense(batch["hr"])
            hcells = cells_from_dense(hr, spec_hr, max_tokens=int(j.get("max_hr_tokens", 2048)))
            with torch.no_grad():
                t_hr = F.layer_norm(self.target(hcells, hr.sum(1), tcond), (ctx.shape[-1],))
            nq = int(j.get("n_hr_queries", 256))
            pri = torch.rand(hcells.valid.shape, device=hr.device) + (~hcells.valid).float() * 2
            sel = torch.argsort(pri, dim=1)[:, :nq]
            g = lambda x: torch.gather(x, 1, sel if x.dim() == 2 else sel.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            q_coords.append(g(hcells.coords))
            q_level.append(g(hcells.level))
            q_layer.append(g(hcells.layer))
            q_valid.append(g(hcells.valid))
            targets.append(g(t_hr))
        n_in = cells.N
        pred = self.predictor(ctx, cells.coords, ctx_valid, torch.cat(q_coords, 1), torch.cat(q_level, 1),
                              torch.cat(q_layer, 1), torch.cat(q_valid, 1))
        tgt = torch.cat(targets, 1)
        qv = torch.cat(q_valid, 1).float()
        err = (pred - tgt).abs().mean(-1)
        l_in = (err[:, :n_in] * qv[:, :n_in]).sum() / qv[:, :n_in].sum().clamp(min=1)
        out["jepa_in"] = l_in
        loss = l_in
        if j.get("cross_scale", True):
            l_x = (err[:, n_in:] * qv[:, n_in:]).sum() / qv[:, n_in:].sum().clamp(min=1)
            out["jepa_cross"] = l_x
            loss = loss + float(j.get("cross_weight", 1.0)) * l_x
        out["loss"] = loss
        out["mask_ratio"] = (masked.float().sum() / cells.valid.float().sum().clamp(min=1)).detach()
        out["target_std"] = t_lr[cells.valid].std(0).mean().detach()  # collapse monitor
        return out


def ema_momentum(step: int, total: int, start: float, end: float) -> float:
    return end - (end - start) * (math.cos(math.pi * min(step, total) / max(total, 1)) + 1) / 2
