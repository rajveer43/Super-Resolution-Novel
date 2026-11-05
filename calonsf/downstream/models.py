"""Downstream physics models. They are trained ONLY on ground-truth HR showers/jets and then applied
to SR outputs, so any drop in performance is caused by the super-resolution, not by retraining.

  qg            : ResNet-style CNN quark/gluon tagger (ROC-AUC, accuracy)
  calogan       : MLP with 3-class particle-ID head (e+/gamma/pi+) and incident-energy regression head
  calochallenge : MLP incident-energy regression
  probe         : V-JEPA attentive probe on the frozen, resolution-agnostic JEPA encoder
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.tokens import cells_from_dense


class _Res(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.b2 = nn.BatchNorm2d(cout)
        self.sk = nn.Sequential(nn.Conv2d(cin, cout, 1, stride), nn.BatchNorm2d(cout)) if (cin != cout or stride != 1) else nn.Identity()

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sk(x))


class QGNet(nn.Module):
    def __init__(self, es: float):
        super().__init__()
        self.es = es
        self.net = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(), _Res(32, 32, 1), _Res(32, 64, 2), _Res(64, 64, 1),
                                 _Res(64, 128, 2), _Res(128, 128, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, grids, batch=None):
        return self.net(torch.log1p(grids["jet"].clamp(min=0) / self.es)).squeeze(-1)


class FlatMLP(nn.Module):
    """MLP on log1p of all native-resolution cells."""

    def __init__(self, n_in: int, es: float, n_classes: int = 0, regress: bool = True, hidden: int = 256):
        super().__init__()
        self.es = es
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.ReLU())
        self.cls = nn.Linear(hidden, n_classes) if n_classes else None
        self.reg = nn.Linear(hidden, 1) if regress else None

    def forward(self, grids, batch=None):
        x = torch.cat([torch.log1p(v.clamp(min=0) / self.es).flatten(1) for _, v in sorted(grids.items())], 1)
        h = self.body(x)
        return {"logits": self.cls(h) if self.cls is not None else None,
                "log_e": self.reg(h).squeeze(-1) if self.reg is not None else None}


class AttentiveProbe(nn.Module):
    """Learnable query cross-attends to frozen encoder tokens (V-JEPA 2 attentive probe)."""

    def __init__(self, encoder, conditioner, geom, n_out: int, d: int, heads: int = 4, max_tokens: int = 4096):
        super().__init__()
        self.encoder, self.conditioner, self.geom = encoder, conditioner, geom
        for p in list(encoder.parameters()) + list(conditioner.parameters()):
            p.requires_grad_(False)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_out))
        self.max_tokens = max_tokens

    def forward(self, grids, batch):
        spec = self.geom.spec({g: tuple(grids[g].shape[-2:]) for g in self.geom.group_names})
        dense = spec.to_dense(grids)
        cells = cells_from_dense(dense, spec, max_tokens=self.max_tokens)
        with torch.no_grad():
            B = dense.shape[0]
            f = self.encoder(cells, dense.sum(1), self.conditioner(batch, B))
        o, _ = self.attn(self.q.expand(B, -1, -1), f, f, key_padding_mask=~cells.valid)
        return self.head(o[:, 0])


def make_task_model(geom, cfg):
    es = geom.energy_scale
    if geom.name == "qg":
        return QGNet(es)
    n_in = sum(g.n_layers * g.shape[0] * g.shape[1] for g in geom.groups)
    if geom.name == "calogan":
        return FlatMLP(n_in, es, n_classes=3, regress=True)
    return FlatMLP(n_in, es, n_classes=0, regress=True, hidden=512)


def task_loss(geom, out, batch):
    if geom.name == "qg":
        return F.binary_cross_entropy_with_logits(out, batch["label"].float())
    loss = F.mse_loss(out["log_e"], torch.log(batch["e_inc"].clamp(min=1e-6)))
    if out["logits"] is not None:
        loss = loss + F.cross_entropy(out["logits"], batch["label"])
    return loss


def probe_target(geom):
    """(n_out, kind): classification for QG / CaloGAN, log-energy regression for CaloChallenge."""
    if geom.name == "qg":
        return 1, "binary"
    if geom.name == "calogan":
        return 3, "multiclass"
    return 1, "regression"


def probe_loss(kind, out, batch):
    if kind == "binary":
        return F.binary_cross_entropy_with_logits(out.squeeze(-1), batch["label"].float())
    if kind == "multiclass":
        return F.cross_entropy(out, batch["label"])
    return F.mse_loss(out.squeeze(-1), torch.log(batch["e_inc"].clamp(min=1e-6)))
