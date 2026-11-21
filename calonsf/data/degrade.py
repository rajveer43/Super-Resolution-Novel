"""LR generation: the LR shower is the sum-pooled HR shower (energy is conserved exactly)."""
from __future__ import annotations

import numpy as np
import torch

from .geometry import Geometry, lr_factors_for
from .ops import pool_grids


class Collate:
    """Stacks HR events and builds the LR views. One nominal LR factor is drawn per batch so a single
    model sees several scale ladders (resolution-invariant training)."""

    def __init__(self, geom: Geometry, cfg, factors: list[int] | None = None, train: bool = True):
        self.geom = geom
        self.cfg = cfg
        self.factors = list(factors if factors is not None else cfg.data.lr_factors)
        self.train = train
        self.overrides = cfg.data.get("factor_overrides", None)
        self.gen = torch.Generator().manual_seed(int(cfg.seed) + (0 if train else 999))

    def __call__(self, items):
        hr = {g: torch.from_numpy(np.stack([it[0][g] for it in items])) for g in self.geom.group_names}
        metas = [it[1] for it in items]
        if self.train:
            f = self.factors[int(torch.randint(len(self.factors), (1,), generator=self.gen))]
        else:  # evaluation: one factor per batch, cycling through the list (a single factor stays fixed)
            f = self.factors[int(items[0][1].get("_index", 0)) % len(self.factors)]
        factors = lr_factors_for(self.geom, f, self.overrides)
        lr = pool_grids(hr, factors)
        return {
            "hr": hr, "lr": lr, "factor": f, "factors": factors,
            "label": torch.tensor([m.get("label", -1) for m in metas], dtype=torch.long),
            "particle": torch.tensor([m.get("particle", -1) for m in metas], dtype=torch.long),
            "e_inc": torch.tensor([m.get("e_inc", -1.0) for m in metas], dtype=torch.float32),
            "index": torch.tensor([m.get("_index", -1) for m in metas], dtype=torch.long),
        }


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, reader, indices):
        self.reader = reader
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        groups, meta = self.reader.get(j)
        meta = dict(meta)
        meta["_index"] = j
        return groups, meta


def make_loader(cfg, reader, indices, geom, train: bool, factors=None, batch_size=None, shuffle=None):
    ds = EventDataset(reader, indices)
    return torch.utils.data.DataLoader(
        ds, batch_size=int(batch_size or cfg.data.batch_size), shuffle=train if shuffle is None else shuffle,
        num_workers=int(cfg.data.get("num_workers", 0)), collate_fn=Collate(geom, cfg, factors, train),
        drop_last=train, persistent_workers=False)
