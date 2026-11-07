"""LR generation: sum-pool -> electronic noise + threshold (Graph Denoising paper Sec. II) ->
optional DiffLense preprocessing (median -> Gaussian -> NLM -> threshold, DiffLense Eq. 2-3)."""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, median_filter

from .geometry import Geometry, lr_factors_for
from .ops import pool_grids


def _sigma_for(sig_cfg, group: str) -> float:
    if isinstance(sig_cfg, dict):
        return float(sig_cfg.get(group, sig_cfg.get("default", 0.0)))
    return float(sig_cfg)


def add_noise(grids: dict, noise_cfg, generator: torch.Generator | None = None) -> dict:
    """Gaussian noise on every LR cell, then keep only cells with E > k*sigma (positive energies only)."""
    if not noise_cfg or not noise_cfg.get("enabled", False):
        return grids
    out = {}
    k = float(noise_cfg.get("threshold_k", 2.0))
    for g, x in grids.items():
        s = _sigma_for(noise_cfg.sigma, g)
        if s <= 0:
            out[g] = x
            continue
        y = x + s * torch.randn(x.shape, generator=generator, dtype=x.dtype)
        out[g] = torch.where(y > k * s, y, torch.zeros_like(y))
    return out


def difflense_preprocess(grids: dict, pp_cfg, energy_scale: float) -> dict:
    """DiffLense denoising pipeline, applied per 2D layer image in log1p(E/E_scale) space.
    Each step can be switched off for the ablation (DiffLense Table 2)."""
    if not pp_cfg or not pp_cfg.get("enabled", False):
        return grids
    from skimage.restoration import denoise_nl_means

    out = {}
    for g, x in grids.items():
        arr = torch.log1p(x / energy_scale).numpy()
        res = np.empty_like(arr)
        for b in range(arr.shape[0]):
            for l in range(arr.shape[1]):
                im = arr[b, l]
                if pp_cfg.get("median", True) and min(im.shape) >= 3:
                    im = median_filter(im, size=3)
                gs = float(pp_cfg.get("gaussian_sigma", 0.5))
                if gs > 0:
                    im = gaussian_filter(im, gs)
                if pp_cfg.get("nlm", True) and min(im.shape) >= 3:
                    sd = float(im.std()) + 1e-8
                    im = denoise_nl_means(im, h=float(pp_cfg.get("nlm_h", 0.6)) * sd, patch_size=3,
                                          patch_distance=min(5, min(im.shape) // 2), fast_mode=True)
                thr = float(pp_cfg.get("threshold_rel", 0.02)) * (im.max() + 1e-12)
                im = np.where(im >= thr, im, 0.0)
                res[b, l] = im
        out[g] = (torch.expm1(torch.from_numpy(res).clamp(min=0)) * energy_scale).to(x.dtype)
    return out


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
        f = self.factors[int(torch.randint(len(self.factors), (1,), generator=self.gen))] \
            if self.train else self.factors[0]
        factors = lr_factors_for(self.geom, f, self.overrides)
        lr_clean = pool_grids(hr, factors)
        lr = add_noise(lr_clean, self.cfg.data.get("noise", None), self.gen)
        lr_in = difflense_preprocess(lr, self.cfg.data.get("preprocess", None), self.geom.energy_scale)
        return {
            "hr": hr, "lr_clean": lr_clean, "lr": lr_in, "factor": f, "factors": factors,
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
