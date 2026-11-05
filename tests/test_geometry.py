import math

import numpy as np
import torch

from calonsf.data.geometry import build_ladder, children_index, lr_factors_for, make_geometry, parent_index
from calonsf.data.ops import repeat_upsample, resample_energy, shift_energy, sum_pool
from calonsf.data.readers import make_reader, verify_flatten_order


def test_ladder_shapes(cfg):
    geom = make_geometry(cfg)
    for f in cfg.data.lr_factors:
        fac = lr_factors_for(geom, int(f))
        lad = build_ladder(geom, fac, extra_steps=2)
        assert lad.specs[lad.native_index].shapes == geom.native_shapes()
        for g in geom.group_names:
            H, W = geom.group(g).shape
            assert lad.specs[0].shapes[g] == (H // fac[g][0], W // fac[g][1])
            sr = geom.group(g).sr_axes
            assert lad.specs[-1].shapes[g] == (H * (4 if sr[0] else 1), W * (4 if sr[1] else 1))


def test_children_parent_roundtrip(cfg):
    geom = make_geometry(cfg)
    lad = build_ladder(geom, lr_factors_for(geom, int(cfg.data.lr_factors[-1])), extra_steps=1)
    for k, step in enumerate(lad.steps):
        sp, sk = lad.specs[k], lad.specs[k + 1]
        idx = torch.arange(sp.T).unsqueeze(0)
        cidx, sval, _ = children_index(idx, sp, sk, step, 16)
        ch = cidx[sval]
        # every child cell appears exactly once
        assert ch.numel() == sk.T and torch.unique(ch).numel() == sk.T
        par = parent_index(cidx, sk, sp, step)
        assert torch.equal(par[sval], idx.unsqueeze(-1).expand_as(cidx)[sval])


def test_rank_is_permutation(cfg):
    geom = make_geometry(cfg)
    spec = geom.native_spec()
    r = spec.rank(torch.device("cpu"))
    assert torch.equal(torch.sort(r).values, torch.arange(spec.T))


def test_energy_ops():
    x = torch.rand(2, 3, 12, 8)
    assert torch.allclose(sum_pool(x, 2, 4).sum(), x.sum())
    assert torch.allclose(repeat_upsample(x, 3, 2).sum(), x.sum())
    for shp in [(7, 5), (25, 13), (12, 8)]:
        assert torch.allclose(resample_energy(x, shp).sum(), x.sum(), rtol=1e-5)
    y = shift_energy(x, 0.5, -1)
    assert y.sum() <= x.sum() + 1e-5


def test_flatten_order_check(synth_dir):
    from calonsf.data.readers import CaloChallengeReader
    r = CaloChallengeReader([str(synth_dir / "dataset_2_1.hdf5")])
    assert verify_flatten_order(r, n=40)["ok"]
    r_bad = CaloChallengeReader([str(synth_dir / "dataset_2_1.hdf5")], flatten_order="zra")
    assert not verify_flatten_order(r_bad, n=40)["ok"]


def test_reader_shapes(cfg):
    geom = make_geometry(cfg)
    groups, meta = make_reader(cfg).get(0)
    for g in geom.group_names:
        assert groups[g].shape == (geom.group(g).n_layers, *geom.group(g).shape)
        assert (groups[g] >= 0).all()
