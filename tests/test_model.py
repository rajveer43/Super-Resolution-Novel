import copy
import math

import torch

from calonsf.data.degrade import make_loader
from calonsf.data.geometry import make_geometry
from calonsf.data.ops import sum_pool
from calonsf.data.readers import make_reader, make_splits
from calonsf.models.baselines import REGISTRY, make_baseline
from calonsf.models.blocks import ContinuousRoPE
from calonsf.models.calonsf import CaloNSF
from calonsf.models.jepa import JEPA


def _batch(cfg, factor=None):
    geom = make_geometry(cfg)
    rd = make_reader(cfg)
    sp = make_splits(cfg, rd)
    f = [factor] if factor else None
    return geom, next(iter(make_loader(cfg, rd, sp["train"], geom, True, factors=f)))


def test_rope_periodic_and_relative():
    rope = ContinuousRoPE(24, periodic=(True, False))
    q = torch.randn(1, 1, 5, 24)
    k = torch.randn(1, 1, 5, 24)
    c = torch.rand(1, 5, 3)

    def scores(cq, ck):
        a, b = rope.cos_sin(cq)
        x, y = rope.cos_sin(ck)
        return ContinuousRoPE.apply(q, a, b) @ ContinuousRoPE.apply(k, x, y).transpose(-1, -2)

    s0 = scores(c, c)
    c2 = c.clone()
    c2[..., 1] += 2 * math.pi  # periodic axis: full turn
    assert torch.allclose(s0, scores(c2, c2), atol=1e-3)
    c3 = c.clone()
    c3[..., 2] += 0.37  # translation on a non-periodic axis: relative positions unchanged
    assert torch.allclose(s0, scores(c3, c3), atol=1e-3)


def test_loss_backward(cfg):
    geom, batch = _batch(cfg)
    j = JEPA(cfg, geom)
    o = j.loss(batch)
    o["loss"].backward()
    assert torch.isfinite(o["loss"])
    m = CaloNSF(cfg, geom)
    m.load_pretrained_encoder(j.state_dict())
    m.set_perceptual_encoder()
    o = m.loss(batch)
    o["loss"].backward()
    assert torch.isfinite(o["loss"])


def test_generate_conserves_energy_any_resolution(cfg):
    geom, batch = _batch(cfg, int(cfg.data.lr_factors[-1]))
    m = CaloNSF(cfg, geom).eval()
    o = m.generate(batch, extra_steps=2, ensemble=2, nfe=2, return_scales=True)
    lad = o["ladder"]
    e_in = sum(v.double().sum((1, 2, 3)) for v in o["lr_init"].values())
    e_out = sum(v.double().sum((1, 2, 3)) for v in o["grids"].values())
    assert torch.allclose(e_in, e_out, rtol=1e-4)
    # pooling any scale back to the previous one reproduces it exactly (hierarchical consistency)
    for k, step in enumerate(lad.steps):
        fine, coarse = o["scales"][k + 1], o["scales"][k]
        for g in fine:
            assert torch.allclose(sum_pool(fine[g], *step[g]), coarse[g], rtol=1e-4, atol=1e-5)
    for g, v in o["grids"].items():
        assert tuple(v.shape[-2:]) == lad.specs[-1].shapes[g]
        assert (v >= 0).all()


def test_ablation_variants_run(cfg):
    geom, batch = _batch(cfg)
    for ov in ({"generator": "regression"}, {"ladder": "one_shot", "max_children": 64}, {"conservation": False},
               {"pos_encoding": "absolute"}):
        c = copy.deepcopy(cfg)
        c.model.update(ov)
        m = CaloNSF(c, geom)
        m.loss(batch)["loss"].backward()
        m.eval().generate(batch, nfe=1)


def test_baselines_shapes(cfg):
    f = int(cfg.eval.lr_factors[0])
    geom, batch = _batch(cfg, f)
    cfg.baseline["ddpm_sample_steps"] = 2
    cfg.baseline["graphfm_nfe"] = 1
    for name in REGISTRY:
        b = make_baseline(name, cfg, geom, f)
        if b.trainable:
            if hasattr(b, "set_epoch"):
                b.set_epoch(0, 2)
            b.loss(batch)["loss"].backward()
        p = b.predict(batch)
        for g in p:
            assert p[g].shape == batch["hr"][g].shape
