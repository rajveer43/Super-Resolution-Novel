"""Beyond-native checks: polar -> Cartesian resampling, large-grid moments, and the end-to-end consistency report."""
import math
import os

import numpy as np
import torch

from calonsf.data.geometry import make_geometry
from calonsf.data.readers import make_reader, make_splits
from calonsf.eval import features
from calonsf.eval.beyond_native import extra_steps_for, run_beyond_native, write_beyond_report
from calonsf.eval.cartesian import PolarToCartesian, image_centre_of_energy
from calonsf.models.calonsf import CaloNSF

R_EDGES = [0, 4.65, 9.3, 13.95, 18.6, 23.25, 27.9, 32.55, 37.2, 41.85]


def test_extra_steps_for_image_sizes():
    assert [extra_steps_for(n) for n in (256, 512, 1024, 2048, 4096)] == [4, 5, 6, 7, 8]
    assert extra_steps_for(18) == 0 and extra_steps_for(8) == 0


def test_polar_to_cartesian_conserves_energy_and_position():
    torch.manual_seed(0)
    A, R, N = 32, 18, 96
    p = PolarToCartesian(A, R, R_EDGES, N, samples=4)
    e = torch.rand(2, A, R, dtype=torch.float64) * (torch.rand(2, A, R) > 0.5)
    img = p(e)
    assert img.shape == (2, N, N)
    assert torch.allclose(img.sum((-1, -2)), e.sum((-1, -2)).to(img.dtype), rtol=1e-12)  # every share is deposited
    one = torch.zeros(1, A, R, dtype=torch.float64)
    one[0, 5, 10] = 3.0  # a single cell: its position must come back to within a pixel
    c = image_centre_of_energy(p(one), p)[0]
    r_c = 0.5 * (features.refine_r_edges(R_EDGES, R)[10] + features.refine_r_edges(R_EDGES, R)[11])
    a_c = (5 + 0.5) * 2 * math.pi / A
    assert abs(c[0] - r_c * math.cos(a_c)) < p.pixel and abs(c[1] - r_c * math.sin(a_c)) < p.pixel


def test_polar_moments_match_high_level_features():
    rng = np.random.default_rng(0)
    for A, R in ((16, 9), (32, 18)):  # native and refined grid
        x = (rng.random((2, 5, A, R)) * (rng.random((2, 5, A, R)) > 0.6)).astype(np.float32)
        x[:, 3] = 0  # an empty layer
        f = features.high_level_features(x, np.ones(2), R_EDGES, 0.0)
        m = features.polar_moments(torch.as_tensor(x), R_EDGES, chunk=2)
        for k in ("CoE_eta", "CoE_phi", "CoE_r", "width_eta", "width_phi", "width_r"):
            assert np.allclose(m[k], f[k], rtol=1e-5, atol=1e-6, equal_nan=True), k
        assert np.allclose(m["E"], f["E_layer"], rtol=1e-6)


def test_beyond_native_consistency(cfg):
    geom = make_geometry(cfg)
    rd = make_reader(cfg)
    sp = make_splits(cfg, rd)
    base = os.path.join(cfg.out_dir, cfg.exp_name)
    model = CaloNSF(cfg, geom).eval()
    res = run_beyond_native(cfg, geom, rd, sp, model, base, torch.device("cpu"), sizes=[48, 96], n_events=2)
    assert set(res["sizes"]) == {"48", "96"} and res["sizes"]["96"]["extra_steps"] == 3
    for N, d in res["sizes"].items():
        for k in ("conservation_polar", "conservation_image", "poolback_lr", "poolback_native"):
            assert d[k]["max"] < 1e-4, (N, k, d[k])
        assert d["hierarchical"]["max"] < 1e-4
        assert d["image_position_error_mm"]["max"] < 2 * d["pixel_mm"] + 1.0, N  # within a pixel or two
    assert os.path.exists(write_beyond_report(base, res, synthetic=True))
