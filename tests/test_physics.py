import numpy as np
import torch

from calonsf.data.geometry import make_geometry
from calonsf.data.ops import repeat_upsample
from calonsf.data.readers import make_reader
from calonsf.eval.metrics import energy_metrics, image_metrics
from calonsf.physics import stats
from calonsf.physics.observables import observables


def _grids(cfg, n=8):
    rd = make_reader(cfg)
    items = [rd.get(i) for i in range(n)]
    g = {k: torch.from_numpy(np.stack([it[0][k] for it in items])) for k in items[0][0]}
    meta = {"e_inc": np.array([it[1]["e_inc"] for it in items]), "label": np.array([it[1]["label"] for it in items])}
    return g, meta


def test_observables_identity(cfg):
    geom = make_geometry(cfg)
    g, meta = _grids(cfg)
    ob = observables(g, geom, meta)
    cmp = stats.compare(ob, ob)
    assert all(v["w1"] == 0 for v in cmp.values())
    im = image_metrics(g, g, geom.energy_scale)
    for v in im.values():
        assert v["MAE"] == 0 and v["SSIM"] > 0.999
    assert abs(energy_metrics(g, g)["response_mean"] - 1) < 1e-9


def test_observables_at_higher_resolution(cfg):
    """Energy-weighted observables are resolution-stable for a uniform split; granularity ones are not."""
    geom = make_geometry(cfg)
    g, meta = _grids(cfg)
    up = {k: repeat_upsample(v, 2 if geom.group(k).sr_axes[0] else 1, 2 if geom.group(k).sr_axes[1] else 1)
          for k, v in g.items()}
    a, b = observables(g, geom, meta), observables(up, geom, meta)
    key = {"qg": "E_ecal", "calogan": "E_tot", "calochallenge": "E_tot"}[geom.name]
    assert np.allclose(a[key], b[key], rtol=1e-4)


def test_fpd_kpd():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 5))
    assert stats.fpd(a, a) < 1e-6
    assert stats.fpd(a, a + 1.0) > 1.0
    assert abs(stats.kpd(a, a, batch=200)[0]) < 0.05
