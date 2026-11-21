"""Training-loop behaviour: LR schedule, validation over all LR factors, metric-based checkpoint selection."""
import os

import numpy as np
import torch

from calonsf.data.degrade import make_loader
from calonsf.data.geometry import make_geometry
from calonsf.data.readers import make_reader, make_splits
from calonsf.eval.select import make_separation_metric
from calonsf.models.calonsf import CaloNSF
from calonsf.train.trainer import fit, lr_lambda


def test_wcd_schedule_decay_fraction():
    total, warm = 1000, 50
    short, long_ = lr_lambda("wcd", total, warm, 0.2), lr_lambda("wcd", total, warm, 0.4)
    assert short(700) == 1.0 and long_(700) < 1.0  # 40 % decay starts at step 600
    assert short(warm - 1) == 1.0 and long_(0) < 0.05  # warmup
    assert abs(long_(800) - 0.5) < 1e-9 and long_(total) == 0.0
    assert lr_lambda("wcd", total, warm)(790) == short(790)  # default decay_frac stays 0.2


def test_validation_cycles_through_lr_factors(cfg):
    geom, rd = make_geometry(cfg), make_reader(cfg)
    sp = make_splits(cfg, rd)
    cfg.data["lr_factors"] = [2, 4]
    seen = [int(b["factor"]) for b in make_loader(cfg, rd, sp["val"], geom, False, batch_size=1)]
    assert set(seen) == {2, 4}
    again = [int(b["factor"]) for b in make_loader(cfg, rd, sp["val"], geom, False, batch_size=1)]
    assert seen == again  # deterministic
    only = {int(b["factor"]) for b in make_loader(cfg, rd, sp["val"], geom, False, factors=[4], batch_size=1)}
    assert only == {4}  # an explicit single factor is unchanged


def test_fit_selects_best_checkpoint_by_metric(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    data = [{"x": torch.randn(4, 2)}]

    def loss_fn(b):
        return {"loss": model(b["x"]).pow(2).mean()}

    scores = iter([0.9, 0.2, 0.5, 0.7])  # the metric is best at epoch 1, whatever the loss does
    fit(model, loss_fn, data, data, epochs=4, lr=1e-2, weight_decay=0.0, device=torch.device("cpu"), out_dir=str(tmp_path),
        val_metric=lambda: next(scores), tag="t", log_every=1000)
    assert torch.load(tmp_path / "best.pt", weights_only=False)["epoch"] == 1
    rows = open(tmp_path / "t_log.csv").read().strip().splitlines()
    assert "val_metric" in rows[0] and len(rows) == 5


def test_separation_metric_is_deterministic_and_keeps_rng(cfg):
    geom, rd = make_geometry(cfg), make_reader(cfg)
    sp = make_splits(cfg, rd)
    cfg.eval["nfe"] = 2
    model = CaloNSF(cfg, geom)
    metric = make_separation_metric(cfg, geom, rd, sp["val"], torch.device("cpu"), n_events=8, n_bins=10)
    model.train()
    torch.manual_seed(5)
    expected_next = torch.rand(3)
    torch.manual_seed(5)
    a = metric(model)
    after = torch.rand(3)
    b = metric(model)
    assert 0.0 <= a <= 1.0 and a == b  # same events and noise every call
    assert torch.equal(after, expected_next)  # training RNG stream untouched
    assert model.training  # train / eval mode restored
