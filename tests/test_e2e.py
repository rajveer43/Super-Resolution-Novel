"""Synthetic end-to-end smoke test on CPU: JEPA -> SR training -> downstream -> evaluation report."""
import os

import torch

from calonsf.data.degrade import make_loader
from calonsf.data.geometry import make_geometry
from calonsf.data.readers import make_reader, make_splits
from calonsf.downstream.models import make_task_model, task_loss
from calonsf.eval.evaluate import run_evaluation
from calonsf.eval.report import write_report
from calonsf.models.calonsf import CaloNSF
from calonsf.models.jepa import JEPA
from calonsf.train.trainer import fit


def test_end_to_end(cfg):
    dev = torch.device("cpu")
    geom = make_geometry(cfg)
    rd = make_reader(cfg)
    sp = make_splits(cfg, rd)
    base = os.path.join(cfg.out_dir, cfg.exp_name)
    tr = make_loader(cfg, rd, sp["train"], geom, True)
    va = make_loader(cfg, rd, sp["val"], geom, False)
    kw = dict(epochs=1, lr=1e-3, weight_decay=0.0, device=dev, max_steps_per_epoch=2, log_every=1000)
    j = JEPA(cfg, geom)
    fit(j, j.loss, tr, va, out_dir=os.path.join(base, "jepa"), tag="jepa", **kw)
    m = CaloNSF(cfg, geom)
    m.load_pretrained_encoder(j.state_dict())
    m.set_perceptual_encoder()
    fit(m, m.loss, tr, va, out_dir=os.path.join(base, "sr"), tag="sr", **kw)
    t = make_task_model(geom, cfg)
    fit(t, lambda b: {"loss": task_loss(geom, t(b["hr"], b), b)}, tr, va, out_dir=os.path.join(base, "downstream", "task"),
        tag="task", **kw)
    cfg.eval["n_events"] = 8
    cfg.eval["ensemble"] = 2
    cfg.eval["nfe"] = 2
    cfg.eval["baselines"] = ["nearest", "bicubic"]
    res = run_evaluation(cfg, geom, rd, sp, m.eval(), base, dev)
    f = str(cfg.eval.lr_factors[0])
    r = res["factors"][f]["methods"]["calonsf"]
    assert r["energy"]["conservation_err_max"] < 1e-4
    assert "downstream_task" in r
    assert os.path.exists(write_report(base, synthetic=True))
