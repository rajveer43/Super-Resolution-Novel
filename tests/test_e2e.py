"""Synthetic end-to-end smoke test on CPU: JEPA -> SR training -> CaloChallenge evaluation report."""
import json
import os

import torch

from calonsf.data.degrade import make_loader
from calonsf.data.geometry import make_geometry
from calonsf.data.readers import make_reader, make_splits
from calonsf.eval.report import write_report
from calonsf.eval.run import run_evaluation
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
    cfg.eval["n_events"] = 24
    cfg.eval["nfe"] = 2
    cfg.eval["classifier"]["run"] = ["dnn_high", "cnn_low"]
    cfg.eval["classifier"]["seeds"] = 1
    cfg.eval["classifier"]["cnn_epochs"] = 1
    res = run_evaluation(cfg, geom, rd, sp, m.eval(), base, dev)
    R = res["factors"][str(cfg.eval.lr_factors[0])]
    assert set(R["methods"]) == {"calonsf", "nearest", "bicubic"}
    for r in R["methods"].values():
        assert 0 <= r["separation_power"]["summary"]["mean_all"] <= 1
        assert set(r["classifier"]) == {"dnn_high", "cnn_low"} and set(r["manifold"]) == {
            "precision", "recall", "density", "coverage"}
    assert R["parameters"]["needed_for_generation"] <= R["parameters"]["trainable_in_training"]
    assert json.load(open(os.path.join(base, "eval", "metrics.json")))["dataset"] == "calochallenge_dataset2"
    f = str(cfg.eval.lr_factors[0])
    for fig in ("separation_power", "histograms", "pcc", "summary"):  # figure errors are swallowed, so check
        assert os.path.exists(os.path.join(base, "eval", f"x{f}", f"{fig}.png")), fig
    assert os.path.exists(write_report(base, synthetic=True))
