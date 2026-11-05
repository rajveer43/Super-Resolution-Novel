"""Shared setup for the entry-point scripts."""
from __future__ import annotations

import argparse
import os

import torch

from ..data.geometry import make_geometry
from ..data.readers import make_reader, make_splits
from ..utils.config import load_config, save_config
from ..utils.misc import get_device, get_logger, run_dir, seed_everything


def parse(desc: str, extra=None):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", required=True)
    p.add_argument("-o", "--override", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=2")
    if extra:
        extra(p)
    return p.parse_args()


def setup(args, sub: str):
    cfg = load_config(args.config, args.override)
    seed_everything(int(cfg.seed))
    base = run_dir(cfg)
    out = os.path.join(base, sub)
    os.makedirs(out, exist_ok=True)
    log = get_logger(log_file=os.path.join(base, "log.txt"))
    save_config(cfg, os.path.join(out, "config.yaml"))
    device = get_device(cfg.device)
    geom = make_geometry(cfg)
    reader = make_reader(cfg)
    splits = make_splits(cfg, reader)
    log.info(f"exp={cfg.exp_name} sub={sub} device={device}")
    return cfg, base, out, log, device, geom, reader, splits


def load_calonsf(cfg, geom, base: str, device, ckpt: str | None = None):
    from ..models.calonsf import CaloNSF
    model = CaloNSF(cfg, geom).to(device)
    path = ckpt or os.path.join(base, "sr", "best.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck["model"]
    if any(k.startswith("percep.") for k in state):
        model.set_perceptual_encoder()
        model.to(device)
    model.load_state_dict(state)
    model.eval()
    return model
