"""Sanity-check a dataset: shapes, units, energy stats, LR sum-pooling closure and (CaloChallenge)
the voxel flattening order.

    python scripts/check_data.py --config configs/experiments/calochallenge.yaml
"""
import numpy as np

from calonsf.data.degrade import make_loader
from calonsf.data.readers import CaloChallengeReader, verify_flatten_order
from calonsf.train.common import parse, setup

args = parse(__doc__)
cfg, base, out, log, device, geom, reader, splits = setup(args, "check")
groups, meta = reader.get(0)
for g, v in groups.items():
    log.info(f"group {g}: shape {v.shape}, sum {v.sum():.4g}, nonzero frac {(v > 0).mean():.3f}")
log.info(f"meta: {meta}")
if isinstance(reader, CaloChallengeReader):
    verify_flatten_order(reader)
for f in cfg.data.lr_factors:
    b = next(iter(make_loader(cfg, reader, splits["train"][:64], geom, False, factors=[int(f)], batch_size=32)))
    hr = sum(float(v.sum()) for v in b["hr"].values())
    lr = sum(float(v.sum()) for v in b["lr_clean"].values())
    log.info(f"factor {f}: per-group factors {b['factors']}, LR shapes "
             f"{ {g: tuple(v.shape[-2:]) for g, v in b['lr'].items()} }, sum-pool closure {lr / hr:.8f}")
    e = b["e_inc"].numpy()
    log.info(f"incident energy range in batch: {np.min(e):.4g} .. {np.max(e):.4g}")
