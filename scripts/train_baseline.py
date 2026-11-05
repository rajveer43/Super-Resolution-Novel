"""Train one (or all) baselines for a fixed LR factor (eval.lr_factors[0] unless --factor is given).

    python scripts/train_baseline.py --config configs/experiments/qg_synthetic.yaml --name edsr
    python scripts/train_baseline.py --config ... --name all
"""
import os

import torch

from calonsf.data.degrade import make_loader
from calonsf.models.baselines import REGISTRY, make_baseline
from calonsf.train.common import parse, setup
from calonsf.train.trainer import fit
from calonsf.utils.misc import count_params


def extra(p):
    p.add_argument("--name", required=True, help=f"one of {list(REGISTRY)} or 'all'")
    p.add_argument("--factor", type=int, default=None)


args = parse(__doc__, extra)
cfg, base, out, log, device, geom, reader, splits = setup(args, "baselines")
factor = args.factor or int(cfg.eval.lr_factors[0])
names = cfg.eval.baselines if args.name == "all" else [args.name]
b = cfg.baseline
for name in names:
    model = make_baseline(name, cfg, geom, factor).to(device)
    d = os.path.join(out, f"{name}_x{factor}")
    if not model.trainable:
        os.makedirs(d, exist_ok=True)
        torch.save({"model": model.state_dict()}, os.path.join(d, "best.pt"))
        log.info(f"[{name}] no training needed")
        continue
    log.info(f"[{name}] params {count_params(model) / 1e6:.2f}M, factor {factor}")
    tr = make_loader(cfg, reader, splits["train"], geom, True, factors=[factor])
    va = make_loader(cfg, reader, splits["val"], geom, False, factors=[factor])
    epochs = int(b.epochs)
    on_epoch = (lambda e, m=model: m.set_epoch(e, epochs)) if hasattr(model, "set_epoch") else None
    fit(model, model.loss, tr, va, epochs=epochs, lr=float(b.lr), weight_decay=0.0, device=device, out_dir=d,
        warmup_frac=0.02, grad_clip=1.0, max_steps_per_epoch=cfg.train.get("max_steps_per_epoch"),
        on_epoch=on_epoch, tag=name, log_every=int(cfg.train.log_every))
