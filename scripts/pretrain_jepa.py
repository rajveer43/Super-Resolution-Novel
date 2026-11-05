"""Stage 1: self-supervised JEPA pretraining of the cell encoder (V-JEPA 2 recipe, cell-set version).

    python scripts/pretrain_jepa.py --config configs/experiments/qg_synthetic.yaml
"""
import os

import torch

from calonsf.data.degrade import make_loader
from calonsf.models.jepa import JEPA, ema_momentum
from calonsf.train.common import parse, setup
from calonsf.train.trainer import fit
from calonsf.utils.misc import count_params

args = parse(__doc__)
cfg, base, out, log, device, geom, reader, splits = setup(args, "jepa")
j = cfg.jepa
model = JEPA(cfg, geom).to(device)
log.info(f"JEPA params (trainable): {count_params(model) / 1e6:.2f}M")
bs = int(j.get("batch_size", cfg.data.batch_size))
tr = make_loader(cfg, reader, splits["train"], geom, True, batch_size=bs)
va = make_loader(cfg, reader, splits["val"], geom, False, batch_size=bs)
ema = list(j.get("ema", [0.996, 1.0]))
prog = j.get("progressive_factors", None)
epochs = int(j.epochs)


def on_step(step, total):
    model.ema_update(ema_momentum(step, total, ema[0], ema[1]))


def on_epoch(epoch):
    # progressive resolution (V-JEPA 2): coarse LR factors first, finer ones in the last phase
    if prog:
        stage = min(len(prog) - 1, int(epoch / max(1, epochs) * len(prog)))
        tr.collate_fn.factors = list(prog[stage])
        log.info(f"[jepa] epoch {epoch}: LR factors {tr.collate_fn.factors}")


fit(model, model.loss, tr, va, epochs=epochs, lr=float(j.lr), weight_decay=float(cfg.train.weight_decay),
    device=device, out_dir=out, schedule="wcd", warmup_frac=float(cfg.train.warmup_frac),
    grad_clip=float(cfg.train.grad_clip), max_steps_per_epoch=cfg.train.get("max_steps_per_epoch"),
    on_step=on_step, on_epoch=on_epoch, tag="jepa", log_every=int(cfg.train.log_every))
torch.save(model.state_dict(), os.path.join(out, "encoder.pt"))
log.info(f"saved {os.path.join(out, 'encoder.pt')}")
