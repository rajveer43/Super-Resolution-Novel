"""Stage 2: train the CaloNSF super-resolution model (optionally on a JEPA-pretrained encoder).

    python scripts/train_sr.py --config configs/experiments/qg_synthetic.yaml
"""
import os

import torch

from calonsf.data.degrade import make_loader
from calonsf.models.calonsf import CaloNSF
from calonsf.train.common import parse, setup
from calonsf.train.trainer import fit
from calonsf.utils.misc import count_params

args = parse(__doc__)
cfg, base, out, log, device, geom, reader, splits = setup(args, "sr")
t = cfg.train
model = CaloNSF(cfg, geom)
pre = t.get("pretrained_encoder", "auto")
if pre == "auto":
    pre = os.path.join(base, "jepa", "encoder.pt")
    pre = pre if os.path.exists(pre) else None
if pre:
    miss, unexp = model.load_pretrained_encoder(torch.load(pre, map_location="cpu"))
    log.info(f"loaded JEPA encoder from {pre} (missing={len(miss)}, unexpected={len(unexp)})")
    if float(cfg.loss.get("jepa", 0)) > 0:
        model.set_perceptual_encoder()
        log.info("physics perceptual loss: ON (frozen JEPA encoder)")
else:
    log.info("no pretrained encoder: training from scratch, perceptual loss OFF")
model.to(device)
log.info(f"CaloNSF params (trainable): {count_params(model) / 1e6:.2f}M")
enc_params = list(model.encoder.parameters()) + list(model.conditioner.parameters())
enc_ids = {id(p) for p in enc_params}
rest = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
groups = [{"params": rest}, {"params": enc_params, "lr": float(t.lr) * float(t.encoder_lr_mult) if pre else float(t.lr)}]
freeze = int(t.get("freeze_encoder_epochs", 0)) if pre else 0


def on_epoch(epoch):
    flag = epoch >= freeze
    for p in enc_params:
        p.requires_grad_(flag)
    if epoch == 0 and freeze:
        log.info(f"encoder frozen for the first {freeze} epoch(s)")


tr = make_loader(cfg, reader, splits["train"], geom, True)
va = make_loader(cfg, reader, splits["val"], geom, False)
fit(model, model.loss, tr, va, epochs=int(t.epochs), lr=float(t.lr), weight_decay=float(t.weight_decay), device=device,
    out_dir=out, schedule=t.get("schedule", "cosine"), warmup_frac=float(t.warmup_frac), grad_clip=float(t.grad_clip),
    max_steps_per_epoch=t.get("max_steps_per_epoch"), param_groups=groups, on_epoch=on_epoch, tag="sr",
    log_every=int(t.log_every))
log.info(f"done: {out}/best.pt")
