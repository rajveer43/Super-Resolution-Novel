"""Build CaloNSF for SR training: JEPA encoder, perceptual loss, optimiser groups, encoder freeze.
Shared by scripts/train_sr.py and scripts/tune.py so both train exactly the same way."""
from __future__ import annotations

import os

import torch

from ..models.calonsf import CaloNSF
from ..utils.misc import count_params


def build_sr(cfg, geom, base: str, device, log):
    """Returns (model, param_groups, on_epoch) ready for trainer.fit."""
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
    groups = [{"params": rest},
              {"params": enc_params, "lr": float(t.lr) * float(t.encoder_lr_mult) if pre else float(t.lr)}]
    freeze = int(t.get("freeze_encoder_epochs", 0)) if pre else 0

    def on_epoch(epoch):
        flag = epoch >= freeze
        for p in enc_params:
            p.requires_grad_(flag)
        if epoch == 0 and freeze:
            log.info(f"encoder frozen for the first {freeze} epoch(s)")

    return model, groups, on_epoch
