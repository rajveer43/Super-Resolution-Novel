"""Stage 2: train the CaloNSF super-resolution model (optionally on a JEPA-pretrained encoder).

    python scripts/train_sr.py --config configs/experiments/calochallenge_synthetic.yaml
"""
from calonsf.data.degrade import make_loader
from calonsf.eval.select import make_separation_metric
from calonsf.train.common import parse, setup
from calonsf.train.sr import build_sr
from calonsf.train.trainer import fit

args = parse(__doc__)
cfg, base, out, log, device, geom, reader, splits = setup(args, "sr")
t = cfg.train
model, groups, on_epoch = build_sr(cfg, geom, base, device, log)
tr = make_loader(cfg, reader, splits["train"], geom, True)
va = make_loader(cfg, reader, splits["val"], geom, False)
val_metric = None
n_sel = int(t.get("select_events", 0))
if n_sel > 0:  # choose best.pt by the CaloChallenge separation power on validation showers, not by the loss
    sel = make_separation_metric(cfg, geom, reader, splits["val"], device, n_sel, n_bins=int(t.get("select_bins", 20)))
    val_metric = lambda: sel(model)
    log.info(f"checkpoint selection: mean separation power on {n_sel} validation showers (lower is better)")
fit(model, model.loss, tr, va, epochs=int(t.epochs), lr=float(t.lr), weight_decay=float(t.weight_decay), device=device,
    out_dir=out, schedule=t.get("schedule", "cosine"), warmup_frac=float(t.warmup_frac), grad_clip=float(t.grad_clip),
    max_steps_per_epoch=t.get("max_steps_per_epoch"), param_groups=groups, on_epoch=on_epoch, tag="sr",
    log_every=int(t.log_every), val_metric=val_metric)
log.info(f"done: {out}/best.pt")
