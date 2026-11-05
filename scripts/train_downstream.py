"""Train downstream physics models on GROUND-TRUTH HR data only (never on SR outputs):
the task model (q/g tagger, particle ID + energy regression, energy regression) and, if a JEPA encoder
exists, the attentive probe on the frozen encoder.

    python scripts/train_downstream.py --config configs/experiments/qg_synthetic.yaml
"""
import os

import torch

from calonsf.data.degrade import make_loader
from calonsf.downstream.models import AttentiveProbe, make_task_model, probe_loss, probe_target, task_loss
from calonsf.models.encoder import CellEncoder, Conditioner
from calonsf.train.common import parse, setup
from calonsf.train.trainer import fit

args = parse(__doc__)
cfg, base, out, log, device, geom, reader, splits = setup(args, "downstream")
ds = cfg.downstream
bs = int(ds.batch_size)
tr = make_loader(cfg, reader, splits["train"], geom, True, batch_size=bs)
va = make_loader(cfg, reader, splits["val"], geom, False, batch_size=bs)
common = dict(weight_decay=1e-4, device=device, warmup_frac=0.02, grad_clip=1.0,
              max_steps_per_epoch=cfg.train.get("max_steps_per_epoch"), log_every=int(cfg.train.log_every))

model = make_task_model(geom, cfg).to(device)
fit(model, lambda b: {"loss": task_loss(geom, model(b["hr"], b), b)}, tr, va, epochs=int(ds.epochs),
    lr=float(ds.lr), out_dir=os.path.join(out, "task"), tag="task", **common)

enc_path = os.path.join(base, "jepa", "encoder.pt")
if os.path.exists(enc_path):
    st = torch.load(enc_path, map_location="cpu")
    enc = CellEncoder(cfg.model, geom)
    con = Conditioner(int(cfg.model.d_model), cfg.model.get("conditioning", {}))
    enc.load_state_dict({k[8:]: v for k, v in st.items() if k.startswith("encoder.")})
    con.load_state_dict({k[12:]: v for k, v in st.items() if k.startswith("conditioner.")})
    n_out, kind = probe_target(geom)
    probe = AttentiveProbe(enc, con, geom, n_out, int(cfg.model.d_model), int(cfg.model.heads)).to(device)
    fit(probe, lambda b: {"loss": probe_loss(kind, probe(b["hr"], b), b)}, tr, va, epochs=int(ds.probe_epochs),
        lr=float(ds.lr), out_dir=os.path.join(out, "probe"), tag="probe", **common)
else:
    log.info("no JEPA encoder found: skipping attentive probe")
