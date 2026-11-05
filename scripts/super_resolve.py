"""Super-resolve test events to ANY output size with one model, e.g. QG to 2048x2048 (or 300x300).
Generates on the next-scale ladder up to the next power-of-two refinement >= the requested size, then
uses exact area-overlap resampling (energy conserving) to hit the size exactly.

    python scripts/super_resolve.py --config configs/experiments/qg_synthetic.yaml --out-size 2048 --n 2
"""
import math
import os

import numpy as np

from calonsf.data.degrade import make_loader
from calonsf.data.ops import resample_energy
from calonsf.eval import plots
from calonsf.eval.plots import _view
from calonsf.train.common import load_calonsf, parse, setup
from calonsf.utils.misc import to_device


def extra(p):
    p.add_argument("--out-size", type=int, required=True, help="target size of the first SR axis of the first group")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--factor", type=int, default=None)


args = parse(__doc__, extra)
cfg, base, out, log, device, geom, reader, splits = setup(args, "super_resolve")
model = load_calonsf(cfg, geom, base, device)
g0 = geom.groups[0]
ax = 0 if g0.sr_axes[0] else 1
scale = args.out_size / g0.shape[ax]
n_extra = max(0, math.ceil(math.log2(scale) - 1e-9)) if scale > 1 else 0
factor = args.factor or int(cfg.eval.lr_factors[0])
dl = make_loader(cfg, reader, splits["test"][: args.n], geom, False, factors=[factor], batch_size=1, shuffle=False)
for i, batch in enumerate(dl):
    o = model.generate(to_device(batch, device), extra_steps=n_extra, nfe=int(cfg.eval.nfe), solver=cfg.eval.solver)
    res = {}
    for g, v in o["grids"].items():
        G = geom.group(g)
        tgt = tuple(int(round(G.shape[a] * scale)) if G.sr_axes[a] else G.shape[a] for a in range(2))
        res[g] = resample_energy(v, tgt).cpu().numpy()
    np.savez_compressed(os.path.join(out, f"event{i}_size{args.out_size}.npz"), **res,
                        **{f"lr_{g}": v.numpy() for g, v in batch["lr"].items()})
    img = _view(geom, res, 0)
    plots.save_image(os.path.join(out, f"event{i}_size{args.out_size}.png"), img,
                     f"CaloNSF output {img.shape[0]}x{img.shape[1]}")
    e_in = sum(float(v.sum()) for v in o["lr_init"].values())
    e_out = sum(float(v.sum()) for v in res.values())
    log.info(f"event {i}: shapes {[v.shape for v in res.values()]}, energy out/in = {e_out / max(e_in, 1e-12):.6f}")
log.info(f"wrote outputs to {out}")
