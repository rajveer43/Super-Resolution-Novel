"""Super-resolution beyond native resolution: consistency checks at Cartesian image sizes 256, 512, 1024, 2048, ...
(x = eta, y = phi) for CaloChallenge Dataset 2. Writes beyond_native/metrics.json and report.md.

    python scripts/beyond_native.py --config configs/experiments/calochallenge.yaml --sizes 256 512 1024 2048 --n 8
"""
import os

from calonsf.eval.beyond_native import run_beyond_native, write_beyond_report
from calonsf.train.common import load_calonsf, parse, setup


def extra(p):
    p.add_argument("--sizes", type=int, nargs="+", default=None, help="image sizes N (N x N); default eval.beyond_native.sizes")
    p.add_argument("--n", type=int, default=None, help="number of test showers")
    p.add_argument("--factor", type=int, default=None, help="LR factor; default eval.lr_factors[0]")
    p.add_argument("--images", type=int, default=None, help="save sample images for the first N showers (default eval.beyond_native.images)")
    p.add_argument("--ckpt", default=None)


args = parse(__doc__, extra)
cfg, base, out, log, device, geom, reader, splits = setup(args, "beyond_native")
ck = args.ckpt or os.path.join(base, "sr", "best.pt")
model = load_calonsf(cfg, geom, base, device, ck)
res = run_beyond_native(cfg, geom, reader, splits, model, base, device, args.sizes, args.n, args.factor, args.images)
log.info(f"report: {write_beyond_report(base, res, synthetic='synthetic' in str(cfg.data.files))}")
