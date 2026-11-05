"""Evaluate CaloNSF and all trained baselines on the test split and write eval/metrics.json + report.md.

    python scripts/evaluate.py --config configs/experiments/qg_synthetic.yaml
"""
import os

from calonsf.eval.evaluate import run_evaluation
from calonsf.eval.report import write_report
from calonsf.train.common import load_calonsf, parse, setup


def extra(p):
    p.add_argument("--ckpt", default=None)


args = parse(__doc__, extra)
cfg, base, out, log, device, geom, reader, splits = setup(args, "eval")
ck = args.ckpt or os.path.join(base, "sr", "best.pt")
model = load_calonsf(cfg, geom, base, device, ck) if os.path.exists(ck) else None
if model is None:
    log.warning(f"no CaloNSF checkpoint at {ck}; evaluating baselines only")
run_evaluation(cfg, geom, reader, splits, model, base, device)
path = write_report(base, synthetic="synthetic" in str(cfg.data.files))
log.info(f"report: {path}")
