"""Evaluate CaloNSF on CaloChallenge Dataset 2 with the challenge metrics (separation power of the high-level
features, layer-energy PCC, binary classifier AUCs, FPD / KPD, precision / recall / density / coverage, timing,
parameter counts), next to the nearest and bicubic upsamplers. Writes eval/metrics.json, report.md and figures.

    python scripts/evaluate.py --config configs/experiments/calochallenge.yaml [-o eval.classifier.seeds=3 ...]
"""
import os

from calonsf.eval.report import write_report
from calonsf.eval.run import run_evaluation
from calonsf.train.common import load_calonsf, parse, setup


def extra(p):
    p.add_argument("--ckpt", default=None)


args = parse(__doc__, extra)
cfg, base, out, log, device, geom, reader, splits = setup(args, "eval")
ck = args.ckpt or os.path.join(base, "sr", "best.pt")
model = load_calonsf(cfg, geom, base, device, ck) if os.path.exists(ck) else None
if model is None:
    log.warning(f"no CaloNSF checkpoint at {ck}; evaluating the nearest / bicubic upsamplers only")
run_evaluation(cfg, geom, reader, splits, model, base, device)
log.info(f"report: {write_report(base, synthetic='synthetic' in str(cfg.data.files))}")
