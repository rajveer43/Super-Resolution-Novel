"""Plot the JEPA and super-resolution training curves of an experiment from its CSV logs (also works on a run
that is still training): eval/training_jepa.png and eval/training_sr.png (+ PDF).

    python scripts/plot_training.py --config configs/experiments/calochallenge_large.yaml
"""
from calonsf.eval.training_plots import plot_training
from calonsf.train.common import parse, setup

args = parse(__doc__)
cfg, base, out, log, device, geom, reader, splits = setup(args, "plot_training")
for p in plot_training(base):
    log.info(f"wrote {p}")
