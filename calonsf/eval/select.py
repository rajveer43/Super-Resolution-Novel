"""Checkpoint selection by a CaloChallenge metric instead of the training loss.

The SR training loss (flow matching + occupancy + ...) has a floor and stops moving long before the showers stop
improving, and it is not what the challenge measures. `make_separation_metric` returns a function that, for the
CURRENT model, generates one sample for each of `n_events` validation showers (always the same events, the same
random numbers, so epochs are comparable) and returns the mean separation power of the high-level features
(layer energy, centres of energy and widths in eta / phi / r, sparsity, E_dep / E_inc; see `separation.py`) with
respect to the Geant4 showers. Lower is better. Training uses it to choose best.pt."""
from __future__ import annotations

import numpy as np
import torch

from ..data.degrade import make_loader
from ..utils.misc import to_device
from . import features, separation


def make_separation_metric(cfg, geom, reader, idx, device, n_events: int, factor: int | None = None, n_bins: int = 20,
                           nfe: int | None = None):
    factor = int(factor or cfg.data.lr_factors[-1])
    dl = make_loader(cfg, reader, idx[:n_events], geom, False, factors=[factor], batch_size=int(cfg.eval.batch_size), shuffle=False)
    batches, hr, e_inc = [], [], []
    for b in dl:  # read the validation events once; HDF5 is not touched again during training
        batches.append({k: v for k, v in b.items()})
        hr.append(b["hr"]["shower"].numpy())
        e_inc.append(b["e_inc"].numpy())
    hr, e_inc = np.concatenate(hr), np.concatenate(e_inc)
    r_edges, thr = geom.extras["r_edges"], float(cfg.eval.threshold_mev)
    f_ref = features.high_level_features(hr, e_inc, r_edges, thr)
    edges = separation.make_edges(f_ref, n_bins)
    h_ref = separation.feature_hists(f_ref, edges)
    nfe = int(nfe or cfg.eval.nfe)

    @torch.no_grad()
    def metric(model) -> float:
        was_training = model.training
        model.eval()
        gen_devices = [device] if device.type == "cuda" else []
        outs = []
        with torch.random.fork_rng(devices=gen_devices):  # same noise every call, training RNG untouched
            torch.manual_seed(1234)
            for b in batches:
                o = model.generate(to_device(b, device), ensemble=1, nfe=nfe, solver=cfg.eval.solver)
                outs.append(o["grids"]["shower"].float().cpu().numpy())
        model.train(was_training)
        pred = features.readout_cutoff(np.concatenate(outs), thr)
        h_p = separation.feature_hists(features.high_level_features(pred, e_inc, r_edges, thr), edges)
        return float(separation.summarize(separation.sep_from_hists(h_ref, h_p))["mean_all"])

    return metric
