"""CaloChallenge Dataset 2 evaluation of CaloNSF (and the nearest / bicubic reference upsamplers).

For every LR factor in eval.lr_factors the HR showers of the test split are the "Geant4" reference and each
method's super-resolution of their LR version is the "submission". The metrics are those of the CaloChallenge
(Fischer et al., section 8): separation power of the high-level feature histograms, layer-energy PCC,
binary classifier AUCs (DNN low / high level, 3D ResNet), FPD / KPD, precision / recall / density / coverage,
generation time and parameter counts. Only computed numbers are written."""
from __future__ import annotations

import os

import numpy as np
import torch

from ..data.degrade import make_loader
from ..models.baselines import make_baseline
from ..utils.misc import free_cache, get_logger, save_json, to_device
from . import classifier, correlations, distances, features, manifold, plots, resources, separation

log = get_logger()


def _np(grids: dict) -> np.ndarray:
    return grids["shower"].detach().float().cpu().numpy()


def collect(cfg, geom, reader, idx, factor: int, model, baselines: dict, device, threshold: float = 0.0):
    """HR truth and each method's SR output for the test events: ({name: (N, Z, A, R)}, e_inc (N,)). Every method's
    output gets the detector readout cutoff `threshold` (MeV), as the challenge does for all submissions."""
    ev = cfg.eval
    n = len(idx) if ev.get("n_events") in (None, 0) else min(int(ev.n_events), len(idx))
    dl = make_loader(cfg, reader, idx[:n], geom, False, factors=[factor], batch_size=int(ev.batch_size), shuffle=False)
    acc = {"gt": [], **({"calonsf": []} if model is not None else {}), **{k: [] for k in baselines}}
    e_inc = []
    for batch in dl:
        b = to_device(batch, device)
        acc["gt"].append(_np(batch["hr"]))
        if model is not None:
            with torch.no_grad():
                acc["calonsf"].append(features.readout_cutoff(_np(model.generate(b, ensemble=1, nfe=int(ev.nfe), solver=ev.solver)["grids"]), threshold))
        for k, m in baselines.items():
            acc[k].append(features.readout_cutoff(_np(m.predict(b)), threshold))
        e_inc.append(batch["e_inc"].numpy())
        free_cache(device)
    return {k: np.concatenate(v) for k, v in acc.items()}, np.concatenate(e_inc)


def _split_halves(n: int, seed: int):
    p = np.random.default_rng(seed).permutation(n)
    return p[: n // 2], p[n // 2: 2 * (n // 2)]


def run_evaluation(cfg, geom, reader, splits, model, base, device) -> dict:
    if geom.name != "calochallenge":
        raise ValueError("the evaluation implements the CaloChallenge Dataset 2 metrics; use a calochallenge config")
    ev = cfg.eval
    out_dir = os.path.join(base, "eval")
    os.makedirs(out_dir, exist_ok=True)
    r_edges = geom.extras["r_edges"]
    thr = float(ev.threshold_mev)
    cc = ev.classifier
    results = {"dataset": "calochallenge_dataset2", "threshold_mev": thr, "factors": {}}
    for factor in ev.lr_factors:
        factor = int(factor)
        bl = {n: make_baseline(n, cfg, geom, factor) for n in ev.get("baselines", [])}
        log.info(f"[eval] x{factor}: CaloNSF + {list(bl)} on {min(len(splits['test']), int(ev.n_events or 10**12))} events")
        data, e_inc = collect(cfg, geom, reader, splits["test"], factor, model, bl, device, thr)
        ref = data.pop("gt")
        n = len(ref)
        methods = list(data)

        # ---- reference (Geant4) quantities
        f_ref = features.high_level_features(ref, e_inc, r_edges, thr)
        edges = separation.make_edges(f_ref, int(ev.n_bins))
        vedges = separation.voxel_edges(ref, int(ev.n_bins))
        vox_ref = separation.voxel_event_hists(ref, vedges)
        h_ref = separation.feature_hists(f_ref, edges)
        h_ref["voxel_energy"] = [vox_ref.sum(0)]
        band = separation.geant4_band(f_ref, vox_ref, edges, int(ev.band_pairs), int(cfg.seed))
        hlf_ref = features.feature_matrix(f_ref)
        pcc_ref = correlations.layer_pcc(f_ref["E_layer"])
        a, b = _split_halves(n, int(cfg.seed))
        # Manifold metrics use DISJOINT events for the reference and the generated set: a generated shower is
        # the super-resolution of a reference shower, so with the same events each point would sit next to its
        # own truth and precision / recall / coverage would be trivially ~1. The permutation keeps the E_inc
        # distributions of the two halves the same.
        mr, mg = _split_halves(n, int(cfg.seed) + 1)
        mr, mg = mr[: int(ev.manifold.events)], mg[: int(ev.manifold.events)]
        mf_real = manifold.manifold_features(ref[mr])
        g4 = {"separation_band": band,
              "fpd": distances.fpd(hlf_ref[a], hlf_ref[b]),
              "kpd": distances.kpd(hlf_ref[a], hlf_ref[b], int(ev.kpd_batch))[0]}
        if len(mr) > int(ev.manifold.k) + 1:
            g4["manifold"] = manifold.prdc(mf_real, manifold.manifold_features(ref[mg]), int(ev.manifold.k), device)
        R = {"n_events": int(n), "geant4_vs_geant4": g4, "methods": {}}
        plot_data = {"h": {}, "pcc": {}, "sep": {}}

        for m in methods:
            pred = data[m]
            f_p = features.high_level_features(pred, e_inc, r_edges, thr)
            h_p = separation.feature_hists(f_p, edges)
            h_p["voxel_energy"] = [separation.voxel_event_hists(pred, vedges).sum(0)]
            sep = separation.sep_from_hists(h_ref, h_p)
            hlf_p = features.feature_matrix(f_p)
            pcc_p = correlations.layer_pcc(f_p["E_layer"])
            r = {"separation_power": {"summary": separation.summarize(sep), "per_layer": sep},
                 "pcc": correlations.pcc_difference(pcc_ref, pcc_p),
                 "fpd": distances.fpd(hlf_ref, hlf_p)}
            kmean, kstd = distances.kpd(hlf_ref, hlf_p, int(ev.kpd_batch))
            r["kpd"], r["kpd_std"] = kmean, kstd
            r["manifold"] = manifold.prdc(mf_real, manifold.manifold_features(pred[mg]), int(ev.manifold.k), device)
            src_ref, src_gen = classifier._Source(ref, e_inc, hlf_ref), classifier._Source(pred, e_inc, hlf_p)
            r["classifier"] = {}
            for kind in cc.run:
                dnn = kind != "cnn_low"
                log.info(f"[eval] x{factor} {m}: classifier {kind} ({int(cc.seeds)} seeds)")
                r["classifier"][kind] = classifier.run_binary(
                    kind, src_ref, src_gen, seeds=int(cc.seeds), epochs=int(cc.dnn_epochs if dnn else cc.cnn_epochs),
                    lr=float(cc.dnn_lr if dnn else cc.cnn_lr), batch=int(cc.dnn_batch if dnn else cc.cnn_batch),
                    device=device, seed0=int(cfg.seed))
            R["methods"][m] = r
            plot_data["h"][m], plot_data["pcc"][m], plot_data["sep"][m] = h_p, pcc_p, sep
            log.info(f"[eval] x{factor} {m}: mean separation power {r['separation_power']['summary']['mean_all']:.4f}, "
                     f"FPD {r['fpd']:.3f}, AUC " + ", ".join(f"{k} {v['auc_mean']:.3f}" for k, v in r["classifier"].items()))
            free_cache(device)

        # ---- generation time and memory
        gens = {k: (lambda bt, mm=mm: mm.predict(bt)) for k, mm in bl.items()}
        if model is not None:
            gens["calonsf"] = lambda bt: model.generate(bt, ensemble=1, nfe=int(ev.nfe), solver=ev.solver)
        R["timing_s_per_event"] = {k: resources.time_generation(g, cfg, reader, geom, splits["test"], factor,
                                                                list(ev.timing.batch_sizes), int(ev.timing.events), device)
                                   for k, g in gens.items()}
        if model is not None:
            R["parameters"] = resources.parameter_counts(model)
        R["device"] = str(device)
        results["factors"][str(factor)] = R

        fdir = os.path.join(out_dir, f"x{factor}")
        os.makedirs(fdir, exist_ok=True)
        info = f"CaloChallenge Dataset 2  |  LR factor $\\times${factor}  |  {n} test showers  |  reference: Geant4"
        plots.make_all(fdir, edges, vedges, h_ref, band, pcc_ref, plot_data, R, info)
    save_json(results, os.path.join(out_dir, "metrics.json"))
    return results
