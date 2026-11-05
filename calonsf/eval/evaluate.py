"""Full evaluation: CaloNSF + every trained baseline on the test split.
Image metrics, energy closure, physics observables (overall and per class / particle / energy bin),
FPD/KPD, classifier two-sample test, downstream tasks (trained on GT only), uncertainty calibration,
and the CaloGAN two-shower separation test. Only computed numbers are written."""
from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import roc_curve

from ..data.degrade import make_loader
from ..downstream.models import AttentiveProbe, make_task_model, probe_target
from ..downstream.tasks import classifier_test, probe_metrics, run_model, task_metrics, two_shower_test
from ..models.baselines import make_baseline
from ..models.encoder import CellEncoder, Conditioner
from ..physics import stats
from ..physics.observables import KEY_OBS, observables
from ..utils.misc import Timer, free_cache, get_logger, save_json, to_device
from . import plots
from .metrics import calibration, energy_metrics, image_metrics

log = get_logger()


def _cat(lst):
    return {g: np.concatenate([x[g] for x in lst]) for g in lst[0]}


def _np(grids):
    return {g: v.detach().float().cpu().numpy() for g, v in grids.items()}


def load_baselines(cfg, geom, base, factor, device):
    out = {}
    for name in cfg.eval.get("baselines", []):
        p = os.path.join(base, "baselines", f"{name}_x{factor}", "best.pt")
        m = make_baseline(name, cfg, geom, factor).to(device)
        if not m.trainable:
            out[name] = m.eval()
            continue
        if os.path.exists(p):
            m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model"])
            if hasattr(m, "stage"):
                m.stage = 2
            out[name] = m.eval()
        else:
            log.info(f"[eval] baseline {name} x{factor}: no checkpoint, skipped")
    return out


def collect(cfg, geom, reader, idx, factor, model, baselines, device):
    ev = cfg.eval
    dl = make_loader(cfg, reader, idx[: int(ev.n_events)], geom, False, factors=[factor],
                     batch_size=int(ev.batch_size), shuffle=False)
    acc = {"gt": [], "lr": [], "lr_init": [], "calonsf": [], "std": []}
    acc.update({n: [] for n in baselines})
    meta = {"label": [], "e_inc": [], "particle": []}
    times = {n: 0.0 for n in ["calonsf", *baselines]}
    for batch in dl:
        b = to_device(batch, device)
        acc["gt"].append(_np(batch["hr"]))
        acc["lr"].append(_np(batch["lr"]))
        if model is not None:
            with Timer() as t:
                o = model.generate(b, ensemble=int(ev.ensemble), nfe=int(ev.nfe), solver=ev.solver)
            times["calonsf"] += t.elapsed
            acc["calonsf"].append(_np(o["grids"]))
            acc["std"].append(_np(o["std"]))
            acc["lr_init"].append(_np(o["lr_init"]))
        for n, m in baselines.items():
            with Timer() as t:
                p = m.predict(b)
            times[n] += t.elapsed
            acc[n].append(_np(p))
        for k in meta:
            meta[k].append(batch[k].numpy())
        free_cache(device)
    out = {k: _cat(v) for k, v in acc.items() if v}
    meta = {k: np.concatenate(v) for k, v in meta.items()}
    n = len(meta["label"])
    return out, meta, {k: v / max(n, 1) for k, v in times.items()}


def _load_downstream(cfg, geom, base, device):
    task, probe = None, None
    p = os.path.join(base, "downstream", "task", "best.pt")
    if os.path.exists(p):
        task = make_task_model(geom, cfg).to(device)
        task.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model"])
    p = os.path.join(base, "downstream", "probe", "best.pt")
    if os.path.exists(p):
        n_out, _ = probe_target(geom)
        probe = AttentiveProbe(CellEncoder(cfg.model, geom), Conditioner(int(cfg.model.d_model),
                               cfg.model.get("conditioning", {})), geom, n_out, int(cfg.model.d_model),
                               int(cfg.model.heads)).to(device)
        probe.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model"])
    return task, probe


def _features(geom, grids, obs):
    f = stats.feature_matrix(obs)
    if geom.name != "qg":  # voxel-level features for the small detectors (CaloChallenge-style)
        f = np.concatenate([f] + [np.log1p(np.clip(v, 0, None) / geom.energy_scale).reshape(len(v), -1)
                                  for _, v in sorted(grids.items())], 1)
    return f


def run_evaluation(cfg, geom, reader, splits, model, base, device) -> dict:
    out_dir = plots.ensure(os.path.join(base, "eval"))
    task, probe = _load_downstream(cfg, geom, base, device)
    results = {"dataset": geom.name, "factors": {}}
    for factor in cfg.eval.lr_factors:
        factor = int(factor)
        bl = load_baselines(cfg, geom, base, factor, device)
        log.info(f"[eval] factor x{factor}: CaloNSF + baselines {list(bl)}")
        data, meta, times = collect(cfg, geom, reader, splits["test"], factor, model, bl, device)
        gt = data["gt"]
        methods = [m for m in ["calonsf", *bl] if m in data]
        obs_gt = observables(gt, geom, meta)
        subs = stats.subsets(meta, cfg, geom.name)
        feats_gt = _features(geom, gt, obs_gt)
        R = {"n_events": int(len(meta["label"])), "methods": {}}
        obs_all = {}
        down_ref = {}
        if task is not None:
            down_ref["task"] = task_metrics(geom, run_model(task, gt, meta, device), meta, cfg)
        if probe is not None:
            down_ref["probe"] = probe_metrics(geom, run_model(probe, gt, meta, device, bs=16), meta, cfg)
        R["downstream_on_gt"] = down_ref
        rocs = {}
        for m in methods:
            pred = data[m]
            ob = observables(pred, geom, meta)
            obs_all[m] = ob
            r = {"time_per_event_s": times.get(m, float("nan")),
                 "image": image_metrics(gt, pred, geom.energy_scale),
                 "energy": energy_metrics(gt, pred, data["lr_init"] if m == "calonsf" else data["lr"]),
                 "observables": stats.compare(obs_gt, ob),
                 "observables_by_subset": {s: stats.compare(obs_gt, ob, mk) for s, mk in subs.items() if s != "all"}}
            fp = _features(geom, pred, ob)
            hlf_gt, hlf_p = stats.feature_matrix(obs_gt), stats.feature_matrix(ob)
            try:
                r["FPD"] = stats.fpd(hlf_gt, hlf_p)
                r["KPD"] = stats.kpd(hlf_gt, hlf_p, batch=min(500, len(hlf_gt)))
            except Exception as e:  # singular covariances on tiny eval sets
                log.warning(f"[eval] FPD/KPD failed for {m}: {e}")
            r["classifier_auc"] = classifier_test(feats_gt, fp, seed=int(cfg.seed))
            if task is not None:
                o = run_model(task, pred, meta, device)
                r["downstream_task"] = task_metrics(geom, o, meta, cfg)
                r["downstream_task_by_subset"] = {s: task_metrics(geom, (o[mk] if not isinstance(o, dict) else
                                                  {k: (v[mk] if v is not None else None) for k, v in o.items()}),
                                                  {k: v[mk] for k, v in meta.items()}, cfg)
                                                  for s, mk in subs.items() if s != "all" and mk.sum() > 10}
                if geom.name == "qg":
                    p = 1 / (1 + np.exp(-o))
                    fpr, tpr, _ = roc_curve(meta["label"], p)
                    rocs[m] = (fpr, tpr, r["downstream_task"].get("roc_auc", float("nan")))
            if probe is not None:
                r["downstream_probe"] = probe_metrics(geom, run_model(probe, pred, meta, device, bs=16), meta, cfg)
            if m == "calonsf" and int(cfg.eval.ensemble) > 1:
                r["uncertainty"] = calibration(gt, pred, data["std"])
            key = [k for k in KEY_OBS[geom.name] if k in r["observables"]]
            r["key_obs_mean_w1_norm"] = float(np.mean([r["observables"][k]["w1_norm"] for k in key])) if key else None
            R["methods"][m] = r
        if task is not None and geom.name == "qg":
            o = run_model(task, gt, meta, device)
            fpr, tpr, _ = roc_curve(meta["label"], 1 / (1 + np.exp(-o)))
            rocs = {"gt": (fpr, tpr, down_ref["task"].get("roc_auc", float("nan"))), **rocs}
        # figures
        fdir = plots.ensure(os.path.join(out_dir, f"x{factor}"))
        others = [m for m in bl if m != "nearest"]
        best_bl = min(others, key=lambda m: R["methods"][m]["key_obs_mean_w1_norm"] or 1e9) if others else None
        show = [m for m in ["calonsf", "nearest", best_bl] if m and m in obs_all]
        if "calonsf" in data:
            plots.plot_examples(os.path.join(fdir, "examples.png"), geom, gt, data["nearest"] if "nearest" in data else data["lr"],
                                data, data.get("std"), n=min(int(cfg.eval.n_examples), len(meta["label"])),
                                methods=[m for m in ["calonsf", best_bl] if m])
        plots.plot_observables(os.path.join(fdir, "observables.png"), obs_gt, obs_all, KEY_OBS[geom.name], show)
        if rocs:
            sel = {k: v for k, v in rocs.items() if k in ("gt", *show)}
            plots.plot_roc(os.path.join(fdir, "roc.png"), sel)
        if geom.name == "calogan" and model is not None:
            from ..data.geometry import lr_factors_for
            ts = two_shower_test(model, geom, gt, meta, lr_factors_for(geom, factor), device,
                                 n_pairs=min(64, len(meta["label"])))
            R["two_shower"] = ts
            plots.plot_curve(os.path.join(fdir, "two_shower.png"), ts["shift_cells"],
                             {"nearest": ts["lr_agree"], "calonsf": ts["sr_agree"]},
                             "separation [layer-1 cells]", "peak count matches ground truth", "Two-shower separation")
        R["best_baseline"] = best_bl
        results["factors"][str(factor)] = R
    save_json(results, os.path.join(out_dir, "metrics.json"))
    return results
