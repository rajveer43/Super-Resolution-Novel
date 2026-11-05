"""Running downstream models on any set of grids + metrics, the classifier two-sample test, and the
two-shower separation test."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from ..data.ops import pool_grids, shift_energy
from ..utils.misc import free_cache
from .models import probe_target


@torch.no_grad()
def run_model(model, grids: dict, meta: dict, device, bs: int = 64, probe: bool = False):
    model.eval()
    N = len(next(iter(grids.values())))
    outs = []
    for s in range(0, N, bs):
        g = {k: torch.as_tensor(v[s:s + bs]).float().to(device) for k, v in grids.items()}
        b = {"e_inc": torch.as_tensor(meta["e_inc"][s:s + bs]).float().to(device),
             "particle": torch.as_tensor(meta["particle"][s:s + bs]).long().to(device),
             "label": torch.as_tensor(meta["label"][s:s + bs]).long().to(device)}
        o = model(g, b)
        if isinstance(o, dict):
            outs.append({k: (v.cpu().numpy() if v is not None else None) for k, v in o.items()})
        else:
            outs.append(o.cpu().numpy())
        free_cache(device)
    if isinstance(outs[0], dict):
        return {k: (np.concatenate([o[k] for o in outs]) if outs[0][k] is not None else None) for k in outs[0]}
    return np.concatenate(outs)


def _energy_res(pred_e, true_e, bins):
    rel = (pred_e - true_e) / true_e
    out = {"rel_bias": float(np.mean(rel)), "rel_resolution": float(np.std(rel))}
    if bins:
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (true_e >= lo) & (true_e < hi)
            if m.sum() >= 5:
                out[f"resolution_E[{lo},{hi})"] = float(np.std(rel[m]))
    return out


def task_metrics(geom, out, meta: dict, cfg) -> dict:
    lab = np.asarray(meta["label"])
    bins = cfg.analysis.get("energy_bins", None)
    if geom.name == "qg":
        p = 1 / (1 + np.exp(-out))
        res = {"accuracy": float(accuracy_score(lab, p > 0.5))}
        if len(np.unique(lab)) == 2:
            res["roc_auc"] = float(roc_auc_score(lab, p))
        return res
    res = {}
    e = np.asarray(meta["e_inc"])
    res.update(_energy_res(np.exp(out["log_e"]), e, bins))
    if out.get("logits") is not None:
        lg = out["logits"]
        pr = np.exp(lg - lg.max(1, keepdims=True))
        pr /= pr.sum(1, keepdims=True)
        res["accuracy"] = float(accuracy_score(lab, pr.argmax(1)))
        if len(np.unique(lab)) == pr.shape[1]:
            res["roc_auc_ovr"] = float(roc_auc_score(lab, pr, multi_class="ovr"))
        names = cfg.analysis.get("class_names", None) or [str(i) for i in range(pr.shape[1])]
        for i, n in enumerate(names):
            m = lab == i
            if m.any():
                res[f"acc_{n}"] = float((pr[m].argmax(1) == i).mean())
    return res


def probe_metrics(geom, out, meta, cfg) -> dict:
    n, kind = probe_target(geom)
    lab = np.asarray(meta["label"])
    if kind == "binary":
        p = 1 / (1 + np.exp(-out[:, 0]))
        return {"accuracy": float(accuracy_score(lab, p > 0.5)),
                "roc_auc": float(roc_auc_score(lab, p)) if len(np.unique(lab)) == 2 else float("nan")}
    if kind == "multiclass":
        return {"accuracy": float(accuracy_score(lab, out.argmax(1)))}
    return _energy_res(np.exp(out[:, 0]), np.asarray(meta["e_inc"]), cfg.analysis.get("energy_bins", None))


def classifier_test(feats_gt: np.ndarray, feats_pred: np.ndarray, seed: int = 0) -> float:
    """CaloChallenge-style classifier test: AUC of a classifier separating GT from SR on held-out data.
    0.5 = indistinguishable (best), 1.0 = trivially separable."""
    n = min(len(feats_gt), len(feats_pred))
    rng = np.random.default_rng(seed)
    X = np.concatenate([feats_gt[:n], feats_pred[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    X = np.nan_to_num(X)
    perm = rng.permutation(2 * n)
    X, y = X[perm], y[perm]
    h = n
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    clf.fit(X[:h], y[:h])
    p = clf.predict_proba(X[h:])[:, 1]
    if len(np.unique(y[h:])) < 2:
        return float("nan")
    auc = roc_auc_score(y[h:], p)
    return float(max(auc, 1 - auc))


# ------------------------------------------------------------------------------------- two-shower test
def _count_peaks(img: torch.Tensor, rel: float = 0.2) -> torch.Tensor:
    """Local maxima (3x3) above rel * max in (N, H, W)."""
    g = torch.Generator().manual_seed(0)
    img = img + 1e-6 * img.amax((-1, -2), keepdim=True) * torch.rand(img.shape, generator=g)  # break ties
    mx = torch.nn.functional.max_pool2d(img.unsqueeze(1), 3, 1, 1).squeeze(1)
    thr = rel * img.amax((-1, -2), keepdim=True)
    return ((img == mx) & (img > thr) & (img > 0)).sum((-1, -2))


def two_shower_test(model, geom, hr: dict, meta: dict, factors: dict, device, shifts=(0.5, 1.0, 1.5, 2.0, 3.0),
                    n_pairs: int = 64, ensemble: int = 1, seed: int = 0, ref_layer: str = "layer_1") -> dict:
    """Overlay two showers shifted by d cells of `ref_layer` along v, downsample, super-resolve, and count
    resolved peaks in `ref_layer` (GT and SR at native resolution, LR at its own granularity)."""
    rng = np.random.default_rng(seed)
    N = len(meta["e_inc"])
    W_ref = geom.group(ref_layer).shape[1]
    # gt/lr/sr: fraction of overlays with exactly 2 peaks; *_agree: fraction whose peak count equals GT's
    res = {"shift_cells": list(shifts), "gt": [], "lr": [], "sr": [], "lr_agree": [], "sr_agree": []}
    for d in shifts:
        i = rng.choice(N, n_pairs)
        j = rng.choice(N, n_pairs)
        ov = {}
        for g in geom.group_names:
            a = torch.as_tensor(hr[g][i]).float()
            b = torch.as_tensor(hr[g][j]).float()
            s = d * geom.group(g).shape[1] / W_ref
            ov[g] = a + shift_energy(b, s, -1)
        lr = pool_grids(ov, factors)
        batch = {"lr": {g: v.to(device) for g, v in lr.items()}, "factors": factors,
                 "e_inc": torch.as_tensor(meta["e_inc"][i] + meta["e_inc"][j]).float().to(device),
                 "particle": torch.as_tensor(meta["particle"][i]).long().to(device)}
        sr = model.generate(batch, ensemble=ensemble)["grids"][ref_layer].cpu()[:, 0]
        free_cache(device)
        up = lr[ref_layer][:, 0]  # peaks counted at LR granularity
        gt = ov[ref_layer][:, 0]
        ng, nl, ns = _count_peaks(gt), _count_peaks(up), _count_peaks(sr)
        res["gt"].append(float((ng == 2).float().mean()))
        res["lr"].append(float((nl == 2).float().mean()))
        res["sr"].append(float((ns == 2).float().mean()))
        res["lr_agree"].append(float((nl == ng).float().mean()))
        res["sr_agree"].append(float((ns == ng).float().mean()))
    return res
