"""Image-level metrics. Computed per layer group on t = log1p(E/E_scale) (where PSNR/SSIM make sense on
sparse, heavy-tailed data) and on raw energy where noted."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _ssim(a: torch.Tensor, b: torch.Tensor, rng: torch.Tensor) -> torch.Tensor:
    """SSIM of (M, H, W) image pairs with a uniform window (size <= 7, fits tiny calorimeter layers)."""
    H, W = a.shape[-2:]
    ws = min(7, H, W)
    ws = ws if ws % 2 == 1 else ws - 1
    ws = max(ws, 1)
    a, b = a.unsqueeze(1), b.unsqueeze(1)
    k = torch.ones(1, 1, ws, ws) / ws ** 2
    mu_a, mu_b = F.conv2d(a, k), F.conv2d(b, k)
    saa = F.conv2d(a * a, k) - mu_a ** 2
    sbb = F.conv2d(b * b, k) - mu_b ** 2
    sab = F.conv2d(a * b, k) - mu_a * mu_b
    R = rng.view(-1, 1, 1, 1)
    c1, c2 = (0.01 * R) ** 2, (0.03 * R) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return s.mean((1, 2, 3))


def image_metrics(gt: dict, pred: dict, energy_scale: float) -> dict:
    out = {}
    for g in gt:
        a = torch.as_tensor(gt[g]).float()
        b = torch.as_tensor(pred[g]).float()
        N, n, H, W = a.shape
        ta, tb = torch.log1p(a.clamp(min=0) / energy_scale), torch.log1p(b.clamp(min=0) / energy_scale)
        err = tb - ta
        mse = (err ** 2).mean((-1, -2))  # (N, n)
        rng = ta.amax((-1, -2))
        ok = rng > 0
        psnr = 10 * torch.log10(rng ** 2 / mse.clamp(min=1e-12))
        ssim = _ssim(ta.reshape(-1, H, W), tb.reshape(-1, H, W), rng.reshape(-1).clamp(min=1e-6)).view(N, n)
        occ_gt, occ_pr = a > 0, b > 0
        tp = (occ_gt & occ_pr).sum().float()
        Eg = a.sum((-1, -2))
        out[g] = {
            "MAE": float(err.abs().mean()), "MSE": float(mse.mean()), "RMSE": float(mse.mean().sqrt()),
            "PSNR": float(psnr[ok].mean()) if ok.any() else float("nan"),
            "SSIM": float(ssim[ok].mean()) if ok.any() else float("nan"),
            "MAE_raw_rel": float((b - a).abs().sum((-1, -2))[Eg > 0].div(Eg[Eg > 0]).mean()),
            "occ_precision": float(tp / occ_pr.sum().clamp(min=1)),
            "occ_recall": float(tp / occ_gt.sum().clamp(min=1)),
            "sparsity_gt": float(occ_gt.float().mean()), "sparsity_pred": float(occ_pr.float().mean()),
        }
    return out


def energy_metrics(gt: dict, pred: dict, ref: dict | None = None) -> dict:
    """Response sum(SR)/sum(GT) and, if ref (the LR the model started from) is given, the conservation
    error |sum(SR) - sum(ref)| / sum(ref) (exactly 0 for CaloNSF by construction)."""
    eg = sum(torch.as_tensor(gt[g]).cpu().double().sum((1, 2, 3)) for g in gt)
    ep = sum(torch.as_tensor(pred[g]).cpu().double().sum((1, 2, 3)) for g in pred)
    ok = eg > 0
    r = (ep[ok] / eg[ok]).numpy()
    out = {"response_mean": float(r.mean()), "response_std": float(r.std())}
    if ref is not None:
        er = sum(torch.as_tensor(ref[g]).cpu().double().sum((1, 2, 3)) for g in ref)
        okr = er > 0
        c = ((ep[okr] - er[okr]).abs() / er[okr]).numpy()
        out["conservation_err_max"] = float(c.max()) if len(c) else 0.0
        out["conservation_err_mean"] = float(c.mean()) if len(c) else 0.0
    # per-layer energy closure
    out["layer_rel_err"] = {g: float(((torch.as_tensor(pred[g]).sum((-1, -2)) - torch.as_tensor(gt[g]).sum((-1, -2))).abs()
                                      .sum() / torch.as_tensor(gt[g]).sum().clamp(min=1e-12))) for g in gt}
    return out


def calibration(gt: dict, mean: dict, std: dict, ks=(1, 2)) -> dict:
    """Fraction of non-empty GT cells inside mean +- k*sigma of the ensemble (Gaussian ideal: 0.68, 0.95)."""
    out = {}
    a = np.concatenate([np.asarray(gt[g]).ravel() for g in gt])
    m = np.concatenate([np.asarray(mean[g]).ravel() for g in gt])
    s = np.concatenate([np.asarray(std[g]).ravel() for g in gt])
    sel = a > 0
    if sel.sum() == 0 or s.max() == 0:
        return {}
    for k in ks:
        out[f"cover_{k}sigma"] = float((np.abs(m[sel] - a[sel]) <= k * s[sel] + 1e-12).mean())
    out["mean_rel_sigma"] = float((s[sel] / (m[sel] + 1e-12)).mean())
    return out
