"""Physics observables computed from energy grids at ANY resolution (geometry-aware coordinates).

All functions take grids {group: tensor (N, n_layers, H, W)} (torch or numpy) and return
{name: np.ndarray of shape (N,) or (N, k)}.

QG       : per-channel energy & multiplicity, jet mass proxy, girth/width, pT_D, core fraction, radial profile.
CaloGAN  : CaloGAN paper shower shapes (layer energies/fractions, shower depth + width, lateral widths,
           sparsity, E_ratio).
CaloChal.: CaloChallenge high-level features (E per layer, centre of energy & width in x/y per layer,
           sparsity per layer, E_tot/E_inc, longitudinal and radial profiles).
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _t(x) -> torch.Tensor:
    return x.detach().float().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _axis_coords(n: int, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
    return lo + (torch.arange(n).float() + 0.5) * (hi - lo) / n


def _wmean_std(img: torch.Tensor, c: torch.Tensor, dim: int):
    """Energy-weighted mean / std of coordinate c along `dim` of img (N, H, W).
    float64: E[c^2] - m^2 cancels badly in float32 (a single-cell shower would get width ~0.02 mm, not 0)."""
    prof = img.double().sum(dim=dim)  # (N, L)
    c = c.double()
    tot = prof.sum(-1).clamp(min=1e-12)
    m = (prof * c).sum(-1) / tot
    v = (prof * c ** 2).sum(-1) / tot - m ** 2
    return m.float(), v.clamp(min=0).sqrt().float()


# ---------------------------------------------------------------------------------- QG
def qg_observables(grids: dict, geom, mult_threshold: float = 0.0) -> dict:
    x = _t(grids["jet"])  # (N, 3, H, W)
    N, C, H, W = x.shape
    hw = float(geom.extras.get("halfwidth", 1.0))
    u = _axis_coords(H) * hw
    v = _axis_coords(W) * hw
    U, V = torch.meshgrid(u, v, indexing="ij")
    out = {}
    names = ["tracks", "ecal", "hcal"]
    for c in range(C):
        out[f"E_{names[c]}"] = x[:, c].sum((-1, -2))
        out[f"mult_{names[c]}"] = (x[:, c] > mult_threshold).float().sum((-1, -2))
    img = x.sum(1)
    E = img.sum((-1, -2)).clamp(min=1e-12)
    cu = (img * U).sum((-1, -2)) / E
    cv = (img * V).sum((-1, -2)) / E
    r2 = (U[None] - cu[:, None, None]) ** 2 + (V[None] - cv[:, None, None]) ** 2
    r = r2.sqrt()
    # massless-constituent small-angle mass: m^2 = sum_{i<j} E_i E_j dR_ij^2 = E * sum E_i r_i^2 (about centroid)
    out["mass"] = (E * (img * r2).sum((-1, -2))).clamp(min=0).sqrt()
    out["girth"] = (img * r).sum((-1, -2)) / E
    out["ptd"] = (img ** 2).sum((-1, -2)).sqrt() / E
    out["core_frac"] = (img * (r < 0.1 * hw)).sum((-1, -2)) / E
    for c, nm in ((1, "ecal"), (2, "hcal")):
        Ec = x[:, c].sum((-1, -2)).clamp(min=1e-12)
        out[f"width_{nm}"] = (x[:, c] * r).sum((-1, -2)) / Ec
    edges = torch.linspace(0, 0.8 * hw, 9)
    prof = torch.stack([(img * ((r >= lo) & (r < hi))).sum((-1, -2)) / E for lo, hi in zip(edges[:-1], edges[1:])], 1)
    out["radial_profile"] = prof
    return {k: v.numpy() for k, v in out.items()}


# ---------------------------------------------------------------------------------- CaloGAN
def calogan_observables(grids: dict, geom) -> dict:
    ext = float(geom.extras.get("extent_mm", 480.0))
    out = {}
    Es = []
    for k in range(3):
        x = _t(grids[f"layer_{k}"])[:, 0]  # (N, H, W)
        N, H, W = x.shape
        E = x.sum((-1, -2))
        Es.append(E)
        out[f"E_{k}"] = E
        cu = _axis_coords(H, -ext / 2, ext / 2)
        cv = _axis_coords(W, -ext / 2, ext / 2)
        _, su = _wmean_std(x, cu, dim=2)
        _, sv = _wmean_std(x, cv, dim=1)
        out[f"width_eta_{k}"] = su
        out[f"width_phi_{k}"] = sv
        out[f"sparsity_{k}"] = (x > 0).float().mean((-1, -2))
        flat = x.reshape(N, -1)
        top = flat.topk(min(2, flat.shape[1]), dim=1).values
        if top.shape[1] == 2:
            out[f"E_ratio_{k}"] = (top[:, 0] - top[:, 1]) / (top[:, 0] + top[:, 1]).clamp(min=1e-12)
    Et = sum(Es).clamp(min=1e-12)
    out["E_tot"] = Et
    for k in range(3):
        out[f"frac_{k}"] = Es[k] / Et
    d = sum(k * Es[k] for k in range(3)) / Et
    out["shower_depth"] = d
    out["depth_width"] = (sum(k ** 2 * Es[k] for k in range(3)) / Et - d ** 2).clamp(min=0).sqrt()
    return {k: v.numpy() for k, v in out.items()}


# ---------------------------------------------------------------------------------- CaloChallenge
def _r_centers(edges: list[float], n: int) -> torch.Tensor:
    """Radial bin centres at resolution n, subdividing the physical edges uniformly."""
    e = np.asarray(edges, dtype=np.float64)
    nb = len(e) - 1
    pos = np.linspace(0, nb, n + 1)
    fine = np.interp(pos, np.arange(nb + 1), e)
    return torch.tensor(0.5 * (fine[1:] + fine[:-1]), dtype=torch.float32)


def calochallenge_observables(grids: dict, geom, e_inc=None) -> dict:
    x = _t(grids["shower"])  # (N, Z, A, R)
    N, Z, A, R = x.shape
    rc = _r_centers(geom.extras.get("r_edges", list(range(10))), R)
    ac = (torch.arange(A).float() + 0.5) * 2 * math.pi / A
    X = rc[None, :] * torch.cos(ac)[:, None]  # (A, R)
    Y = rc[None, :] * torch.sin(ac)[:, None]
    El = x.sum((-1, -2))  # (N, Z)
    tot = El.clamp(min=1e-12)
    cx = (x * X).sum((-1, -2)) / tot
    cy = (x * Y).sum((-1, -2)) / tot
    wx = ((x * X ** 2).sum((-1, -2)) / tot - cx ** 2).clamp(min=0).sqrt()
    wy = ((x * Y ** 2).sum((-1, -2)) / tot - cy ** 2).clamp(min=0).sqrt()
    Et = El.sum(1).clamp(min=1e-12)
    out = {"E_layer": El, "CoE_x": cx, "CoE_y": cy, "width_x": wx, "width_y": wy,
           "sparsity": (x > 0).float().mean((-1, -2)), "E_tot": Et,
           "long_profile": El / Et[:, None],
           "radial_profile": x.sum((1, 2)) / Et[:, None]}
    if e_inc is not None:
        out["E_ratio_inc"] = Et / _t(e_inc).clamp(min=1e-12)
    return {k: v.numpy() for k, v in out.items()}


def observables(grids: dict, geom, meta: dict | None = None) -> dict:
    if geom.name == "qg":
        return qg_observables(grids, geom)
    if geom.name == "calogan":
        return calogan_observables(grids, geom)
    if geom.name == "calochallenge":
        return calochallenge_observables(grids, geom, None if meta is None else meta.get("e_inc"))
    raise ValueError(geom.name)


# key observables for plots / headline tables
KEY_OBS = {
    "qg": ["mass", "girth", "ptd", "core_frac", "mult_ecal", "E_hcal"],
    "calogan": ["E_tot", "frac_0", "shower_depth", "width_eta_1", "width_phi_1", "E_ratio_0", "sparsity_1"],
    "calochallenge": ["E_ratio_inc", "width_x", "CoE_x", "sparsity", "long_profile", "radial_profile"],
}

# energy-weighted observables that should NOT change when the same energy is drawn at finer granularity
# (multiplicity, sparsity, pT_D and E_ratio depend on granularity by definition)
STABLE_OBS = {
    "qg": ["mass", "girth", "core_frac", "width_ecal", "width_hcal"],
    "calogan": ["E_tot", "frac_0", "frac_1", "shower_depth", "width_eta_1", "width_phi_1"],
    "calochallenge": ["E_ratio_inc", "width_x", "width_y", "long_profile"],
}
