"""CaloChallenge high-level features (Dataset 2), computed from voxel energies I_ia of shape (N, Z, A, R):

  E_layer          E_i = sum_a I_ia                                   (N, Z)
  E_tot_over_Einc  sum_{a,i} I_ia / E_inc                             (N,)
  CoE_{eta,phi,r}  sum_a l_a I_ia / sum_a I_ia, per layer             (N, Z)
  width_{eta,phi,r}  sqrt( sum_a l_a^2 I_ia / sum_a I_ia - CoE^2 )    (N, Z)
  sparsity         1 - fraction of voxels per layer with I_ia > threshold   (N, Z)

with eta_a = r_a cos(alpha_a), phi_a = r_a sin(alpha_a) and r_a, alpha_a the voxel centres (r: mean of the
bin boundaries). CoE and width are undefined for a layer with no energy: they are NaN there.
"""
from __future__ import annotations

import math

import numpy as np

LAYERED = ["E_layer", "CoE_eta", "CoE_phi", "CoE_r", "width_eta", "width_phi", "width_r", "sparsity"]
SCALAR = ["E_tot_over_Einc"]
LOG_SCALE = {"E_layer"}  # spans orders of magnitude (E_inc runs 1 GeV - 1 TeV): histogrammed in log10


def readout_cutoff(x: np.ndarray, threshold: float) -> np.ndarray:
    """The Par04 readout stores no cell energy below 0.5 keV = 15.15 keV after the 1/0.033 scaling, so Geant4
    showers have exact zeros there; the challenge applies the same cutoff to every submission before the
    evaluation (paper, sec. 2.2). Voxels with 0 < E < threshold are set to 0 (their energy is dropped)."""
    return np.where(x < threshold, 0.0, x).astype(x.dtype, copy=False) if threshold > 0 else x


def voxel_coords(r_edges, n_alpha: int, n_r: int):
    """(eta, phi, r) of the voxel centres, each (A, R). Radial bins are the mean of the boundary values."""
    e = np.asarray(r_edges, dtype=np.float64)
    assert len(e) == n_r + 1, f"r_edges has {len(e)} entries, expected {n_r + 1}"
    r = 0.5 * (e[1:] + e[:-1])
    a = (np.arange(n_alpha) + 0.5) * 2 * math.pi / n_alpha
    eta = r[None, :] * np.cos(a)[:, None]
    phi = r[None, :] * np.sin(a)[:, None]
    return eta, phi, np.broadcast_to(r[None, :], eta.shape)


def refine_r_edges(r_edges, n_r: int) -> np.ndarray:
    """Radial bin boundaries at a finer resolution: every native bin is split into equal parts."""
    e = np.asarray(r_edges, dtype=np.float64)
    nb = len(e) - 1
    return np.interp(np.linspace(0, nb, n_r + 1), np.arange(nb + 1), e)


def high_level_features(shower: np.ndarray, e_inc: np.ndarray, r_edges, threshold: float, chunk: int = 2000) -> dict:
    """shower (N, Z, A, R) in MeV, e_inc (N,) in MeV, threshold in MeV."""
    N, Z, A, R = shower.shape
    if len(r_edges) != R + 1:  # a refined (above-native) grid: split the native radial bins uniformly
        r_edges = refine_r_edges(r_edges, R)
    eta, phi, rr = voxel_coords(r_edges, A, R)
    coords = {"eta": eta, "phi": phi, "r": rr}
    out = {k: np.zeros((N, Z), np.float64) for k in LAYERED}
    out["E_tot_over_Einc"] = np.zeros(N, np.float64)
    for s in range(0, N, chunk):
        x = shower[s:s + chunk].astype(np.float64)
        El = x.sum((-1, -2))
        has = El > 0
        den = np.where(has, El, 1.0)
        out["E_layer"][s:s + chunk] = El
        out["E_tot_over_Einc"][s:s + chunk] = El.sum(1) / np.maximum(np.asarray(e_inc[s:s + chunk], np.float64), 1e-12)
        out["sparsity"][s:s + chunk] = 1.0 - (x > threshold).mean((-1, -2))
        for name, L in coords.items():
            m = (x * L).sum((-1, -2)) / den
            v = (x * L ** 2).sum((-1, -2)) / den - m ** 2
            out[f"CoE_{name}"][s:s + chunk] = np.where(has, m, np.nan)
            out[f"width_{name}"][s:s + chunk] = np.where(has, np.sqrt(np.clip(v, 0, None)), np.nan)
    return out


def feature_matrix(feats: dict) -> np.ndarray:
    """All high-level features as one (N, F) float32 matrix (undefined entries -> 0), for classifiers / FPD / KPD."""
    cols = [feats[k].reshape(len(feats[k]), -1) for k in [*SCALAR, *LAYERED]]
    return np.nan_to_num(np.concatenate(cols, 1), nan=0.0).astype(np.float32)


def polar_moments(x, r_edges, chunk: int = 9) -> dict:
    """Centre of energy and its width in eta, phi and r for every layer of LARGE grids x (B, Z, A, R) (torch, any
    device), the same quantities as `high_level_features` but without building (A, R)-sized coordinate arrays:
    sum_a l_a I_a is evaluated as cos(alpha) . (I @ r) etc., in float64 on the CPU, `chunk` layers at a time.
    Returns {"E": (B, Z), "CoE_eta" ..., "width_r": (B, Z)} as numpy arrays (NaN where a layer has no energy)."""
    import torch
    B, Z, A, R = x.shape
    e = np.asarray(r_edges, np.float64)
    e = e if len(e) == R + 1 else refine_r_edges(e, R)
    r = torch.as_tensor(0.5 * (e[1:] + e[:-1]))
    al = (torch.arange(A, dtype=torch.float64) + 0.5) * 2 * math.pi / A
    c, s_ = torch.cos(al), torch.sin(al)
    out = {k: np.full((B, Z), np.nan) for k in ("CoE_eta", "CoE_phi", "CoE_r", "width_eta", "width_phi", "width_r")}
    out["E"] = np.zeros((B, Z))
    for b in range(B):
        for z0 in range(0, Z, chunk):
            I = x[b, z0:z0 + chunk].detach().cpu().double()  # (z, A, R)
            Ir, Ir2 = I @ r, I @ (r * r)  # (z, A)
            E = Ir.new_zeros(Ir.shape[0]) + I.sum((-1, -2))
            ok = E > 0
            den = torch.where(ok, E, torch.ones_like(E))
            mom = {"eta": ((Ir * c).sum(-1), (Ir2 * c * c).sum(-1)), "phi": ((Ir * s_).sum(-1), (Ir2 * s_ * s_).sum(-1)),
                   "r": (Ir.sum(-1), Ir2.sum(-1))}
            sl = slice(z0, z0 + len(E))
            out["E"][b, sl] = E.numpy()
            for k, (m1, m2) in mom.items():
                m = m1 / den
                out[f"CoE_{k}"][b, sl] = np.where(ok.numpy(), m.numpy(), np.nan)
                out[f"width_{k}"][b, sl] = np.where(ok.numpy(), (m2 / den - m * m).clamp(min=0).sqrt().numpy(), np.nan)
    return out
