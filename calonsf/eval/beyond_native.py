"""Consistency checks for super-resolution BEYOND the native 16 x 9 (alpha x r) grid of CaloChallenge Dataset 2.

There is no Geant4 ground truth above native resolution, so the CaloChallenge metrics cannot score these outputs.
What can be checked is whether the output is self-consistent. For an image size N x N (Cartesian x-y, see
`cartesian.py`) the shower is generated on the next-scale ladder up to the first polar grid whose cells are no
larger than one pixel radially (R = 9 * 2^k >= N / 2); one generation per shower yields every size, since each
ladder step is one size. For every size the following is measured, averaged over showers:

  energy conservation      |sum(output) - sum(LR input)| / sum(LR input), for the polar grid and the N x N image
  pool-back to LR          L1 relative difference of the output sum-pooled to the LR grid and the LR input
  pool-back to native      same, against the model's own native-resolution output
  hierarchical consistency worst pool-back of any ladder step to the previous step
  image position error     |centre of energy of the N x N image - that of the polar grid| in mm (pixel size given)
  feature drift            change of the per-layer centres of energy [mm] and widths [relative] with respect to
                           the previous ladder step and to the native output: granularity-independent
                           observables should converge, not drift, as the grid is refined
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from ..data.degrade import make_loader
from ..data.ops import sum_pool
from ..utils.misc import free_cache, get_logger, save_json, to_device
from .beyond_figures import plot_ladder, plot_zoom, render_polar_density
from .cartesian import PolarToCartesian, image_centre_of_energy
from .features import polar_moments

log = get_logger()
KEYS = ("CoE_eta", "CoE_phi", "CoE_r")
WIDTHS = ("width_eta", "width_phi", "width_r")


def extra_steps_for(size: int, n_r: int = 9) -> int:
    """Ladder steps above native so that the radial cell size is at most one pixel (R = n_r * 2^k >= size / 2)."""
    return max(0, math.ceil(math.log2(size / (2 * n_r)) - 1e-9))


def _l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().sum().double().cpu() / a.sum().double().cpu().clamp(min=1e-30))


def _drift(m_new: dict, m_ref: dict) -> dict:
    ok = np.isfinite(m_new["CoE_r"]) & np.isfinite(m_ref["CoE_r"]) & (m_ref["E"] > 0)
    out = {}
    for k in KEYS:
        out[f"{k}_shift_mm"] = float(np.abs(m_new[k] - m_ref[k])[ok].mean()) if ok.any() else float("nan")
    for k in WIDTHS:
        w = ok & (m_ref[k] > 0)
        out[f"{k}_rel_change"] = float(np.abs(m_new[k] / np.where(m_ref[k] > 0, m_ref[k], 1) - 1)[w].mean()) if w.any() else float("nan")
    return out


def run_beyond_native(cfg, geom, reader, splits, model, base, device, sizes=None, n_events=None, factor=None,
                      n_images=None) -> dict:
    if geom.name != "calochallenge":
        raise ValueError("beyond-native checks are implemented for CaloChallenge Dataset 2")
    bn = cfg.eval.beyond_native
    sizes = sorted(int(s) for s in (sizes or bn.sizes))
    n_events = int(n_events or bn.n_events)
    factor = int(factor or bn.get("factor") or cfg.eval.lr_factors[0])
    n_images = int(bn.get("images", 0) if n_images is None else n_images)
    img_dir = os.path.join(base, "beyond_native", "images")
    r_edges = geom.extras["r_edges"]
    nz, (na, nr) = geom.group("shower").n_layers, geom.group("shower").shape
    steps = {N: extra_steps_for(N, nr) for N in sizes}
    k_max = max(steps.values())
    log.info(f"[beyond] x{factor}, {n_events} showers, sizes {sizes} -> extra ladder steps {steps}")
    pix = {N: PolarToCartesian(na * 2 ** steps[N], nr * 2 ** steps[N], r_edges, N, int(bn.samples), device) for N in sizes}
    dl = make_loader(cfg, reader, splits["test"][:n_events], geom, False, factors=[factor], batch_size=1, shuffle=False)
    acc = {N: [] for N in sizes}
    for ev_i, batch in enumerate(dl):
        b = to_device(batch, device)
        o = model.generate(b, extra_steps=k_max, ensemble=1, nfe=int(cfg.eval.nfe), solver=cfg.eval.solver, return_scales=True)
        ni = o["ladder"].native_index
        sc = [s["shower"] for s in o["scales"]]
        lr = b["lr"]["shower"]
        native = sc[ni]
        cache, keep = {}, {}

        def moments(j):  # per-layer centres of energy / widths of ladder step ni + j, computed once
            if j not in cache:
                cache[j] = polar_moments(sc[ni + j], r_edges)
            return cache[j]

        for N in sizes:
            j = steps[N]
            G = sc[ni + j]
            r = {"polar_shape": [int(G.shape[-2]), int(G.shape[-1])]}
            tot = G.double().sum()
            r["conservation_polar"] = float(abs(tot - lr.double().sum()) / lr.double().sum())
            fu, fv = G.shape[-2] // lr.shape[-2], G.shape[-1] // lr.shape[-1]
            r["poolback_lr"] = _l1(lr, sum_pool(G, fu, fv))
            r["poolback_native"] = _l1(native, sum_pool(G, 2 ** j, 2 ** j)) if j > 0 else 0.0
            r["hierarchical"] = max([_l1(sc[ni + i - 1], sum_pool(sc[ni + i], 2, 2)) for i in range(1, j + 1)] or [0.0])
            img = pix[N](G.sum(1))
            r["conservation_image"] = float(abs(img.double().sum() - tot) / tot)
            mom_polar = polar_moments(G.sum(1, keepdim=True), r_edges)
            c_img = image_centre_of_energy(img, pix[N])
            r["image_position_error_mm"] = float(max(abs(c_img[0, 0] - mom_polar["CoE_eta"][0, 0]),
                                                     abs(c_img[0, 1] - mom_polar["CoE_phi"][0, 0])))
            r["pixel_mm"] = pix[N].pixel
            if j > 0:
                r.update({f"vs_previous_{k}": v for k, v in _drift(moments(j), moments(j - 1)).items()})
                r.update({f"vs_native_{k}": v for k, v in _drift(moments(j), moments(0)).items()})
            acc[N].append(r)
            if ev_i < n_images:
                keep[N] = img[0].float().cpu().numpy()
            del G, img
        if ev_i < n_images:
            os.makedirs(img_dir, exist_ok=True)
            _save_sample_images(img_dir, ev_i, batch, native, lr, keep, pix, r_edges, factor, float(cfg.eval.beyond_native.get("zoom_mm", 10.0)))
        free_cache(device)
    res = {"factor": factor, "n_events": n_events, "native_shape": [na, nr], "samples": int(bn.samples), "sizes": {}}
    for N in sizes:
        rows = acc[N]
        agg = {"extra_steps": steps[N], "polar_shape": rows[0]["polar_shape"], "pixel_mm": rows[0]["pixel_mm"]}
        for k in rows[0]:
            if k in ("polar_shape", "pixel_mm"):
                continue
            v = np.array([r[k] for r in rows], float)
            agg[k] = {"mean": float(np.nanmean(v)) if np.isfinite(v).any() else None,
                      "max": float(np.nanmax(v)) if np.isfinite(v).any() else None}
        res["sizes"][str(N)] = agg
    save_json(res, os.path.join(base, "beyond_native", "metrics.json"))
    return res


def _save_sample_images(img_dir, i, batch, native, lr, keep, pix, r_edges, factor, zoom_mm):
    half = next(iter(pix.values())).half
    ref = 512  # display grid for the native-resolution panels
    view = lambda g: render_polar_density(g.sum(1)[0].float().cpu().numpy(), r_edges, ref, half)
    sizes = sorted(keep)
    panels = [(f"LR input ($\\times${factor}): {lr.shape[-2]}$\\times${lr.shape[-1]} cells/layer", view(lr)),
              (f"Geant4 truth: {native.shape[-2]}$\\times${native.shape[-1]} cells/layer", view(batch["hr"]["shower"].to(lr.device))),
              (f"CaloNSF, native: {native.shape[-2]}$\\times${native.shape[-1]} cells/layer", view(native))]
    dens = {N: keep[N] / (pix[N].pixel ** 2) for N in sizes}
    panels += [(f"CaloNSF, {N} $\\times$ {N} image", dens[N]) for N in sizes]
    e = float(batch["e_inc"][0])
    info = f"CaloChallenge Dataset 2  |  test shower {i}  |  $E_{{\\mathrm{{inc}}}}$ = {e / 1000:.3g} GeV  |  LR factor $\\times${factor}"
    plot_ladder(os.path.join(img_dir, f"event{i}_ladder.png"), panels, half, info)
    nat = native.sum(1)[0].float().cpu().numpy()
    c = polar_moments(native.sum(1, keepdim=True), r_edges)
    centre = (float(c["CoE_eta"][0, 0]), float(c["CoE_phi"][0, 0]))
    plot_zoom(os.path.join(img_dir, f"event{i}_zoom.png"), {N: keep[N] for N in sizes}, {N: pix[N].pixel for N in sizes},
              half, centre, zoom_mm, info)


def write_beyond_report(base: str, res: dict, synthetic: bool = False) -> str:
    def f(d, k, p=3):
        v = d.get(k)
        return "n/a" if v is None or v["mean"] is None else f"{v['mean']:.{p}g}"

    def fm(d, k, p=3):
        v = d.get(k)
        return "n/a" if v is None or v["max"] is None else f"{v['max']:.{p}g}"

    L = ["# Super-resolution beyond native resolution: consistency checks", ""]
    if synthetic:
        L += ["> Synthetic toy data: a pipeline check only.", ""]
    na, nr = res["native_shape"]
    L += [f"CaloChallenge Dataset 2, LR factor x{res['factor']}, {res['n_events']} showers. Native grid per layer: "
          f"{na} (alpha) x {nr} (r). Image sizes are Cartesian x-y images (x = eta, y = phi) of the whole detector, "
          f"each polar layer resampled energy-conservingly ({res['samples']}x{res['samples']} points per cell). "
          "There is no Geant4 ground truth at these resolutions: the numbers below test self-consistency, not accuracy.", ""]
    head = ["image N x N", "ladder steps", "polar grid per layer", "pixel [mm]", "energy err. (polar)", "energy err. (image)",
            "pool-back to LR", "pool-back to native", "hierarchical (max)", "image position err. [mm]"]
    rows = []
    for N, d in res["sizes"].items():
        rows.append([f"{N} x {N}", str(d["extra_steps"]), f"{d['polar_shape'][0]} x {d['polar_shape'][1]}", f"{d['pixel_mm']:.3g}",
                     f(d, "conservation_polar"), f(d, "conservation_image"), f(d, "poolback_lr"), f(d, "poolback_native"),
                     fm(d, "hierarchical"), f(d, "image_position_error_mm")])
    L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]
    L += ["Relative errors (mean over showers); 0 is perfect, float32 rounding gives about 1e-7.", "",
          "## Feature drift of the granularity-independent observables", "",
          "Mean absolute change of the per-layer centre of energy [mm] and mean relative change of its width, "
          "against the previous ladder step (each step halves the cell size) and against the native-resolution output. "
          "The native voxel is 4.65 mm in r.", ""]
    head = ["image N x N", "dCoE eta prev [mm]", "dCoE phi prev [mm]", "dCoE r prev [mm]", "d width eta prev", "d width phi prev",
            "d width r prev", "dCoE eta native [mm]", "dCoE phi native [mm]", "dCoE r native [mm]"]
    rows = [[f"{N} x {N}", f(d, "vs_previous_CoE_eta_shift_mm"), f(d, "vs_previous_CoE_phi_shift_mm"), f(d, "vs_previous_CoE_r_shift_mm"),
             f(d, "vs_previous_width_eta_rel_change"), f(d, "vs_previous_width_phi_rel_change"), f(d, "vs_previous_width_r_rel_change"),
             f(d, "vs_native_CoE_eta_shift_mm"), f(d, "vs_native_CoE_phi_shift_mm"), f(d, "vs_native_CoE_r_shift_mm")]
            for N, d in res["sizes"].items()]
    L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]
    L += ["## Notes", "",
          "* Above native resolution every sub-cell of a non-empty parent receives energy (`model.above_native_occupancy: keep`), "
          "so the finer outputs are smooth refinements, not new hit patterns.",
          "* The polar grid is generated at the first ladder step whose radial cells are at most one pixel wide; the angular "
          "cells at the outer radius can still be several pixels wide for the largest circumferences.",
          "* Centre-of-energy drift contains a coordinate-discretisation part that shrinks as the grid is refined; "
          "widths of the centre of energy are computed from voxel centres at each resolution."]
    path = os.path.join(base, "beyond_native", "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path
