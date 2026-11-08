"""Markdown report from metrics.json (+ sweep.json if present). Prints only computed numbers."""
from __future__ import annotations

import json
import os

from .plots import LABEL


def _f(x, nd=4):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def _main_down(r):
    t = r.get("downstream_task", {})
    for k in ("roc_auc", "accuracy", "rel_resolution"):
        if k in t:
            return k, t[k]
    return None, None


def write_report(base: str, synthetic: bool = False) -> str:
    m = json.load(open(os.path.join(base, "eval", "metrics.json")))
    L = [f"# CaloNSF evaluation – {m['dataset']}", ""]
    if synthetic:
        L += ["> **Synthetic toy data.** These numbers only show that the pipeline runs end to end. "
              "They say nothing about physics performance.", ""]
    for f, R in m["factors"].items():
        L += [f"## LR factor ×{f}  ({R['n_events']} test events)", ""]
        dref = R.get("downstream_on_gt", {})
        if dref:
            L += ["Downstream models evaluated on the true HR showers (reference): " +
                  "; ".join(f"{k}: " + ", ".join(f"{a}={_f(b)}" for a, b in v.items() if not a.startswith("resolution_E"))
                            for k, v in dref.items()), ""]
        L += ["| method | MAE (log) | PSNR | SSIM | E response | key-obs W1/σ ↓ | FPD ↓ | classifier AUC (→0.5) | downstream | s/event |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in R["methods"].items():
            img = list(r["image"].values())
            avg = lambda k: sum(i[k] for i in img) / len(img)
            dk, dv = _main_down(r)
            L.append(f"| {LABEL.get(name, name)} | {_f(avg('MAE'))} | {_f(avg('PSNR'))} | {_f(avg('SSIM'))} | "
                     f"{_f(r['energy']['response_mean'])} ± {_f(r['energy']['response_std'], 2)} | "
                     f"{_f(r.get('key_obs_mean_w1_norm'))} | {_f(r.get('FPD'))} | {_f(r.get('classifier_auc'))} | "
                     f"{(dk + '=' + _f(dv)) if dk else '–'} | {_f(r.get('time_per_event_s'), 3)} |")
        L.append("")
        c = R["methods"].get("calonsf")
        if c:
            e = c["energy"]
            L += [f"CaloNSF energy conservation vs its (denoised) LR input: max rel. error "
                  f"{_f(e.get('conservation_err_max'))}, mean {_f(e.get('conservation_err_mean'))}.", ""]
            if "uncertainty" in c:
                L += ["Ensemble calibration (non-empty truth cells inside mean ± kσ; Gaussian ideal 0.68 / 0.95): " +
                      ", ".join(f"{k}={_f(v)}" for k, v in c["uncertainty"].items()), ""]
            bn =LABEL.get(R.get("best_baseline"), "baseline")
            L += ["### Physics observables (CaloNSF vs best baseline)", "",
                  "Rel. bias = (mean(pred) − mean(truth)) / |mean(truth)|, the shift of the distribution mean.", "",
                  f"| observable | CaloNSF W1/σ | CaloNSF KS | CaloNSF rel. bias | {bn} W1/σ | {bn} rel. bias |",
                  "|---|---|---|---|---|---|"]
            bb = R["methods"].get(R.get("best_baseline") or "", {}).get("observables", {})
            for k, v in c["observables"].items():
                L.append(f"| {k} | {_f(v['w1_norm'])} | {_f(v['ks'])} | {_f(v['rel_bias'])} | "
                         f"{_f(bb.get(k, {}).get('w1_norm'))} | {_f(bb.get(k, {}).get('rel_bias'))} |")
            L.append("")
            if c.get("observables_by_subset"):
                L += ["### Per class / particle / energy bin (CaloNSF, mean W1/σ over all observables)", "",
                      "| subset | mean W1/σ | downstream |", "|---|---|---|"]
                for s, d in c["observables_by_subset"].items():
                    if not d:
                        continue
                    w = sum(v["w1_norm"] for v in d.values()) / len(d)
                    ds = c.get("downstream_task_by_subset", {}).get(s, {})
                    L.append(f"| {s} | {_f(w)} | " + ", ".join(f"{a}={_f(b)}" for a, b in ds.items()
                                                             if not a.startswith("resolution_E")) + " |")
                L.append("")
        if "two_shower" in R:
            t = R["two_shower"]
            L += ["### Two-shower separation (layer 1)", "",
                  "Fraction of overlays with exactly 2 peaks, and fraction whose peak count matches the truth "
                  "(speckle inflates the first number but not the second).", "",
                  "| separation [cells] | Truth: 2 peaks | LR: 2 peaks | CaloNSF: 2 peaks | LR matches truth | CaloNSF matches truth |",
                  "|---|---|---|---|---|---|"]
            for i, d in enumerate(t["shift_cells"]):
                L.append(f"| {d} | {_f(t['gt'][i])} | {_f(t['lr'][i])} | {_f(t['sr'][i])} | "
                         f"{_f(t.get('lr_agree', [None] * 99)[i])} | {_f(t.get('sr_agree', [None] * 99)[i])} |")
            L.append("")
        L += [f"Figures: `eval/x{f}/examples.png`, `eval/x{f}/observables.png`" +
              (", `roc.png`" if m["dataset"] == "qg" else "") + (", `two_shower.png`" if "two_shower" in R else ""), ""]
    sp = os.path.join(base, "eval", "sweep.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        L += ["## Resolution sweep (above native resolution there is no truth to compare with)", "",
              "| output shape | s/event | memory MB | energy cons. err | pooled-to-native max diff | rel. drift of energy-weighted obs. vs native |",
              "|---|---|---|---|---|---|"]
        for row in s["rows"]:
            L.append(f"| {row['shape']} | {_f(row['time_s'], 3)} | {_f(row['memory_mb'], 4)} | {_f(row['energy_err'])} | "
                     f"{_f(row['pooled_consistency'])} | {_f(row.get('obs_drift'))} |")
        L.append("")
        L += ["Above-native outputs are physically plausible refinements, not measurements: they are checked "
              "only for energy conservation, consistency when pooled back to native resolution, and stability "
              "of physics observables.", ""]
    path = os.path.join(base, "eval", "report.md")
    open(path, "w").write("\n".join(L))
    return path
