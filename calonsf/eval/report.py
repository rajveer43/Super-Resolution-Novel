"""report.md from eval/metrics.json: only numbers that were computed are shown."""
from __future__ import annotations

import json
import os

import numpy as np

from .features import LAYERED, SCALAR

NAME = {"calonsf": "CaloNSF", "nearest": "Nearest", "bicubic": "Bicubic"}


def _f(x, p=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}g}"


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def write_report(base: str, synthetic: bool = False) -> str:
    res = json.load(open(os.path.join(base, "eval", "metrics.json")))
    L = ["# CaloChallenge Dataset 2: CaloNSF evaluation", ""]
    if synthetic:
        L += ["> Synthetic toy data: the numbers are only a pipeline check, never report them.", ""]
    L += [f"Voxel activity threshold: {res['threshold_mev']} MeV. Reference = Geant4 HR showers of the test split; "
          "each method super-resolves their LR version. Lower is better except precision, recall, coverage "
          "(higher) and density (closer to 1); a classifier AUC of 0.5 means indistinguishable.", ""]
    for f, R in res["factors"].items():
        L += [f"## LR factor x{f}  ({R['n_events']} events, device {R.get('device', '?')})", ""]
        have = {k for m in R["methods"].values() for k in m["classifier"]}
        kinds = [k for k in ("dnn_low", "dnn_high", "cnn_low") if k in have]
        head = ["method", "mean sep. power", "FPD", "KPD", *[f"AUC {k}" for k in kinds], "precision", "recall",
                "density", "coverage", "PCC mean |Δ|"]
        g4 = R["geant4_vs_geant4"]
        mf = g4.get("manifold", {})
        band_mean = {k: float(np.nanmean([*v["min"], *v["max"]])) for k, v in g4["separation_band"].items()}
        rows = [["Geant4 vs Geant4", _f(float(np.mean(list(band_mean.values())))), _f(g4["fpd"]), _f(g4["kpd"]),
                 *["0.5" for _ in kinds], _f(mf.get("precision")), _f(mf.get("recall")), _f(mf.get("density")),
                 _f(mf.get("coverage")), "0"]]
        for m, r in R["methods"].items():
            c = r["classifier"]
            mm = r["manifold"]
            rows.append([NAME.get(m, m), _f(r["separation_power"]["summary"]["mean_all"]), _f(r["fpd"]),
                         f"{_f(r['kpd'])} ± {_f(r['kpd_std'], 2)}",
                         *[(f"{c[k]['auc_mean']:.3f} ± {c[k]['auc_std']:.3f}" if k in c else "n/a") for k in kinds],
                         _f(mm["precision"]), _f(mm["recall"]), _f(mm["density"]), _f(mm["coverage"]),
                         _f(r["pcc"]["mean_abs_diff"])])
        L += _table(head, rows) + [""]
        L += ["### Separation power per feature (mean over layers)", ""]
        feats = [*SCALAR, "voxel_energy", *LAYERED]
        rows = [["Geant4 vs Geant4 (mean of band)", *[_f(float(np.nanmean([*g4["separation_band"][k]["min"],
                                                                           *g4["separation_band"][k]["max"]])), 3) for k in feats]]]
        for m, r in R["methods"].items():
            s = r["separation_power"]["summary"]
            rows.append([NAME.get(m, m), *[_f(s.get(k), 3) for k in feats]])
        L += _table(["method", *feats], rows) + [""]
        L += ["### Generation time (seconds per shower) and parameters", ""]
        bss = sorted({b for t in R["timing_s_per_event"].values() for b in t}, key=int)
        rows = [[NAME.get(m, m), *[_f(t[b]["s_per_event"], 3) if b in t else "n/a" for b in bss]]
                for m, t in R["timing_s_per_event"].items()]
        L += _table(["method", *[f"batch {b}" for b in bss]], rows) + [""]
        if "parameters" in R:
            p = R["parameters"]
            L += [f"CaloNSF parameters: {p['trainable_in_training']:,} trainable in training, "
                  f"{p['needed_for_generation']:,} needed for generation.", ""]
        L += ["Figures: `x%s/separation_power.png`, `histograms.png`, `pcc.png`, `summary.png`." % f, ""]
    L += ["## Notes", "",
          "* Histogram bin edges are 50 equal-width bins over the reference range (log10 for layer and voxel "
          "energies), not the challenge's published reference binning; values outside are counted in the edge bins.",
          "* The Geant4 band is the min/max over random half-splits of the single Geant4 test set (the challenge "
          "used ten independent Geant4 datasets).",
          "* Precision / recall / density / coverage compare the Geant4 showers of one half of the events with the "
          "generated showers of the other half (disjoint events), because a super-resolved shower sits next to the "
          "Geant4 shower it was made from. The `Geant4 vs Geant4` row is the same two halves of the Geant4 set.",
          "* As the challenge does for every submission, voxels below the readout cutoff (0.5 keV / 0.033 = 15.15 keV, paper sec. 2.2) "
          "are set to 0 for all methods before the metrics; the dropped energy is tiny, so layer energies and E_dep/E_inc stay "
          "conserved to ~1e-6 relative and their separation power and PCC difference are ~0 for every energy-conserving method.",
          "* Classifiers are trained per method on the Geant4 vs generated showers with a 60/20/20 split; the "
          "multiclass (submission-vs-submission) test is not implemented.",
          "* Timing is for super-resolving a shower from its LR input on the evaluation device, with the model "
          "already loaded."]
    path = os.path.join(base, "eval", "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path
