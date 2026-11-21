# CaloNSF — resolution-invariant, physics-aware super-resolution for CaloChallenge Dataset 2

One model super-resolves **CaloChallenge Dataset 2** showers (45 layers × 16 α × 9 r voxels) at any upsampling
factor (×2, ×4, and above native resolution). It conserves energy exactly and is evaluated with the
**CaloChallenge metrics** against the Geant4 test set (separation power of high-level features, layer-energy
correlations, classifier AUCs, FPD/KPD, precision/recall/density/coverage, timing and parameter counts).

Runs on a Mac (Apple-silicon GPU via MPS), CUDA or CPU.

## The architecture

```
 LR cells (sum-pooled HR)
        │  only non-empty cells become tokens; continuous coords (depth, α, r); Morton order
        ▼
 ┌────────────────────────────┐   pretrained with JEPA: predict EMA-encoder features of
 │ Cell encoder (transformer, │   (a) masked LR blocks and (b) HR cells at HR coordinates
 │ continuous axial RoPE)     │   from the visible LR context (cross-scale JEPA, new)
 └────────────┬───────────────┘
              │ cross-attention
              ▼
 for each scale step k = 1..K  (VAR next-scale ladder, ONE shared decoder):
   parents  = non-empty cells at scale k-1   (+ hidden state inherited from their own parent)
   decoder  = self-attn among parents (windowed) + cross-attn to LR tokens + AdaLN(cond)
   heads    = occupancy per child  +  flow-matching velocity on the child-fraction logits
   children = parent_E · softmax(sampled logits) · occupancy   ← children sum to the parent exactly
        ▼
 output at scale K (native = GT resolution; more steps = above-native refinement)
```

| Component | Source |
|---|---|
| Calorimeter cells as a sparse graph/set; upsampling by repetition as the condition; conditional flow matching (σ_min = 1e-4); ensemble averaging | Graph Denoising (Kakati, Dreyer, Gross) |
| Coarse-to-fine **next-scale prediction**, one transformer shared over scales, scale *k* only sees coarser scales | VAR (Tian et al.) |
| JEPA mask-denoising in feature space with an EMA target; 3D-RoPE (here continuous); warmup-constant-decay schedule; progressive resolution | V-JEPA 2 (Assran et al.) |

**New in this work**
1. **Exact energy-conserving split.** Each non-empty parent cell predicts how its energy divides among its
   children. Energy is conserved at every scale by construction, so sum-pooling any output reproduces the
   coarser output exactly (tested).
2. **Flow matching on child-fraction logits** instead of raw energies, which removes the energy scale (1 GeV – 1 TeV).
3. **Resolution invariance.** Continuous RoPE + per-step factor/level features + one shared decoder, so the same
   weights handle ×2/×4 ladders, the polar grid (periodic α) and above-native refinement. Any exact size is
   reached with exact area-overlap resampling.
4. **Cross-scale JEPA pretraining** and a **frozen-JEPA "physics perceptual" loss** (replaces LPIPS, which is meaningless on calorimeter data).
5. **Sparse tree generation.** Only children of non-empty parents are created, so cost scales with occupancy.

## Quick start (laptop, synthetic data)

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q                       # CPU, includes an end-to-end run with the evaluation
bash scripts/run_synthetic_all.sh                   # pretrain → SR → evaluation on toy data
```

The synthetic data (`scripts/make_synthetic.py`) is written in the **exact file schema** of the real
dataset, so the real reader gets exercised. Its physics is only qualitative. Never report numbers from it.

## Real data

CaloChallenge Dataset 2 (Zenodo record 6366271): `data/real/calochallenge_v2/dataset_2_{1,2}.hdf5`.
File 1 is used for training and validation, **file 2 is the official evaluation set** (the Geant4 reference).

```bash
.venv/bin/python scripts/download_data.py                     # resumable, checksums verified (~2.7 GB)
.venv/bin/python scripts/check_data.py --config configs/experiments/calochallenge.yaml   # verifies the voxel flatten order
bash scripts/run_all.sh configs/experiments/calochallenge.yaml  # pretrain → SR → evaluation
bash scripts/run_ablations.sh configs/experiments/calochallenge.yaml
.venv/bin/python scripts/super_resolve.py --config configs/experiments/calochallenge.yaml --out-size 64 --n 4
```

Override any setting from the command line: `-o train.epochs=30 model.d_model=192 data.max_events=100000`.
The radial bin edges (`data.r_edges`) are the official ones from `binning_dataset_2.xml`.

Mac notes: `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically. Defaults are sized for an M-series laptop
(d_model 128, 4+4 layers, ~3.2 M trainable parameters). On a CUDA box, raise `model.d_model`, layers and `data.batch_size`.

NERSC Perlmutter: see `slurm/` (`setup_env.sh` once, `download_data.sh`, then `submit_all.sh --smoke` and `submit_all.sh`;
settings in `slurm/settings.sh`).

## Pipeline stages and outputs (`runs/<exp_name>/`)

| Stage | Script | Output |
|---|---|---|
| JEPA pretraining | `pretrain_jepa.py` | `jepa/encoder.pt` |
| SR training (multi-factor ladders) | `train_sr.py` | `sr/best.pt`, `sr/sr_log.csv` |
| CaloChallenge evaluation | `evaluate.py` | `eval/metrics.json`, `eval/report.md`, `eval/x<f>/*.png` |
| Beyond-native consistency checks | `beyond_native.py` | `beyond_native/metrics.json`, `beyond_native/report.md` |

## Training recipe (what the first Perlmutter run taught us)

* **JEPA** (`pretrain_jepa.py`): the d_model-128 run was not collapsed (loss 3-5 % of a predict-the-mean baseline) but
  under-trained: a 5e-4 peak learning rate was unstable right after warmup, validation sat on a plateau through the
  constant-LR phase, and most of the gain came in the final LR decay while the loss was still falling steeply. The
  defaults are now `jepa.lr` 2e-4, `jepa.warmup_frac` 0.05 (at least one epoch), `jepa.decay_frac` 0.4 and more epochs.
* **Validation** now cycles through all `data.lr_factors` (it used to use only the first), so validation and training
  losses are comparable.
* **SR** plateaued with train ≈ validation loss, i.e. it was capacity-limited, and Optuna found no hyperparameter that
  mattered (all 14 trials within 3 %). `configs/experiments/calochallenge_large.yaml` is the A100 preset: d_model 256,
  6 + 6 layers (19.8 M parameters instead of 3.3 M), a larger encoder learning-rate multiplier (0.3).
* **Checkpoint selection**: set `train.select_events` (the large preset uses 300) and `best.pt` is chosen by the mean
  separation power of the high-level features on that many validation showers (`val_metric` in `sr_log.csv`; lower is
  better) instead of the flow-matching loss, which has a floor and does not measure what the challenge measures.
* Check first that JEPA pretraining helps at all: `scripts/run_ablations.sh` has the `no_jepa` run.

## Evaluation (`scripts/evaluate.py`, settings in `eval:` of `configs/base.yaml`)

For each LR factor, the HR showers of the test split are the Geant4 reference and each method's super-resolution
of their LR version is the submission. Methods: **CaloNSF** and the non-trainable **nearest** (upsampling by
repetition) and **bicubic** upsamplers. The metrics follow the CaloChallenge paper (section 8):

* **High-level features and separation power** (8.1): voxel energy, layer energies E_i, E_dep/E_inc, centres of
  energy and widths in η, φ, r per layer, and sparsity (1 − fraction of voxels above `eval.threshold_mev`).
  The separation power S(h₁,h₂) = ½ Σ (h₁ − h₂)² / (h₁ + h₂) is reported per feature and layer, with a grey
  band from random Geant4 half-splits.
* **Correlations** (8.2): Pearson correlation matrix of the 45 layer energies and its difference to Geant4.
* **Classifier tests** (8.3): binary Geant4-vs-generated classifiers — DNN on low-level features, DNN on
  high-level features and a 3D ResNet-18, 10 seeds, 60/20/20 split, isotonic calibration, AUC on the evaluation split.
* **FPD / KPD** (8.4) on the standardised high-level features.
* **Precision, recall, density, coverage** (8.5), k = 5, on log10 voxel energies, between the Geant4 showers of
  one half of the events and the generated showers of the other half (disjoint events, since a super-resolved
  shower is paired with the Geant4 shower it came from).
* **Generation time** per shower for several batch sizes (8.6) and **parameter counts** (8.7).

The multiclass (submission-vs-submission) classifier is not implemented.

Figures (`eval/x<f>/`) follow publication conventions: serif font with maths, axis labels with symbols and
units (energies in MeV, positions in mm), panel letters, a legend naming every series, a header with the
dataset / LR factor / number of showers, and a footnote explaining bands and error bars. Each is written as a
300 dpi PNG and a vector PDF.

## Beyond native resolution (`scripts/beyond_native.py`)

The native grid is 16 (α) × 9 (r) per layer. For image sizes **256², 512², 1024², 2048², …** (set
`eval.beyond_native.sizes`, or `--sizes 256 512 1024 2048 4096`) a shower is generated on the next-scale ladder
(each extra step halves the cell size in α and r) up to the first polar grid whose radial cells are at most one
pixel wide (R = 9·2ᵏ ≥ N/2: 256² → k = 4, 2048² → k = 7), then each layer is resampled energy-conservingly onto an N × N
Cartesian x–y grid (x = η = r cos α, y = φ = r sin α). One generation per shower gives every size.
There is no Geant4 ground truth at these resolutions, so the report tests **self-consistency**, not accuracy:

* energy conservation of the polar output and of the N × N image,
* exact pool-back of the output to the LR input and to the native-resolution output, and of every ladder step to the
  previous one,
* position error of the N × N image (its centre of energy against the polar grid's, in mm, next to the pixel size),
* drift of the granularity-independent observables (centres of energy in mm, widths relative) from one ladder step to
  the next and against the native output: they should converge as the grid is refined.

Memory and time grow about ×3 per doubling of N (on very sparse synthetic showers: 1024² ≈ 4 s and 2 GB, 2048² ≈ 11 s
and 6 GB per shower). Real showers are denser, and every sub-cell of a non-empty parent is kept above native
resolution, so budget memory before going to 4096² or beyond (lower `eval.beyond_native.n_events`).

## Caveats
* **Energy fidelity is exact by construction.** CaloNSF and the baselines only split LR cells in α and r and
  conserve every cell's energy, so layer energies, E_dep/E_inc and their correlations match Geant4 exactly
  (separation power and PCC difference ≈ 0). At ×2 the r axis is not upsampled at all (the largest divisor of 9
  that is ≤ 2 is 1), so the r features also match exactly. The metrics that discriminate are the centres of
  energy / widths in η and φ, sparsity, voxel energy, the classifiers, FPD/KPD and the manifold metrics.
* **No ground truth above native resolution.** CaloChallenge Dataset 2 does not resolve anything finer than its
  native 45×16×9 grid, so above-native outputs are plausible refinements, not measurements. They are only checked
  for energy conservation and exact consistency when pooled to native.
* Histogram bin edges are derived from the Geant4 reference (50 bins), not the challenge's published binning, and
  the Geant4 band comes from half-splits of the single test set. Every method's output gets the readout
  cutoff of the challenge (paper sec. 2.2: Par04 stores nothing below 0.5 keV = 15.15 keV after the 1/0.033 scaling,
  `eval.threshold_mev`) before any metric, so generated showers have exact zeros like Geant4.
* Multi-ladder training covers native-and-below scales. Steps above native extrapolate in the level features;
  `model.abs_level_feature=false` is the scale-free alternative.

## Layout
```
calonsf/data      geometry & scale ladders, CaloChallenge reader, synthetic data, LR generation, tokens, ops
calonsf/models    blocks (continuous RoPE, windowed attention, AdaLN), encoder, jepa, calonsf, baselines/ (nearest, bicubic)
calonsf/eval      features, separation, correlations, classifier, distances, manifold, resources, plots, report, run,
                  cartesian + beyond_native (polar-to-image resampling and the above-native checks)
calonsf/train     trainer, common
calonsf/utils     config, misc, viz (PNG output for super_resolve.py)
configs/          base.yaml, data/calochallenge.yaml, experiments/calochallenge{,_synthetic}.yaml
scripts/          entry points + run_all.sh, run_synthetic_all.sh, run_ablations.sh
tests/            geometry, RoPE, energy conservation, baselines, metrics, beyond-native, end-to-end
```
