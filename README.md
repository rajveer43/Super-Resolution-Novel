# CaloNSF — resolution-invariant, physics-aware super-resolution for calorimeter data

One model super-resolves **Quark–Gluon jet images**, **CaloChallenge Dataset 2** showers and **CaloGAN**
(e⁺ / γ / π⁺) showers at **any upsampling factor** (×2, ×4, ×8, and above native resolution,
e.g. QG 128 → 256 → 512 → 1024 → 2048). It conserves energy exactly and is evaluated on physics
observables and downstream tasks, not only PSNR/SSIM.

Runs on a Mac (Apple-silicon GPU via MPS), CUDA or CPU.

## The architecture

```
 LR cells (sum-pooled HR [+ noise] [+ DiffLense preprocessing])
        │  only non-empty cells become tokens; continuous coords (depth, u, v); Morton order
        ▼
 ┌────────────────────────────┐   pretrained with JEPA: predict EMA-encoder features of
 │ Cell encoder (transformer, │   (a) masked LR blocks and (b) HR cells at HR coordinates
 │ continuous axial RoPE)     │   from the visible LR context (cross-scale JEPA, new)
 └────────────┬───────────────┘
              │ cross-attention                       [optional] LR denoising head
              ▼
 for each scale step k = 1..K  (VAR next-scale ladder, ONE shared decoder):
   parents  = non-empty cells at scale k-1   (+ hidden state inherited from their own parent)
   decoder  = self-attn among parents (windowed) + cross-attn to LR tokens + AdaLN(cond)
   heads    = occupancy per child  +  flow-matching velocity on the child-fraction logits
   children = parent_E · softmax(sampled logits) · occupancy   ← children sum to the parent exactly
        ▼
 output at scale K (native = GT resolution; more steps = above-native refinement)
 ensemble of N samples → mean (prediction) + σ (per-cell uncertainty)
```

| Component | Source |
|---|---|
| Calorimeter cells as a sparse graph/set; upsampling by repetition as the condition; conditional flow matching (σ_min = 1e-4); ensemble averaging; denoising noisy LR | Graph Denoising (Kakati, Dreyer, Gross) |
| Coarse-to-fine **next-scale prediction**, one transformer shared over scales, scale *k* only sees coarser scales; multi-scale residual VQ (baseline) | VAR (Tian et al.) |
| JEPA mask-denoising in feature space with an EMA target; 3D-RoPE (here continuous); warmup-constant-decay schedule; progressive resolution; attentive probe | V-JEPA 2 (Assran et al.) |
| Conditional generative SR from a preprocessed LR input; median → Gaussian → NLM → threshold pipeline and its ablation; L1-on-noise DDPM (baseline) | DiffLense (Reddy et al.) |

**New in this work**
1. **Exact energy-conserving split.** Each non-empty parent cell predicts how its energy divides among its
   children. Energy is conserved at every scale by construction, so sum-pooling any output reproduces the
   coarser output exactly (tested).
2. **Flow matching on child-fraction logits** instead of raw energies, which removes the energy scale (1 GeV – 1 TeV in CaloChallenge).
3. **Resolution invariance.** Continuous RoPE + per-step factor/level features + one shared decoder, so the same
   weights handle ×2/×4/×8 ladders, heterogeneous CaloGAN layers (3×96, 12×12, 12×6), CaloChallenge's polar
   grid (periodic α) and above-native refinement. Any exact size (e.g. 300×300) is reached with exact
   area-overlap resampling.
4. **Cross-scale JEPA pretraining** and a **frozen-JEPA "physics perceptual" loss** (replaces LPIPS, which is meaningless on calorimeter data).
5. **Sparse tree generation.** Only children of non-empty parents are created, so cost scales with occupancy
   and 2048×2048×3 fits on a laptop.

## Quick start (laptop, synthetic data)

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q                       # 37 tests, ~20 s on CPU
bash scripts/run_synthetic_all.sh                   # all 3 datasets end to end on toy data
```

The synthetic data (`scripts/make_synthetic.py`) is written in the **exact file schemas** of the real
datasets, so the real readers get exercised. Its physics is only qualitative. Never report numbers from it.

## Real data

| Dataset | Put files at (or edit `configs/data/*.yaml`) | Notes |
|---|---|---|
| Quark–Gluon (CMS Open Data / ML4SCI) | `data/real/qg/quark-gluon.h5` with `X_jets`, `y`, `pt`, `m0` | Parquet: `scripts/convert_qg_parquet.py --out data/real/qg/quark-gluon.h5 *.parquet`. 125×125 is zero-padded to 128. |
| CaloChallenge DS2 (Zenodo record 6366271 — verify) | `data/real/calochallenge_v2/dataset_2_{1,2}.hdf5` | File 2 is the official evaluation set. Run `scripts/check_data.py` to verify the voxel flatten order (`data.flatten_order`), and set `data.r_edges` from the official binning XML for physical radii. |
| CaloGAN (Mendeley Data "Electromagnetic Calorimeter Shower Images") | `data/real/calogan/{eplus,gamma,piplus}.hdf5` | energy in GeV, deposits in MeV; asserted at load. |

```bash
.venv/bin/python scripts/check_data.py --config configs/experiments/calochallenge.yaml
bash scripts/run_all.sh configs/experiments/qg.yaml            # pretrain → SR → baselines → downstream → eval → sweep
bash scripts/run_all.sh configs/experiments/qg_noisy.yaml      # noisy LR + DiffLense preprocessing + denoise head
bash scripts/run_ablations.sh configs/experiments/qg.yaml
.venv/bin/python scripts/super_resolve.py --config configs/experiments/qg.yaml --out-size 2048 --n 4
```

Override any setting from the command line: `-o train.epochs=30 model.d_model=192 data.max_events=100000`.

Mac notes: `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically. Defaults are sized for an M-series laptop
(d_model 128, 4+4 layers, ~3.2 M trainable parameters). On a CUDA box, raise `model.d_model`, layers and `data.batch_size`.

## Pipeline stages and outputs (`runs/<exp_name>/`)

| Stage | Script | Output |
|---|---|---|
| JEPA pretraining | `pretrain_jepa.py` | `jepa/encoder.pt` |
| SR training (multi-factor ladders) | `train_sr.py` | `sr/best.pt`, `sr/sr_log.csv` |
| Baselines: nearest, bicubic, EDSR, RCAN, SwinIR, LIIF, DDPM (DiffLense-style), Graph-FM, VAR-VQ | `train_baseline.py --name all` | `baselines/<name>_x<f>/` |
| Downstream models, trained on **GT HR only** | `train_downstream.py` | `downstream/task`, `downstream/probe` |
| Evaluation | `evaluate.py` | `eval/metrics.json`, `eval/report.md`, figures |
| Resolution sweep (native → native·2^k) | `resolution_sweep.py` | `eval/sweep.json`, `eval/sweep/*.png` |

## Evaluation (each split by class / particle / incident-energy bin)
* **Image:** MAE / MSE / RMSE / PSNR / SSIM on log1p(E/E_s), relative raw-energy MAE, occupancy precision/recall, sparsity.
* **Energy:** response ΣSR/ΣGT; conservation error vs the model's LR input (0 for CaloNSF); per-layer closure.
* **Physics observables** (W1/σ, KS, per-event bias), FPD, KPD:
  QG: jet mass, girth, pT_D, core fraction, multiplicities, channel energies/widths, radial profile.
  CaloGAN: E₀/E₁/E₂, fractions, shower depth and width, lateral widths per layer, sparsity, E_ratio.
  CaloChallenge: E per layer, centre of energy and width in x/y per layer, sparsity, E_tot/E_inc, longitudinal and radial profiles.
* **Classifier test:** gradient-boosted classifier separating SR from GT (AUC → 0.5 is best).
* **Downstream:** quark/gluon ROC-AUC; CaloGAN particle ID and energy resolution; CaloChallenge energy resolution;
  JEPA attentive probe. All trained on GT and applied unchanged to SR.
* **Uncertainty:** coverage of GT within the ensemble mean ± kσ.
* **Two-shower separation (CaloGAN):** overlay two showers d cells apart, downsample, super-resolve, count peaks.

## Caveats
* **No ground truth above native resolution.** These datasets do not resolve anything finer than their
  native grids (QG 125×125, CaloChallenge 45×16×9, CaloGAN 3×96 / 12×12 / 12×6). 256 … 2048 outputs are
  physically plausible refinements, not measurements. They are checked only for energy conservation,
  exact consistency when pooled to native, and stability of observables. Granularity-dependent observables
  (pT_D, multiplicity, sparsity) change with resolution by definition.
* Multi-ladder training covers native-and-below scales. Steps above native extrapolate in the level features;
  `model.abs_level_feature=false` is the scale-free alternative.
* The two-shower test runs a model trained on single showers on overlays (a generalisation test).
* CaloChallenge radial bins default to uniform spacing until `data.r_edges` is set.

## Layout
```
calonsf/data      geometry & scale ladders, readers, synthetic data, degradation (noise, DiffLense), tokens, ops
calonsf/models    blocks (continuous RoPE, windowed attention, AdaLN), encoder, jepa, calonsf, baselines/
calonsf/physics   observables, stats (W1, KS, FPD, KPD, subsets)
calonsf/downstream  task models, attentive probe, classifier test, two-shower test
calonsf/eval      metrics, plots, evaluate, report
calonsf/train     trainer, common
configs/          base.yaml, data/*.yaml, experiments/*.yaml
scripts/          entry points + run_all.sh, run_synthetic_all.sh, run_ablations.sh
tests/            geometry, RoPE, energy conservation, physics, baselines, end-to-end
```
