# Phase 0 Track 1 — Environment Setup & Baseline Reproduction

**Phase:** 0 — Environment Setup & Baseline Reproduction
**Track:** 1 — Core Inference Architecture
**Status:** In Progress
**Weeks:** 1–2
**Branch:** `phase-0-track-1`

---

## Objective

Establish a verified development environment and reproduce the two baseline retrieval systems — Vasist 2023 (NPE) and Gebhard 2025 (FMPE) — to confirm the training pipeline is correct and establish ground-truth reference numbers. Nothing new is built in this phase. The output is a trusted foundation: if Phase 1's transformer encoder produces wrong posteriors, the fault is in the new code, not the environment underneath it.

**Why reproduce baselines before building anything:**
MIRAGE's primary evaluation metric is IS-efficiency ε. Gebhard 2025 achieved ε = 19.1% on simulated data. MIRAGE must surpass this on real JWST data. Without a verified reproduction of that 19.1% number, there is no ground truth to compare against.

---

## 1. Repository Structure

```
Project/
├── Documentation/
│   ├── problems_and_decisions.md     ← design decisions D1–D11
│   └── 00_phase0_track1_environment_baselines.md   ← this file
├── Paper/
│   ├── Gebhard 2025.pdf
│   └── Vasist 2023.pdf
├── data/                             ← gitignored, downloaded locally
│   └── gebhard/
│       ├── model__best_fmpe.pt       ← pretrained FMPE weights (1.2 GB)
│       ├── model__best_npe.pt        ← pretrained NPE weights (1.2 GB)
│       ├── fmpe.hdf                  ← reference posteriors, σ=0.125754 (1.6 GB)
│       ├── noise-free__sigma-0.125754__R-400__pRT-2.6.7.hdf   ← benchmark spectrum (13 KB)
│       └── test-default__R-400__seed-42.hdf   ← test set (4.4 MB)
├── configs/
│   └── fmpe_abc/
│       ├── config.yaml               ← fm4ar FMPE config for ABC
│       └── model__best.pt            ← best FMPE checkpoint (gitignored via data/)
├── checkpoints/
│   └── abc_npe_best.pt               ← best NPE checkpoint (gitignored)
├── figures/
│   ├── gebhard_fmpe_cornerplot.png   ← Phase 0 verification output (not for paper)
│   ├── gebhard_npe_cornerplot.png    ← NPE baseline verification (not for paper)
│   ├── abc_fmpe_planet2020.png       ← FMPE posterior on ABC Planet_2020
│   └── abc_npe_planet2020.png        ← NPE posterior on ABC Planet_2020
├── fm4ar/                            ← cloned: github.com/timothygebhard/fm4ar (gitignored)
├── sbi-ear/                          ← cloned: github.com/MalAstronomy/sbi-ear (gitignored)
├── scripts/
│   ├── plot_gebhard_cornerplot.py
│   ├── plot_gebhard_npe_cornerplot.py
│   ├── prepare_abc_hdf5.py           ← Level2Data → abc_{train,valid,test}.hdf
│   ├── train_npe_abc.py              ← standalone NPE training (lampe 0.9)
│   ├── infer_fmpe_abc.py             ← FMPE inference + corner plot on ABC
│   └── infer_npe_abc.py              ← NPE inference + corner plot on ABC
├── .gitignore
├── ACKNOWLEDGEMENTS.md
├── environment.yml
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 2. Environment

**Conda environment:** `mirage` (Python 3.11)
**Location:** `/Users/angshumansmac/anaconda3/envs/mirage/`

```bash
conda create -n mirage python=3.11 -y
conda activate mirage
python -m pip install -r requirements.txt
```

**Critical:** Always use `python -m pip`, never bare `pip`. The system `pip` points to Python 3.12 (`/Library/Frameworks/Python.framework/Versions/3.12/bin/pip`), so bare `pip install` installs into the wrong Python entirely. This caused torch to silently install into system Python while the conda env remained empty — only caught when `import torch` failed despite "already installed" messages.

**Key packages installed:**

| Package | Version | Purpose |
|---|---|---|
| torch | 2.2.2 | Neural network training and inference |
| fm4ar | 0.0.1 | Gebhard 2025 FMPE backbone (pip-installable) |
| lampe | latest | Normalising flows library (sbi-ear dependency) |
| zuko | latest | Flow transforms (sbi-ear dependency) |
| numpy | 1.26.4 | Array operations |
| astropy | 7.2.0 | FITS file handling for JWST data |
| astroquery | 0.4.11 | MAST archive queries |
| wandb | 0.26.1 | Experiment tracking |
| corner | 2.2.2 | Posterior corner plots |
| tqdm | 4.66.4 | Progress bars |

**GPU:** `torch.cuda.is_available() = False` — Mac CPU only. Training runs go on university HPC cluster. Phase 0 runs fine on CPU.

---

## 3. Baseline Codebases

### fm4ar (Gebhard 2025)

`github.com/timothygebhard/fm4ar` — pip-installable as `fm4ar==0.0.1`.

This is the direct predecessor to MIRAGE's inference engine. MIRAGE's flow-matching posterior estimator builds on top of this codebase. Getting it running correctly in Phase 0 means Phase 1 (adding the transformer encoder) starts from a verified foundation.

**Key design (from reading the source):**
- Continuous normalising flow (CNF) trained via flow-matching
- Conditioned on observed spectrum `x` and noise level `σ`
- 16 atmospheric parameters, petitRADTRANS forward model, 379 spectral bins at R=400
- Training: 2²⁵ = 33.5M spectra, noise hyper-prior σ ~ U(0.05, 0.50)
- 310M parameters, ~55 hours to train on H100

**What fm4ar is NOT:** A drop-in that runs on ABC data. It expects petitRADTRANS-format spectra and its own dataset structure. Adapting it to ABC format (TauREx3) is the main work of this phase — see Section 6.

### sbi-ear (Vasist 2023)

`github.com/MalAstronomy/sbi-ear` — scripts only, no `setup.py`. Cannot be `pip install`-ed.

**Architecture:**
- Embedding network: ResidualMLP (10 residual blocks)
- Flow: masked neural autoregressive flow, 3 transforms
- Training: 12M spectra, petitRADTRANS, 16 parameters, fixed noise σ = 0.1257 × 10⁻¹⁷

**Dependencies installed separately:**
```bash
python -m pip install lampe zuko dawgz
```
`petitradtrans` and `multinest` were intentionally skipped — both require complex native builds (Fortran compiler, C++ libraries) and are only needed for data generation, not inference. Pre-generated data is used instead.

---

## 4. Gebhard 2025 Dataset

**Source:** `doi.org/10.17617/3.LYSSVN` (Edmond — Max Planck data repository)

The full dataset is 66 files and too large to download entirely. Five specific files were selected:

| File | Size | Purpose |
|---|---|---|
| `model__best_fmpe.pt` | 1.2 GB | Pretrained FMPE model weights |
| `model__best_npe.pt` | 1.2 GB | Pretrained NPE model weights |
| `fmpe.hdf` (σ=0.125754) | 1.6 GB | Pre-computed FMPE posteriors — reference to compare against |
| `noise-free__sigma-0.125754__R-400__pRT-2.6.7.hdf` | 13 KB | The benchmark spectrum (ground truth θ₀) |
| `test-default__R-400__seed-42.hdf` | 4.4 MB | Test set for large-scale evaluation |

**Why σ=0.125754:** This is the noise level used by Vasist 2023 (σ = 0.1257 × 10⁻¹⁷ W m⁻² μm⁻¹), making it the direct comparison point between both papers. There are 6 `fmpe.hdf` files at different σ — only the σ=0.125754 one was downloaded.

**Why not the training data:** Training on 33.5M spectra is not needed for Phase 0. The pretrained weights + pre-computed posteriors are sufficient to verify the pipeline and establish ground-truth numbers.

**Structure of `fmpe.hdf`:**
```
samples          (1048576, 16)  — 1M posterior samples, 16 parameters
weights          (1048576,)     — IS importance weights
n_eff            scalar         — effective sample size
sampling_efficiency  scalar     — ε = n_eff / n_samples
log_evidence     scalar         — Bayesian evidence estimate
flux             (1048576, 379) — simulated spectra for IS verification
log_likelihoods  (1048576,)
log_prior_values (1048576,)
```

---

## 5. FMPE Baseline Verification — `scripts/plot_gebhard_cornerplot.py`

The verification script loads `fmpe.hdf` directly and produces a corner plot of 6 selected parameters (matching Fig. 3 in Gebhard 2025). It does not rerun inference — the posterior samples are already computed.

**Method:** IS-weighted resampling. The 1M samples are resampled with replacement (N=10,000) using normalised importance weights, giving an unweighted sample set that approximates the true posterior. `corner.corner()` plots these.

**Result:**

| Metric | This run | Gebhard 2025 Table 1 |
|---|---|---|
| ESS | 199,762 | 199,761.8 ✅ |
| ε | 19.05% | 19.1% ✅ |

Numbers match the paper to within rounding. The posterior shapes match Fig. 3 — ground truth values (black lines) sit correctly within the posterior contours for all 6 parameters. `log_P_quench` and `T_2` show the same characteristic flat/uninformative shapes as in the paper.

**Output:** `figures/gebhard_fmpe_cornerplot.png`

**IS-efficiency formula** (equations 9–11 in Gebhard 2025):

```
weights_i ∝ p(x | θ_i) · p(θ_i) / q(θ_i | x)
N_eff = (Σ w_i)² / Σ w_i²
ε = N_eff / N
```

This is the primary evaluation metric for the entire project. MIRAGE must achieve ε > 1% (ESS > 500) on real JWST observations.

---

## 6. Remaining Steps

### Step 6 — sbi-ear (NPE) corner plot
Load `npe.hdf` (same folder, σ=0.125754) and produce an equivalent corner plot. Expected: ε ≈ 11.6%, ESS ≈ 121,856 (Gebhard 2025 Table 1). This establishes the NPE baseline number.

### Step 7 — Download ABC database ✅

Changeat & Yip 2023, `zenodo.org/records/6770103`. Downloaded all files into `data/abc/`:

| File/Folder | Size | Contents |
|---|---|---|
| `Level1Data/` | 3.8 GB | 105,887 synthetic spectra + parameters (training set) |
| `Level2Data/` | 3.2 GB | 91,392 nested-sampling posteriors (ground truth) |
| `NeurIPS taurex tutorial/` | 67.6 MB | Tutorial code |
| `Tutorial - How To Use.ipynb` | 7 kB | Format walkthrough |

### Step 8 — ABC Format ✅

**Level1Data — 105,887 planets:**

`observations.hdf5` — keyed by `Planet_0`, `Planet_1`, ..., `Planet_105886`. Each planet has 4 arrays:

```
instrument_spectrum  (52,)  float64  — transmission spectrum (Rp/Rs)²
instrument_noise     (52,)  float64  — per-wavelength 1σ noise
instrument_wlgrid    (52,)  float64  — wavelength grid, μm, range ~0.55–7.3
instrument_width     (52,)  float64  — bin widths
```

`all_target.hdf5` — same keys. Each planet has nested sampling posteriors:
```
tracedata  (N_samples, 6)  — posterior samples (N varies per planet, ~2000–3000)
weights    (N_samples,)    — posterior weights
```
Planet attrs contain parameter quartiles but not the parameter names directly.

**6 atmospheric parameters (TauREx3):**

| # | Parameter | Description |
|---|---|---|
| 0 | T | Temperature (K) |
| 1 | log_H2O | Water vapour log abundance |
| 2 | log_CO2 | CO₂ log abundance |
| 3 | log_CH4 | Methane log abundance |
| 4 | log_CO | CO log abundance |
| 5 | log_NH3 | Ammonia log abundance |

**Level2Data — 91,392 planets with validated posteriors:**

`Ground Truth Package/Tracedata.hdf5` — same tracedata/weights structure. Planet attrs confirm:
```
target_order: ['T', 'log_H2O', 'log_CO2', 'log_CH4', 'log_CO', 'log_NH3']
```
`FM_Parameter_Table.csv` — ground truth θ₀ values for each planet (columns: planet_ID, planet_temp, log_H2O, log_CO2, log_CH4, log_CO, log_NH3).

`AuxillaryTable.csv` — stellar/planetary physical parameters (mass, radius, gravity, etc.) — not needed for Phase 0.

**Critical difference from Gebhard/Vasist:**

| Property | ABC (TauREx3) | Gebhard/Vasist (petitRADTRANS) |
|---|---|---|
| Parameters | 6 | 16 |
| Spectral bins | 52 | 379 |
| Wavelength | 0.55–7.3 μm | 0.6–5.0 μm |
| Forward model | TauREx3 | petitRADTRANS |

The adapter (Step 9) must account for all four differences.

### Step 9 — Data loader adapter ✅

**Why Level2Data, not Level1Data:** Level2Data is the only source with matched (spectrum, θ) pairs. `FM_Parameter_Table.csv` has the true forward-model parameters; `SpectralData.hdf5` has the corresponding spectra. Level1Data has posteriors (`all_target.hdf5`) but no clean point-estimate θ for training.

**`scripts/prepare_abc_hdf5.py`** — one-time preprocessing script. Merges `FM_Parameter_Table.csv` + `SpectralData.hdf5`, filters NaN rows, shuffles with seed 42, splits 80/10/10, saves:

```
data/abc/abc_train.hdf   ~73,000 planets
data/abc/abc_valid.hdf   ~9,100 planets
data/abc/abc_test.hdf    ~9,100 planets
```

Each file has keys: `theta (N,6)`, `flux (N,52)`, `wlen (1,52)`, `noise (N,52)`, `planet_id (N,)`. The `planet_id` column (from `FM_Parameter_Table.csv`) is carried through so inference scripts can look up the corresponding nested-sampling posterior in `Tracedata.hdf5`.

**Theta normalisation — `fm4ar/fm4ar/datasets/theta_scalers.py`:** Added an `"abc"` case to `get_mean_and_std()` with statistics computed from `abc_train.hdf` (73,113 planets):

```python
elif dataset == "abc":
    mean = np.array([1201.1842, -5.9989, -6.5019, -5.9979, -4.4954, -6.4910])
    std  = np.array([ 681.0441,  1.7337,  1.4457,  1.7390,  0.8639,  1.4373])
```

Without this, T (range ~100–5000 K) and the log abundances (range ~–9 to –3) are on completely different scales and the FMPE objective diverges (loss ~243,000–888,000 with noise). With normalisation applied via `MeanStdScaler` in the config, loss drops to ~1.9. The config sets:
```yaml
theta_scaler:
  method: "MeanStdScaler"
  kwargs:
    dataset: "abc"
```

**`configs/fmpe_abc/config.yaml`** — fm4ar training config for ABC. fm4ar infers `dim_theta=6` and `dim_context=52` automatically from the data (nothing hardcoded). Context embedding: `Concatenate([wlen, flux])` → (104,) → DenseResidualNet → output_dim=4096. No `AddNoise` transform — ABC spectra already include instrument noise. `n_train_samples=65000`, `n_valid_samples=8113` (must sum exactly to 73,113 — fm4ar's `random_split` enforces this).

**`scripts/train_npe_abc.py`** — standalone NPE training adapted from sbi-ear for ABC dimensions. sbi-ear's `train.py` hardcodes 379 bins and 16 params everywhere, so this is a clean rewrite with `DIM_X=52`, `DIM_THETA=6`, updated for lampe 0.9 API (changed from 0.6.1). Architecture: MLP embedding (52→64) + lampe NPE (6 params, 64 context, 2 MAF transforms). Saves best checkpoint to `checkpoints/abc_npe_best.pt`.

**Run order:**
```bash
# 1. Prepare HDF5 files (once)
python scripts/prepare_abc_hdf5.py

# 2. FMPE smoke test — uses fm4ar's full training infrastructure
python fm4ar/scripts/training/train_local.py --experiment-dir configs/fmpe_abc

# 3. NPE smoke test
python scripts/train_npe_abc.py
```

**Smoke test results (CPU, 1000 samples, 5 epochs):**
- FMPE: ran clean, loss noisy (143k–888k). Root cause: no theta normalisation. Fixed by adding `MeanStdScaler` to config (see above). Loss dropped to ~1.9 after fix.
- NPE: ran clean, loss 10048 → 16.7 (converged). Checkpoint saved at `checkpoints/abc_npe_best.pt`.
- Both pipelines verified end-to-end.

### Step 10 — Full CPU training (512 epochs, 73k samples)

Both models trained to completion on Mac CPU. Timing was much faster than expected because the problem is small: 6 params × 52 bins vs Gebhard's 16 params × 379 bins.

**NPE** — ~10 min on CPU (512 epochs × ~1.2s/epoch):
```bash
python scripts/train_npe_abc.py
# Checkpoint: checkpoints/abc_npe_best.pt
```
Loss: 16.7 → 15.1 over 512 epochs. Slowing by epoch ~230; model likely near its capacity limit for this architecture (2 MAF transforms, 64-dim embedding). Two gradient spikes recovered immediately (epoch 111: 89.4 → 16.2; epoch 186: 20257 → 15.8).

**FMPE** (fm4ar training system) — longer due to larger context embedding (DenseResidualNet output_dim=4096):
```bash
python fm4ar/scripts/training/train_local.py --experiment-dir configs/fmpe_abc
# Checkpoint: configs/fmpe_abc/model__best.pt
```

**Why posteriors are still broad:** Not CPU speed — the NPE architecture (2 transforms, 64-dim context) is undersized for capturing tight posteriors. The Gebhard NPE used 10 residual blocks + 3 MAF transforms. Cluster training with a larger architecture would improve this; raw speed was not the bottleneck here.

### Step 11 — Inference scripts

**`scripts/infer_fmpe_abc.py`** — loads `configs/fmpe_abc/model__best.pt`, replicates the test spectrum N_SAMPLES=10,000 times in the batch dimension (fm4ar's `sample_batch` uses context batch size as num_samples, not the kwarg), runs ODE integration, overlays ground-truth nested sampling posterior from `Tracedata.hdf5`. Output: `figures/abc_fmpe_planet{ID}.png`.

```bash
python scripts/infer_fmpe_abc.py --planet-idx 0   # Planet_2020
```

**`scripts/infer_npe_abc.py`** — loads `checkpoints/abc_npe_best.pt`, calls `model.flow(x).sample((10000,))`, same corner plot format. Output: `figures/abc_npe_planet{ID}.png`.

```bash
python scripts/infer_npe_abc.py --planet-idx 0
```

**Key gotcha — `sample_batch` ignores `num_samples` kwarg:** When a context dict is provided, fm4ar uses `context["flux"].shape[0]` as the sample count, not the kwarg. Fix: replicate the spectrum tensor:
```python
context = {
    "flux": flux.unsqueeze(0).expand(N_SAMPLES, -1).to(device),
    "wlen": wlen.unsqueeze(0).expand(N_SAMPLES, -1).to(device),
}
```

**CPU training results — Planet_2020 (index 0 from abc_test.hdf):**
- FMPE: posteriors physically sensible, broad, centred near training prior means. T posterior ~1200–1800 K range (true=1321 K). Expected for underconverged CPU run.
- NPE: T posterior peak ~1538 K (true=1321 K), log abundances beginning to constrain. Broad vs ground-truth nested sampling.
- Output figures: `figures/abc_fmpe_planet2020.png`, `figures/abc_npe_planet2020.png`

These are CPU smoke-test quality only. Cluster training is required for publishable posteriors and IS-efficiency numbers.

### Step 12 — Compute IS-efficiency on ABC
These numbers — FMPE and NPE IS-efficiency on ABC — are the synthetic benchmark baseline for the entire paper. Every MIRAGE result in Phases 1–4 is compared against these.

---

## Phase 0 Track 1 Completion Checklist

- [x] Conda environment `mirage` created (Python 3.11)
- [x] All dependencies installed via `python -m pip`
- [x] `environment.yml` exported and committed
- [x] fm4ar cloned and installed (`fm4ar==0.0.1`)
- [x] sbi-ear cloned; lampe, zuko, dawgz installed
- [x] Gebhard dataset downloaded (5 files, ~4 GB total)
- [x] FMPE corner plot produced — ε = 19.05%, ESS = 199,762 ✅ matches paper
- [x] NPE corner plot produced — ε = 11.62%, ESS = 121,856 ✅ matches paper
- [x] ABC database downloaded — zenodo.org/records/6770103, Level1Data + Level2Data in data/abc/
- [x] ABC format understood — 52 bins, 6 params (T + 5 molecules), TauREx3, 105,887 spectra
- [x] Data loader adapter written — prepare_abc_hdf5.py + configs/fmpe_abc/config.yaml + train_npe_abc.py
- [x] fm4ar smoke test on ABC — pipeline verified; theta normalisation bug found and fixed (MeanStdScaler + abc case in theta_scalers.py)
- [x] sbi-ear smoke test on ABC — pipeline verified, loss converged ~16.7 in 5 epochs CPU
- [x] Full 512-epoch CPU training — NPE ~10 min (6 params × 52 bins = small problem); FMPE longer due to larger context net
- [x] fm4ar posteriors on ABC — `infer_fmpe_abc.py` run on Planet_2020; `figures/abc_fmpe_planet2020.png`
- [x] sbi-ear posteriors on ABC — `infer_npe_abc.py` run on Planet_2020; `figures/abc_npe_planet2020.png`
- [ ] Cluster training — fm4ar FMPE + NPE on full 73k samples, 512 epochs, GPU (for publishable results)
- [ ] IS-efficiency on ABC — ε for FMPE and NPE (requires TauREx3 likelihood or alternative approach)
- [ ] All scripts and figures committed to `phase-0-track-1` branch

→ See `01_track1_transformer_encoder.md` (to be written at Phase 1 start)
