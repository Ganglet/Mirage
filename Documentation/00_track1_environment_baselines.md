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
├── figures/
│   └── gebhard_fmpe_cornerplot.png   ← Phase 0 verification output
├── fm4ar/                            ← cloned: github.com/timothygebhard/fm4ar
├── sbi-ear/                          ← cloned: github.com/MalAstronomy/sbi-ear
├── scripts/
│   └── plot_gebhard_cornerplot.py
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

### Step 7 — Download ABC database
Changeat & Yip 2023, available on Zenodo. Search "ABC atmospheric retrieval Changeat 2023 Zenodo."
- 105,887 TauREx3 forward model spectra
- 26,109 nested-sampling posterior distributions
- Public, no registration required

### Step 8 — Understand ABC format
Open one spectrum in Python, print the dataset structure, check parameter names, number of spectral bins, wavelength range. The ABC parameter space (TauREx3) differs from Vasist/Gebhard (petitRADTRANS) — different parameter names, possibly different dimensionality.

### Step 9 — Write data loader adapter
fm4ar's data loaders expect petitRADTRANS-format HDF5 files with specific keys. ABC uses TauREx3 with different keys and structure. A thin adapter class or wrapper function maps ABC → fm4ar's expected format without modifying fm4ar's source.

### Step 10 — Run fm4ar on ABC
Train briefly on ABC data (100k spectra is enough for Phase 0 — not the full 105k). Run inference on a held-out ABC test spectrum. Produce corner plot.

### Step 11 — Run sbi-ear on ABC
Same — adapt sbi-ear's data loading to ABC format, run inference, produce corner plot.

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
- [ ] ABC database downloaded
- [ ] ABC format understood
- [ ] Data loader adapter written (fm4ar → ABC)
- [ ] fm4ar posteriors on ABC — corner plot + IS-efficiency number
- [ ] sbi-ear posteriors on ABC — corner plot + IS-efficiency number
- [ ] All scripts and figures committed to `phase-0-track-1` branch

→ See `01_phase1_track1_transformer_encoder.md` (to be written at Phase 1 start)
