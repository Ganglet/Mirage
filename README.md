# MIRAGE
**Multi-Instrument Retrieval with Adaptive Generative Estimation**

Simulation-to-real domain-adaptive atmospheric retrieval for JWST exoplanet transmission spectroscopy. Targets ICML 2027.

---

## What this is

Existing ML atmospheric retrieval systems train on simulated spectra and collapse on real JWST observations. MIRAGE closes that gap through three mechanisms: noise-conditioned flow-matching inference, multi-instrument transformer encoding, and optimal-transport calibration against validated real-data posteriors.

---

## Setup

```bash
conda create -n mirage python=3.11
conda activate mirage
pip install -r requirements.txt
```

---

## Data

Raw JWST observations and training datasets are **not stored in this repository**. Download them using the provided scripts:

```bash
# WASP-39b JWST observations from MAST (Track 2)
python scripts/download_mast.py

# ABC synthetic benchmark (Changeat & Yip 2023)
python scripts/download_abc.py
```

Downloaded data lives in `data/` which is gitignored. See `ACKNOWLEDGEMENTS.md` for data sources and citations.

---

## Structure

```
Project/
├── scripts/          # download and preprocessing scripts
├── src/              # model code (built phase by phase)
├── data/             # gitignored — downloaded locally by each user
├── figures/          # output figures from evaluation
├── Paper/            # reference papers
└── problems_and_decisions.md
```

---

## Status

Pre-development. Phase 0 commencing.

| Phase | Weeks | Status |
|-------|-------|--------|
| 0 — Environment + baseline reproduction | 1–2 | **Track 1 Complete ✓** / Track 2 In progress |
| 1 — Transformer encoder + domain randomisation | 3–5 | Not started |
| 2 — Noise conditioning + corpus expansion | 6–8 | Track 1 in progress / **Track 2 scaffold complete** |
| 3 — OT calibration + first real-data result | 9–10 | Not started |
| 4 — Full validation + stress test | 11–13 | Not started |
| 5 — Manuscript + submission | 14–16 | Not started |

### Phase 0 Details

**Track 1 — Baseline Reproduction (Complete)**
- fm4ar (FMPE) and sbi-ear (NPE) baselines reproduced on ABC synthetic dataset
- IS-efficiency verification: FMPE ε=19.05% (paper: 19.1%), NPE ε=11.62% (paper: 11.6%)
- Environment validated on M1 Pro MPS
- Documentation: `Documentation/00_track1_environment_baselines.md`
- Branch: `phase-0-track-1`

**Track 2 — JWST Data Preparation (In progress)**
- WASP-39b observations retrieved from MAST (4 instruments: NIRSpec PRISM/G395H, NIRISS SOSS, NIRCam F322W2)
- Multi-instrument spectrum standardization and quality filtering
- Unified dataset creation for Phase 3 real-data validation
- Notebook: `MIRAGE_PHASE-0.ipynb`
- Branch: `phase-0-track-2`

### Phase 2 Track 2 Details

**Track 2 — Observational Corpus Expansion + Diagnostics (Scaffold Complete)**
- Registered expansion targets: WASP-96b NIRISS/SOSS and HD 209458b NIRCam grism
- Added posterior diagnostics for IS-efficiency, 68%/95% coverage, corner plots, and deep-ensemble KL summaries
- Documentation: `Documentation/02_track2_observational_corpus_and_diagnostics.md`
- Branch: `phase-2-track-2`
