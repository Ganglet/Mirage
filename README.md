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
| 0 — Environment + baseline reproduction | 1–2 | In progress |
| 1 — Transformer encoder + domain randomisation | 3–5 | Not started |
| 2 — Noise conditioning + corpus expansion | 6–8 | Not started |
| 3 — OT calibration + first real-data result | 9–10 | Not started |
| 4 — Full validation + stress test | 11–13 | Not started |
| 5 — Manuscript + submission | 14–16 | Not started |
