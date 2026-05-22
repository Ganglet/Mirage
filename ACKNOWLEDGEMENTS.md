# Acknowledgements

## Codebases

**fm4ar** — Flow matching for atmospheric retrieval of exoplanets
Gebhard et al. 2025, A&A 693, A42
`github.com/timothygebhard/fm4ar`
Used as the inference backbone for MIRAGE's flow-matching posterior estimator.

**sbi-ear** — Neural posterior estimation for exoplanetary atmospheric retrieval
Vasist et al. 2023, A&A 672, A147
`github.com/MalAstronomy/sbi-ear`
Used as the NPE baseline for synthetic benchmark comparison.

---

## Datasets

**ABC Database** — Atmospheric retrieval benchmark corpus
Changeat & Yip 2023
`zenodo.org` — 105,887 TauREx3 forward models with 26,109 nested-sampling posteriors
Used as the synthetic benchmark for all baseline comparisons and architecture validation.

**WASP-39b JWST Observations**
JWST Transiting Exoplanet Community Early Release Science Program, Proposal ID 1366
PI: Meech, Annabella
Retrieved from MAST: `mast.stsci.edu`
Instruments: NIRSpec PRISM, NIRSpec G395H, NIRISS SOSS, NIRCam F322W2
Used as the primary real-data validation target.

**FASTER-validated posteriors** (WASP-39b)
Lueber et al. 2025
Used as the calibration anchor for the RoPE optimal-transport calibration step.

---

## Forward Models

**petitRADTRANS**
Mollière et al. 2019
Used by fm4ar and sbi-ear for their original training datasets.

**TauREx3**
Al-Refaie et al. 2021
Used by the ABC database.

**PHOENIX Stellar Atmosphere Library**
Husser et al. 2013
Used for stellar contamination templates in the domain randomisation pipeline.
