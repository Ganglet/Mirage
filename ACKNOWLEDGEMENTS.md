# Acknowledgements

## Codebases

**fm4ar** — Flow matching for atmospheric retrieval of exoplanets.
Gebhard et al. 2025, A&A 693, A42. `github.com/timothygebhard/fm4ar`.
Used as the flow-matching posterior-estimation backbone (pristine upstream; MIRAGE extends it
via runtime registration, not a fork).

**sbi-ear** — Neural posterior estimation for exoplanetary atmospheric retrieval.
Vasist et al. 2023, A&A 672, A147. `github.com/MalAstronomy/sbi-ear`.
NPE baseline for the synthetic benchmark.

---

## Methods reimplemented (method only; no source code used)

**SpectraFM** — Transformer foundation model for stellar spectra.
Koblischke & Bovy 2024 (arXiv:2411.04750). MIRAGE's `SpectraEncoder` independently reimplements
their per-bin tokenisation + wavelength positional encoding (their Eq. 2).

**DINGO** — Noise-conditioned neural posterior estimation for gravitational waves.
Dax et al. 2021, PRL 127, 241103. The residual-covariance embedding ports DINGO's PSD noise
conditioning to JWST transmission spectroscopy.

**RoPE — robust posterior estimation via optimal transport.**
Wehenkel et al. 2024. Basis for the optimal-transport calibration (Component 4), anchored here to
an independent nested-sampling reference posterior.

---

## Datasets

### Synthetic
**Ariel Data Challenge (ABC) benchmark** — Changeat & Yip 2023.
TauREx3 forward models + nested-sampling posteriors; the synthetic training/benchmark corpus
(6-parameter transmission: T + log H₂O/CO₂/CH₄/CO/NH₃).

### Real JWST transmission spectra (retrieval targets)
- **WASP-39b** — benchmark spectrum, Carter et al. 2024, Nature Astronomy; JWST ERS Proposal 1366.
- **WASP-96b** — Radica et al. 2023, MNRAS 524, 835 (Awesome SOSS, NIRISS/SOSS).
- **K2-18b** — Madhusudhan et al. 2023, ApJL 956, L13 (NIRISS + NIRSpec G395H).

### Real JWST out-of-transit frames + per-wavelength uncertainties
Reduced from the JWST `x1dints` time-series products (MAST, `mast.stsci.edu`) by the Track-2
data pipeline — per-integration out-of-transit frames feeding the covariance / noise conditioning.

---

## Opacities & forward model

**TauREx3** — Al-Refaie et al. 2021, ApJ 917, 95. The forward model / likelihood engine, matched
to the ABC simulator.

**ExoTransmit opacities** — Kempton et al. 2017 (the ABC-matched line-by-line cross-sections),
including the SO₂ cross-section from the ExoMol **ExoAmes** line list (Underwood et al. 2016).

**ExoMolOP high-fidelity opacities** — Chubb et al. 2021, A&A 646, A21 (R=15000 cross-sections),
built on the ExoMol database (Tennyson et al. 2020): H₂O POKAZATEL (Polyansky et al. 2018),
CO₂ UCL-4000 (Yurchenko et al. 2020), CH₄ YT34to10 (Yurchenko et al. 2017), CO Li2015
(Li et al. 2015), NH₃ CoYuTe (Coles et al. 2019), SO₂ ExoAmes.

*Not used:* petitRADTRANS was installed early as a candidate engine but is the wrong simulator for
the ABC corpus (TauREx3), so it is not part of the pipeline. Published FASTER posteriors use an
incompatible parameterisation, so an independent TauREx nested-sampling reference is used as the
calibration anchor instead.

---

## Core libraries

**astropy** (Harris et al. 2020) — FITS I/O and wavelength conversions.
**pandas** (McKinney 2010) — spectrum standardisation and I/O.
**PyTorch**, **numpy**, **h5py**, **numba** — training, arrays, storage, forward-model acceleration.

---

## Contributors

- **Track 1 (core inference architecture):** Angshuman Chakravertty — transformer encoder,
  flow-matching inference, noise conditioning, radius parameterization, OT calibration, evaluation.
- **Track 2 (JWST data systems):** Vedanth Raj — MAST reduction pipeline, out-of-transit frame
  extraction, domain-randomisation and evaluation tooling.
