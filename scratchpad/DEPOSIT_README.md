# MIRAGE — data, models, and posteriors

Companion data record for MIRAGE, a simulation-based inference pipeline for JWST
exoplanet atmospheric retrieval (radius inference + optimal-transport calibration).
The source code lives in the accompanying software archive; this bundle holds the
reduced data, trained models, and posterior samples needed to reproduce the
real-data results.

## Contents

### `spectra/` — reduced JWST transmission spectra
| file | planet | instrument | notes |
|------|--------|-----------|-------|
| `jwst_wasp39b_spectrum.csv` | WASP-39b | NIRSpec PRISM | hot Saturn |
| `jwst_wasp96b_spectrum.csv` | WASP-96b | NIRISS SOSS | hot Saturn — **self-reduced end-to-end from the raw MAST ramps** (program 2734) with exoTEDRF (box order-1) + a batman light-curve fit; 90 bins, 0.85–2.81 µm. 1.4 µm water feature detected at 5.8σ. |
| `jwst_k2_18b_spectrum.csv` | K2-18b | NIRISS | cold sub-Neptune |

Columns: `wavelength_um`, `transit_depth` = (Rp/R★)², `depth_error`.

### `models/` — trained flow-matching posterior estimators (θ = [Rp, T, log H₂O, log CO₂, log CH₄, log CO, log NH₃])
| dir | target | grid |
|-----|--------|------|
| `noisecond_rad_cov/` | WASP-39b radius arm (covariance noise-conditioning) | ABC |
| `noisecond_rad_nocond/` | WASP-39b radius arm (no conditioning) | ABC |
| `noisecond_rad_nocond_wasp96/` | WASP-96b | 90-bin NIRISS grid |
| `noisecond_rad_nocond_k218/` | K2-18b | 150-bin NIRISS grid |

Each holds `model__best.pt` + `config.yaml`.

### `posteriors/` — real-data retrieval outputs
- `wasp39b_ns_posterior_{fitrad,wasp96,k218}.npz` — independent nested-sampling anchors
  (TauREx, same θ space), used as the calibration reference.
- `{abl_rad_nocond,wasp96,k218cov}_samples.npz` — MIRAGE posterior draws on the real
  spectra (θ, log q, observed depth/σ, covered-bin mask, wavelength grid).

### `figures/`
- `fig6_multitarget.png` — the three real-data retrievals side by side.

## Real-data results (validated against the independent NS anchors)

| planet | best-fit χ²/dof | radius (MIRAGE, RJup) | literature | coverage vs NS |
|--------|-----------------|-----------------------|------------|----------------|
| WASP-39b | 0.06 | 1.23 | 1.28 | 7/7 |
| WASP-96b | 0.079 | 1.196 | 1.20 | 7/7 |
| K2-18b | 0.199 | 0.233 | 0.235 | 6/7 |

The method recovers the planet radius and a water-rich atmosphere across planet class
(hot Saturn ↔ cold sub-Neptune), instrument (NIRSpec PRISM ↔ NIRISS SOSS), and data
provenance (published ↔ self-reduced).

## Reproduce
See the software archive: environment setup, `scripts/generate_training_data.py`,
`scripts/train.py`, `scripts/real_ess.py`, and `scripts/run_wasp96_eval.sh`. The
WASP-96b reduction is driven by `scripts/wasp96_setup_reduction.py` +
`scripts/wasp96_fit_lightcurves.py` (exoTEDRF, run from the raw MAST `uncal`).

## License
MIT.
