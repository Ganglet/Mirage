# Phase 2 Track 2: Observational Corpus Expansion and Diagnostics

**Status:** Implementation scaffold complete  
**Branch:** phase-2-track-2  
**Date:** June 2026  

## Scope

Phase 2 Track 2 expands the MIRAGE real-observation validation corpus and adds reusable posterior evaluation tooling.

New observational targets:

- **WASP-96b NIRISS/SOSS**
- **HD 209458b NIRCam grism**

The target registry lives at `configs/observational_corpus_phase2_track2.csv`. Running `scripts/register_phase2_track2_corpus.py` copies this registry into `data/observational_corpus/` for local, gitignored data work.

## Data Policy

Raw JWST products are not committed to the repository. They should be retrieved from MAST and standardized into the same MIRAGE schema used for WASP-39b:

- `target`
- `instrument`
- `wavelength_um`
- `flux` or `normalized_flux`
- `flux_error` where available
- provenance columns for archive product IDs and reduction notes

This keeps the repository small while preserving reproducible target selection.

## Evaluation Script

`scripts/evaluate_phase2_track2_posteriors.py` accepts simple `.npz` posterior artifacts and computes:

- IS-efficiency from precomputed log weights or from model log density against a reference posterior KDE.
- 68% and 95% central credible-interval coverage when `theta_true` is available.
- Posterior corner plots.
- Deep-ensemble KL diagnostics in the Alvey et al. 2025 style, using pairwise Gaussian KL summaries across ensemble members.

Example:

```bash
python scripts/evaluate_phase2_track2_posteriors.py outputs/wasp96b_niriss_posterior.npz
```

Expected `.npz` arrays:

- `samples`: posterior samples with shape `(n_samples, n_dim)`.
- `log_weights`: optional precomputed log importance weights.
- `log_q`: optional model/proposal log density for `samples`.
- `reference_samples`: optional reference posterior samples for KDE-based IS-efficiency.
- `reference_weights`: optional reference posterior weights.
- `theta_true`: optional reference parameter vector for coverage diagnostics.
- `ensemble_samples`: optional deep-ensemble samples with shape `(n_members, n_samples, n_dim)`.
- `parameter_names`: optional parameter labels.

## Notes

This track is additive. It does not remove or rewrite the existing Phase 0/1 ABC, transformer, or WASP-39b files.
