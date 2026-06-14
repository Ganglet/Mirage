# MIRAGE — Problems & Decisions Log

Running log of every non-obvious design decision and every problem encountered during implementation. Update this file immediately when a new issue arises or a design choice is made. Entries are numbered sequentially; phase is noted in brackets.

---

## Pre-Development Decisions (Blueprint v1.0, May 2026)

### D1 — Flow-matching over masked autoregressive flow [Blueprint]
Chose continuous normalising flow trained via flow-matching (Gebhard et al. 2025 fm4ar) over masked autoregressive flows (Vasist 2023). Reason: flow-matching trains faster, scales better to high-dimensional posteriors, and fm4ar already demonstrated SOTA performance on the ABC synthetic benchmark. Alternative rejected: normalising flows with neural spline coupling — weaker expressivity on multi-modal posteriors.

### D2 — SpectraFM transformer encoder over MLP / 1D-CNN [Blueprint]
Every existing ML retriever uses a flat MLP or 1D-CNN encoder that concatenates all wavelengths into a single vector. JWST observes most targets across four heterogeneous instruments with different wavelength coverage, resolution, and noise structure. A shared MLP discards instrument identity and handles missing instruments poorly. Decision: adapted SpectraFM (Koblischke & Bovy 2024) transformer where each token = (wavelength, flux, uncertainty, instrument_id). Cross-attention aggregates across instruments naturally; masking handles absent instruments without retraining. Alternative rejected: separate per-instrument encoders with late fusion — fails when an instrument is absent and doubles parameter count.

### D3 — Dual noise conditioning (per-wavelength vector + residual covariance embedding) [Blueprint]
Standard noise conditioning passes only the per-wavelength error bars to the network. Real JWST noise is correlated across wavelengths (detector systematics, 1/f noise) — the covariance structure is not captured by marginal uncertainties alone. Decision: condition on both (a) the per-wavelength uncertainty vector and (b) a learned embedding of the residual covariance matrix estimated from out-of-transit baseline segments of the same JWST visit. Directly analogous to DINGO's power-spectral-density conditioning. Alternative rejected: augmenting error bars with a scalar noise inflation factor — too coarse to capture wavelength-dependent correlation structure.

### D4 — Domain randomisation over CycleGAN as primary adaptation strategy [Blueprint]
Two candidate strategies for closing the sim-to-real gap during training: (a) structured domain randomisation — inject realistic perturbations (correlated noise, stellar contamination templates, pipeline calibration offsets) into simulated spectra before the network sees them; (b) CycleGAN domain translation — train a generator to map simulated spectra into the observed spectral domain. Decision: domain randomisation is the primary strategy. CycleGAN is retained as a controlled ablation experiment (Component 5). Reason: domain randomisation is interpretable, physically motivated, and does not require a separate adversarial training loop that could introduce its own instabilities. The ablation answers the SBI community question of when generative adaptation is necessary versus when structured randomisation suffices.

### D5 — PHOENIX stellar atmosphere library for contamination templates [Blueprint]
Stellar contamination augmentation requires realistic heterogeneous stellar surface models (unocculted spots and faculae). Decision: use PHOENIX model grid parametrised by spot temperature, faculae temperature, and surface covering fraction. Reason: PHOENIX is the community standard in transmission spectroscopy; BT-Settl and ATLAS9 produce consistent results in the relevant temperature range. Known limitation: PHOENIX is 1D. This is defensible because stellar atmospheres are genuinely better approximated as 1D than planetary atmospheres — contamination is a surface-fraction-weighted average, not a geometry with day-night gradients or limb asymmetries. The 1D limitation of PHOENIX and the 1D limitation of the planetary forward model are physically distinct cases.

### D6 — RoPE optimal-transport calibration anchored to WASP-39b posteriors [Blueprint]
Domain randomisation reduces but cannot eliminate sim-to-real gap. Decision: post-training calibration using the Robust Posterior Estimation framework (Wehenkel et al. 2024). Calibration anchor: FASTER-validated nested-sampling posteriors on WASP-39b across NIRSpec PRISM, NIRSpec G395H, NIRISS SOSS, and NIRCam. WASP-39b chosen because it is the most deeply validated JWST planet — multiple independent reduction pipelines, multiple instruments, and the Roy-Perez et al. 2026 multi-pipeline corpus now publicly available. Known limitation: one planet is a thin calibration set. Planned mitigation: add WASP-96b (NIRISS SOSS, different stellar type) as a second anchor in Phase 3 to make anchor sensitivity an ablation rather than an unaddressed gap.

### D7 — IS-efficiency ε expressed as ESS threshold, not raw percentage [Post-blueprint, May 2026]
Blueprint v1.0 stated "ε > 1%" as the deployment threshold. This is under-justified and does not specify sample count. Decision: replace with ESS = ε × N > 500 as the primary threshold. At N = 50,000 posterior samples, ε = 1% gives ESS = 500 — adequate for 1D marginal distributions, borderline for joint inference. This grounds the threshold in importance sampling literature and gives a principled answer to "sufficient for what?" Secondary threshold: ε > 5% (ESS > 2,500) for a result to be cited as "high-quality" in the paper.

### D8 — WASP-39b as primary validation target [Blueprint]
Selection rationale: most extensively reduced JWST planet. NIRSpec PRISM, NIRSpec G395H, NIRISS SOSS, NIRCam spectra all publicly available on MAST. FASTER-validated posteriors provide the calibration anchor. Roy-Perez et al. 2026 provides a multi-reduction pipeline corpus for augmentation training. Minimal stellar contamination (G-type host) isolates the noise-conditioning and flow-matching components from contamination confounds.

### D9 — K2-18b scoped as robustness stress test only [Blueprint]
K2-18b has active scientific controversy over claimed DMS and DMDS biosignature detections, high stellar contamination from an M-dwarf host, and contested pipeline reductions. Decision: MIRAGE applied to K2-18b for robustness evaluation only. The project makes no biosignature claims and frames all K2-18b results as uncertainty quantification under model misspecification. This is the scientifically responsible scope given the controversy.

### D10 — NeurIPS ML4PS 2026 workshop as priority-establishment submission [Blueprint]
Decision: submit preliminary real-data results on WASP-39b to NeurIPS ML4PS workshop (December 2026) before the ICML 2027 deadline. Reason: establishes priority on the architecture, receives community peer-review feedback, and provides a forcing function for Phase 3 results (first real-data IS-efficiency measurement) to be ready by November 2026. arXiv preprint released concurrently, cross-listed astro-ph.EP + cs.LG.

---

### D11 — Data sharing: GitHub for scripts only, MAST + Zenodo for data [Phase 0]
JWST FITS files are 500 MB–2 GB each; total WASP-39b corpus is 30–50 GB. GitHub's hard limit is 100 MB per file. Decision: `data/` is gitignored on both tracks. Raw JWST data is sourced directly from MAST by each collaborator running `scripts/download_mast.py`. Processed/standardised corpus is published to Zenodo as a draft record during development, made public at paper submission. This means the Phase 5 public data release is built into the workflow from day one, not retrofitted. Alternative rejected: shared cloud drive (Google Drive/Dropbox) — no versioning, links break, not reproducible for external reviewers.

---

## Phase 0 Log

See `00_track1_environment_baselines.md` for full Phase 0 Track 1 record.

Key outcomes: ABC pipeline verified end-to-end; NPE ε = 0.025% (M1 Pro, 256-dim/3-transform); FMPE trained to early stopping epoch 140 (val loss 1.388); 76% of ABC NS posteriors malformed — 2,204 valid test planets.

---

## Phase 1 Log

### P1-D1 — sample_and_log_prob_batch called on model wrapper, not network [Phase 1]
`FMPEModel.sample_and_log_prob_batch` exists on the wrapper class; `FMPEModel.network` (type `FMPENetwork`) does not have this method. Calling `model.network.sample_and_log_prob_batch` raises `AttributeError`. Fix: call `model.sample_and_log_prob_batch` directly.

### P1-D2 — N_SAMPLES=2,000 for FMPE IS-efficiency (vs 10,000 for NPE) [Phase 1]
Each FMPE IS-efficiency evaluation requires a full ODE solve (dopri5) on CPU — ~30–60s per planet. Using 10,000 samples per planet would take 5–10× longer. Decision: 2,000 samples per planet for M1 Pro diagnostics. ε is ESS/N so the comparison with NPE (N=10,000) is valid on expectation; variance is higher. Cluster evaluation should use N=10,000 for publication-quality numbers.

### P1-D3 — ODE tolerance set to 1e-3 for IS-efficiency speed [Phase 1]
Default tolerance 1e-7 is too slow for 20-planet evaluation on CPU. Decision: 1e-3 for diagnostic runs. Tighten to 1e-5 or 1e-7 for publication-quality IS-efficiency numbers on cluster.

---

## Phase 2 Log

*(Empty)*

---

## Phase 3 Log

*(Empty)*

---

## Phase 4 Log

*(Empty)*

---

## Phase 5 Log

*(Empty)*
