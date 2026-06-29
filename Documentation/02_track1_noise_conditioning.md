# Phase 2 Track 1 — Noise-Conditioning Integration

**Phase:** 2 — Noise-Conditioning Integration
**Track:** 1 — Core Inference Architecture
**Status:** Planning
**Weeks:** 6–8
**Branch:** `phase-2-track-1`

---

## Objective

Condition the flow on the observation's noise structure, not just the spectrum. Phase 1 proved the transformer encoder beats the flat MLP on noise-free-context ABC. Phase 2 makes the network noise-aware in two stages — a per-wavelength uncertainty vector and a learned embedding of the *correlated* noise structure — then promotes the ESS diagnostic to a reusable module and runs the first proper benchmark against all three baselines.

**Why this matters:** real JWST noise is correlated across wavelengths (detector systematics, 1/f noise). A network conditioned only on the spectrum, or on marginal error bars, cannot distinguish a spectral feature from a correlated noise excursion. The posterior it returns is overconfident in exactly the regime the project exists to fix. This is DINGO's power-spectral-density conditioning (Dax et al.) ported from gravitational waves to JWST transmission spectroscopy.

---

## Blueprint mapping

This phase implements the Phase 2 / Track 1 column of `MIRAGE_Blueprint_v1` verbatim, plus one deliberate forward-pull:

| Blueprint item | Work item |
|---|---|
| "per-wavelength error vector conditioning" | WI-1 |
| "learned residual-covariance embedding from out-of-transit baselines" | WI-3 |
| "Effective sample size module implementation" | WI-4 |
| "Full synthetic benchmark evaluation against all three baselines" | WI-5 |
| *(not in this column — pulled forward from D4)* | WI-2 |

WI-2 (correlated-noise injection) is **not** in the Track 1 Phase 2 column. The blueprint phrase "from out-of-transit baselines" assumes real data with real OOT frames; ABC is synthetic and has none. Building and validating a residual-covariance embedding on ABC therefore requires synthesizing the correlated noise it is meant to capture. That injection module belongs to the D4 domain-randomisation pipeline, pulled forward here so Part 2 is testable before real JWST data arrives in Phase 3. See P2-D2.

---

## Design principle — ABC/JWST architectural symmetry

The network must **never see the covariance Σ directly.** It sees out-of-transit (OOT) noise frames and embeds the correlation structure from them. This holds identically on both data sources:

| | Real JWST (Phase 3) | ABC synthetic (Phase 2) |
|---|---|---|
| Source of OOT frames | real out-of-transit integrations of the visit | K realisations drawn from a known kernel Σ_true |
| Σ available to network? | no — estimated from frames | no — estimated from frames |
| Σ available to *evaluator*? | no (empirical Σ̂ only) | yes — ground truth, enables sanity checks |

Because the network's interface is the OOT frames (not Σ), the Phase 2 covariance embedding transfers to Phase 3 with no architectural change. Only the frame source differs. See P2-D3.

---

## Work items

### WI-1 — Per-wavelength error vector conditioning (Part 1)

Add σ as a third token feature in the encoder: `[flux_i, σ_i, instrument_emb_i]`.

- **File:** `fm4ar/fm4ar/nn/spectra_encoder.py`
  - `token_proj`: `Linear(1 + inst_dim → d_model)` → `Linear(2 + inst_dim → d_model)`
  - `required_keys`: add `"error_bars"`
  - `forward`: concat `error_bars` into `token_feat` at the flux concat site
- **σ source:** condition on `context["error_bars"]`, **not** the static HDF5 `instrument_noise`. When the injection transform (WI-2) is active, `error_bars = √diag(Σ_true)` matches the realisation actually added to `flux`. See P2-D1.
- Smallest of the five items: one input feature + config bump.

### WI-2 — Correlated-noise injection + OOT-frame surrogate (D4 forward-pull)

- **File:** `fm4ar/fm4ar/datasets/noise.py`
  - New `CorrelatedNoiseGenerator(NoiseGenerator)`. Builds Σ over wavelength from a kernel: squared-exponential (smooth detector drift) + a 1/f-like component. Hyperparameters (length scale, amplitude) **randomised per draw** — this randomisation *is* the D4 domain-randomisation injection.
  - Sampling via Cholesky of Σ.
  - Extend the interface (currently diagonal-only: `sample_error_bars → (n_bins,)`, independent-bin `sample_noise`) to expose Σ and to draw K OOT noise-only frames.
  - Register in `get_noise_generator`.
- **File:** `fm4ar/fm4ar/datasets/data_transforms.py`
  - Injection transform: per sample, draw a Σ_true, add correlated noise to `flux`, and emit both `error_bars` (= √diag Σ, feeds WI-1) and `oot_frames` (feeds WI-3) into the context dict. One transform serves WI-1, WI-3, and D4.
- **Base spectrum:** inject on top of the existing ABC `flux` (no noise-free base exists in `abc_train.hdf`). ABC's baked-in noise is diagonal — it adds to σ; the injected component is the only off-diagonal structure, so Σ_true's off-diagonals stay exactly known. See P2-D4.

### WI-3 — Residual-covariance embedding net (Part 2)

- **File (new):** `fm4ar/fm4ar/nn/covariance_embedding.py`
  - `oot_frames (K, n_bins) → empirical Σ̂ → top-k eigenmodes → MLP → embedding`.
  - Swappable embedding net behind a small factory (flatten-MLP adequate for ABC's P=52; top-k eigenmode route is the DINGO-PSD analogue that scales to JWST's larger, multi-instrument P).
- **Context wiring:** concat the covariance embedding into the context vector; bump `dim_context` in the flow.

### WI-4 — ESS module (NEW — blueprint "ESS module implementation")

Promote the Phase 1 ad-hoc ESS computation ([P1-D2], [P1-D3] — inline, N=2,000, tol 1e-3) to a reusable module.

- **File (new):** `fm4ar/fm4ar/eval/ess.py` (or `scripts/` if eval pkg absent)
  - ESS = ε · N, with the D7 thresholds baked in: primary ESS > 500, "high-quality" ESS > 2,500.
  - Single entry point consumed by both Phase 2 (this benchmark) and Phase 3 (first real-data ESS measurement). The Phase 3 deliverable needs a module, not a copied script.
- Configurable N and ODE tolerance (diagnostic: N=2,000 / 1e-3; publication: N=10,000 / 1e-5–1e-7 on cluster).

### WI-5 — Full synthetic benchmark (Part 3)

Same held-out ABC planets, same injected correlated noise (where applicable), measured with the WI-4 module.

| Run | Expectation |
|---|---|
| Vasist NPE | baseline |
| Gebhard FMPE | baseline |
| Transformer-FMPE (Phase 1) | Phase 1 result, degraded under correlated noise |
| + σ only (WI-1) | partial recovery |
| + σ + covariance embedding (WI-3) | best — headline number |
| recover-Σ sanity check | embedding correlates with known kernel length scale |

Headline result: *covariance conditioning recovers X% of the ESS lost to correlated noise* (WI-3 row vs WI-1 row). A diagonal-noise-only ABC table cannot show this — under diagonal noise the covariance embedding is information-equivalent to the σ vector. See P2-D2.

---

## Dependency order

WI-2 (generator) → WI-1 (σ wiring) ∥ WI-3 (embedding) → WI-4 (ESS module) → WI-5 (benchmark). WI-2 first: everything downstream needs the injected noise and the OOT frames it produces.

---

## Completion checklist

> Note: per the P2-D8 refactor, all WI code lives in the `mirage/` package (not `fm4ar/`); paths below are updated accordingly.

- [x] `CorrelatedNoiseGenerator` + OOT-frame draw — `mirage/datasets/noise.py` (verified: PD Σ, flat diag=σ², OOT frames recover Σ to 3.9% at 20k frames)
- [x] Injection transform emitting `error_bars` + `oot_frames` — `mirage/datasets/transforms.py` (verified: shared-Σ consistency, error_bars == OOT per-bin std to 0.7%, input not mutated, registration + guard)
- [x] σ token feature wired into `SpectraEncoder` — `mirage/nn/spectra_encoder.py` (verified: `use_error_bars` flag, backward-compatible when off, σ consumed when on, +d_model params, registry path)
- [x] `covariance_embedding.py` written + concatenated into flow context — `mirage/nn/covariance_embedding.py` (verified: flatten + eigen, Σ̂ matches np.cov, dim bump 256→320 inferred at build time, end-to-end with injection transform)
- [x] ESS module with D7 thresholds — `mirage/eval/ess.py`, consumed by `scripts/compute_is_efficiency_transformer_abc.py` (verified: regression-identical to old inline formula on 1000 vectors, threshold boundaries, aggregate; reusable by Phase 3)
- [ ] New config `configs/transformer_abc_noisecond/config.yaml`
- [ ] Benchmark table: 6 rows, ESS on held-out ABC
- [ ] Recover-Σ sanity check: embedding vs known kernel length scale
- [ ] Headline number: % ESS recovered by covariance conditioning

→ See `03_track1_calibration.md` (to be written at Phase 3 start)
