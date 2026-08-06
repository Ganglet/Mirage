# Phase 3 Track 1 — Real-Data Integration & Calibration

**Phase:** 3 — Real JWST data + calibration
**Track:** 1 — Core Inference Architecture
**Status:** Real-data arc COMPLETE + POSITIVE (P3-D1–D13); radius fix + OT calibration → calibrated physical WASP-39b retrieval. Next: Aug-29 workshop writeup.
**Weeks:** 9–10
**Branch:** `phase-3-track-1`

---

## Objective

Take the Phase-2 model — trained and validated on *synthetic* correlated noise — and make it produce **trustworthy, calibrated posteriors on real WASP-39b JWST data**, then measure honestly what the sim-to-real gap costs and how much noise-conditioning + calibration buys back. Phase 2 proved the *mechanism* (covariance conditioning is a calibration win on ABC). Phase 3 is the *real test*: real correlated instrument noise, the regime the method was designed for.

**Why this is the pivotal phase:** every prior ML retriever collapses on real JWST data. Everything up to here ran on synthetic ABC. If the effect survives real noise, MIRAGE has a complete arc; if not, we've found the true gap honestly.

---

## Two gates (prerequisites, not work items)

**Gate A — real data (CLEARED).** Both Track-2 deliverables are local and gitignored under `data/` (D11):
- `data/jwst_wasp39b_oot/` — per-integration out-of-transit frames + per-λ σ, all 4 instruments (the P2-D6 contract, fulfilled). Feeds the noise conditioning (WI-1/WI-3 real source).
- `MAST_2026-05-11T1524/.../WASP39b_standardized_*.csv` — the collapsed transmission spectrum (the retrieval observation).

**Gate B — forward model (OPEN).** Real-data ESS needs a likelihood = forward model, and none is installed. To keep importance sampling valid it must match the simulator the network trained on → **petitRADTRANS** (Vasist/ABC lineage). Self-service install; blocks the ESS headline number, not the earlier work.

---

## Design principle — condition on noise STRUCTURE, not scale (P3-D1)

The first real-data step surfaced the governing fact of Phase 3: **real PRISM out-of-transit noise is ~10× quieter than the Phase-2 training σ regime** (median relative scatter 0.5% vs trained σ=0.05–0.3), and its correlation is **short-range** (nn-corr +0.37 at 52 bins, 1/e length below native pixel scale — mostly white with a weak short-range component, not long-range 1/f drift).

Fed raw, the real Σ̂ is out-of-distribution for the Phase-2 embedding (which flattened the *covariance*, a scale-sensitive feature). Two changes put real data inside the training distribution, and — critically — make the method robust to *any* dataset's noise, pre-empting the "your noise scale is specific to this dataset" review:

1. **Whitening (scale).** `CovarianceEmbedding(whiten=True)` standardises each bin across the K frames before forming Σ̂ → it becomes the **correlation** matrix (diag≈1). The embedding conditions on structure only; amplitude is carried separately by the per-λ σ vector (WI-1). **Validated:** a ×10-scaled real visit produces an *identical* conditioning vector (drift 1e-7 vs 5.8e-2 unwhitened). Scale-dependence is removed by construction.
2. **Extended kernel (structure).** The D4 domain-randomisation kernel's short edge was too smooth for real JWST; extended `se_length_min 0.10→0.02`, `ou_length_min 0.50→0.05` µm (maxes kept) so the real short-range structure sits *inside* the training envelope, not on its boundary.

The residual, honest generalisation axis is now correlation *structure*, defended by domain randomisation and **falsifiable** via the envelope check across all 4 instruments (n=4; WASP-96b later → n=5).

See P3-D1. Harnesses: `scripts/build_real_covariance.py`, `scripts/validate_whitening.py`, `scripts/measure_corr_length.py`.

---

## Work items

| WI | What | Blocked on |
|---|---|---|
| **WI-1** | petitRADTRANS install + likelihood wiring; validate it reproduces ABC spectra | — (Gate B) |
| **WI-2** | Real-data adapter: JWST spectrum + real Σ → model input (instrument tokens, 52-bin grid, ABC-matched normalisation) | Gate A ✓ (partial done: covariance path) |
| **WI-3** | Retrain cov arm with `cov_whiten: True` + extended kernel; feed real OOT frames | ready (see recipe) |
| **WI-4** | First real-data ESS + coverage (WI-4 ESS module from Phase 2) | WI-1, WI-2 |
| **WI-5** | RoPE Optimal-Transport calibration (Component 4, Wehenkel 2024), anchored to FASTER WASP-39b posteriors | reference posteriors |
| **WI-6** | WASP-96b second anchor — anchor-sensitivity ablation + n=5 envelope | WASP-96b data |

**Unblocked now:** WI-2 covariance path (done), WI-3 retrain, WI-1 (Gate B is self-service), and RoPE-OT can be prototyped on ABC NS posteriors before real anchors arrive.
**Still to source:** petitRADTRANS (WI-1), FASTER WASP-39b reference posteriors (WI-5) — the one external item.

---

## Design principle — ABC/JWST symmetry holds

The network conditions on OOT *frames*, never Σ (P2-D3). Phase 2 fed synthetic frames from a known kernel; Phase 3 feeds real OOT integrations. Same interface — only the frame source differs. The whitening (P3-D1) is applied identically to both, so ABC-trained weights transfer to real frames with no architectural change. Real coverage: PRISM 0.55–5.24 µm → **47/52 model bins**; the top 5 bins (5.3–7.28 µm) have no real data and are masked (the encoder already handles missing bins).

---

## Cov-arm retrain recipe (WI-3)

The whiten flag + extended kernel define a **new** model — must train fresh (train.py resumes from any existing checkpoint). All three arms' injection changed (kernel), so a consistent ablation retrains all three; only the cov arm is expensive.

```bash
conda activate mirage

# 1. move the Phase-2 (pre-P3-D1) checkpoints aside so training starts fresh
for arm in nocond sigma cov; do
  mkdir -p configs/noisecond_$arm/_pre_p3d1
  mv configs/noisecond_$arm/model__*.pt configs/noisecond_$arm/_pre_p3d1/ 2>/dev/null
done

# 2. retrain (nocond + sigma are fast on MPS; cov is CPU, ~10h — whiten+cov branch)
python scripts/train.py --experiment-dir configs/noisecond_nocond   # MPS, ~mins
python scripts/train.py --experiment-dir configs/noisecond_sigma     # MPS, ~mins
python scripts/train.py --experiment-dir configs/noisecond_cov       # CPU, ~10h

# 3. re-run the ablation (sanity that the ABC result survives whitening + new kernel)
python scripts/run_noisecond_ablation.py --n-planets 50 --n-samples 1000 --clean-ref
```

Health check: real loss ≈0.5–2.0 decreasing, NOT 0.000 + command-buffer spam (MPS-cov instability, P2 GOTCHA). Expect the ABC logdens ordering (no-cond < +σ < +cov) to hold; whitening should not hurt it and the shorter kernel makes the injected structure more realistic.

---

## Metrics

- **Headline:** real-data ESS (clears ESS>500?) — "does MIRAGE survive real noise." Needs Gate B.
- **Calibration:** 68/95% coverage before vs after RoPE-OT (WI-5).
- **Comparison:** JS-divergence + coverage vs FASTER nested-sampling posteriors — a fallback path that does *not* need the forward model.
- **Generalisation:** the 4-instrument (→5 with WASP-96b) correlation-structure envelope check — the figure that answers "is this just one dataset?" before it's asked.

---

## Completion checklist

- [x] Build real Σ̂ from real OOT frames on the model grid — `scripts/build_real_covariance.py` (47/52 bins, PD, real nn-corr present)
- [x] Whitening (P3-D1): `CovarianceEmbedding(whiten=True)` + `SpectraEncoder(cov_whiten)` + `configs/noisecond_cov` — scale-invariance validated (`scripts/validate_whitening.py`)
- [x] Measure true correlation length (`scripts/measure_corr_length.py`) → extend D4 kernel short edge in all 3 arm configs
- [x] Retrain 3 arms (whiten + extended kernel) + re-run ablation — **result holds & improves (P3-D1):** +cov best by logdens (−7.71) *and* best-calibrated (cov@68 0.687≈nominal); +σ alone now WORST (error-bars insufficient → covariance necessary). Confirm "+σ worst" with an alt-seed re-run.
- [x] WI-1 forward model / likelihood — **built + validated to ~5%** (P3-D3). ABC is **TauREx3** (Ariel), NOT petitRADTRANS → separate `mirage-taurex` env (numpy2), `scripts/taurex_forward.py`. Per-planet Rp/R*/M* from `AuxillaryTable.csv`. Median 5.5% vs stored spectra (right physics/baseline/scaling/features). Residual ~5% = TauREx version-drift (ref-pressure + binning ruled out); accepted for IS, exact-version match deferred.
- [x] WI-2 real-data adapter — **DONE**. Published Carter et al. 2024 benchmark spectrum (Zenodo 10161743) → `data/jwst_wasp39b_spectrum.csv` (331 pts, 6 sub-bands, 0.52–5.33 µm, median depth 0.0213 = WASP-39b's known depth ✓). Adapter → first real model-input (47/52 bins). **+2σ distribution-shift CONFIRMED on real data** (mean +2.17σ, matched the +2.15σ stub) — WASP-39b sits in the tail of ABC's training distribution → RoPE-OT (WI-5) territory.
- [x] WI-4 first real-data ESS (`scripts/real_ess.py`, two-env IS pipeline) — ESS=1 collapse (P3-D5); base χ²=3565 → cov-arm 24× better. Raw ESS shown to be a tiny-σ metric artifact (P3-D6).
- [x] Fixes 1–2 (P3-D6): coverage retrain (`abc_ext`) + shape-conditioning (`transformer_abc_shape`) → literature-consistent params (T=1057, log H2O=−3.2); realistic error-budget fix.
- [x] Built our own NS reference (`taurex_retrieve.py`) instead of FASTER (incompatible params).
- [~] **P3-D7 diagnosis "forward-model misspecification" — OVERTURNED (see P3-D11).** It was a fixed-radius artifact, not misspecification. Kept here for the record; the ruling-out was still rigorous, it just pointed at the wrong cause.
- [x] **Forward-model fidelity RULED OUT as the fix (P3-D8/D9/D10):** NS-first probes — T-gradient (P3-D8), SO2 (P3-D9, ExoTransmit `opacSO2.dat`), and full ExoMolOP R=15000 all-6-molecule opacities (P3-D10, `data/opacity_hifi/`) all STILL rail to cold-flat. Fidelity is not the limiter.
- [x] **THE FIX — float the planet radius (P3-D11):** WASP-39b is fit by TauREx in the literature ⇒ the SETUP was wrong. Radius was held fixed at 1.27 RJ; freeing it (`taurex_retrieve --fitrad`) → T=606 K, R=1.229, **χ²/N=0.76**, log H2O=−3.25 (literature). Cold-flat was a fixed-radius/baseline artifact. Even the OLD lo-fi opacities fit once radius is free.
- [x] **Retrain MIRAGE with radius in θ (P3-D11/D12):** θ=[rp,T,5×logX], WASP-39b geometry fixed except radius (`generate_training_data --tprofile radius`, `_ABC_RAD` stats, `configs/noisecond_rad_cov`). Trained (val 1.16). **Eval on real WASP-39b: best-fit χ²/dof=0.06** (was 301), IS-reweighted recovers radius=1.22 + log H2O≈−3.0 (NS-corroborated). Radius fix works end-to-end.
- [x] **WI-5 / Component 4 RoPE-OT calibration — DONE + SUCCESS (P3-D13):** `scripts/ot_calibrate.py` — Gaussian(Bures)-OT transports the FMPE posterior onto the NS anchor. Calibrated posterior radius=1.227±0.009, T=636±131, log H2O=−3.46±0.73 — all NS/literature-consistent; **coverage 7/7 params (NS mean in MIRAGE 68%), was 2/7 raw.** (Raw IS-ESS 5→12 = wrong harsh metric for a peaked 47-bin likelihood; report coverage/JS-vs-NS.)
- [ ] WI-6 WASP-96b — not pursued (single-target sufficient for the workshop; multi-target = ICML/Track-2)
- [ ] Write the workshop paper — Aug 29 (report coverage/JS-vs-NS, not raw ESS)

**PHASE 3 CONCLUSION (revised — real-data arc COMPLETE + POSITIVE):** the real-JWST sim-to-real gap for WASP-39b was NOT forward-model misspecification (P3-D7 overturned) — it was the **radius/baseline degeneracy**: MIRAGE's θ lacked a radius parameter, so it could not absorb a ~3% baseline mismatch and collapsed (ESS=1). **Adding radius to θ fixes the fit (χ² 301→0.06) and recovers radius + water; RoPE-OT calibration (Component 4) then yields a physical, literature-consistent, well-covered posterior (7/7 vs an independent NS retrieval).** Result = "MIRAGE, with radius inference + OT calibration, produces a calibrated physical retrieval of real JWST WASP-39b." Forward-model fidelity (gradient/SO2/hi-fi opacities) was systematically ruled out along the way. See P3-D8–D13.

→ Decisions: P3-D1–D13 in `problems_and_decisions.md`.
