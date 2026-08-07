# Phase 4 Track 1 — Full Evaluation, Benchmark & Figures

**Phase:** 4 — evaluation / benchmark / figures for the real-data result
**Track:** 1 — Core Inference
**Status:** COMPLETE — ablation + benchmark + 5 figures + robustness + ABC-win figure; feeds the Aug-29 workshop paper
**Branch:** `phase-3-track-1` (Phase-4 work continues here)

---

## Objective

Phase 3 established the real-data result (radius fix + OT calibration → physical WASP-39b
retrieval, P3-D11/D13). Phase 4 turns that into publishable evidence: the noise-conditioning
ablation on real data, the consolidated benchmark table, and the figures — and does so
**honestly**, reporting the negative ablation result rather than over-claiming.

---

## 4.1 / 4.2 — Noise-conditioning ablation on real WASP-39b (P4-D1)

Trained the two missing radius-model arms (`configs/noisecond_rad_nocond`, `_sigma`; cov
already done) and evaluated all three on the real spectrum (`scripts/run_ablation_real.sh`).

| arm | best χ²/dof | coverage/7 vs NS | mean JS→NS |
|---|---|---|---|
| nocond | 0.061 | 3/7 | **0.358** (closest) |
| σ-only | 0.132 | 0/7 | 0.365 |
| σ+cov | 0.116 | 3/7 | **0.430** (farthest) |

**Honest negative result: covariance noise-conditioning does NOT help on real WASP-39b** — it
is marginally *worst* by both agreeing metrics. All arms recover radius≈1.2 (the radius fix is
robust). **Why:** the cov arm is the only one that ingests the real JWST out-of-transit noise
frames, which are out-of-distribution vs the domain-randomised training kernel (P3-D1: real
noise ~10× below trained σ, short-range). It conditions on OOD input and is perturbed; nocond
ignores noise and does fine. Whitening (P3-D1) narrowed but did not close that gap. A single
real target also cannot capture cov's calibration-over-population benefit — that is the Phase-2
ABC result (cov > σ > nocond by coverage over many planets with known truth).

**Framing:** noise-conditioning is validated on **synthetic ABC (Phase 2)**; on real JWST the
dominant sim-to-real gap was the **radius/baseline degeneracy** (fixed, P3-D11) plus
**calibration** (OT, P3-D13). We report the noise-embedding non-transfer honestly — it is a
genuine sim-to-real lesson, and a stronger paper than over-claiming.

---

## 4.3 — Benchmark table

`scripts/benchmark_table.py` → `Documentation/phase4_benchmark.md`. Consolidates every real-data
result: the fixed-radius collapse (best-fit χ²=301, ESS=1) → radius arms (χ²≈0.06, radius
recovered) → cov+OT-calibration (χ²=0.028, radius 1.227, log H₂O −3.46, **7/7 coverage**) → the
independent NS anchor (χ²=0.76). Headline: **radius fix drops χ² 301→0.06; OT calibration gives
a physical, literature-consistent, 7/7-covered posterior.**

---

## 4.4 — Figures (`figures/`, `scripts/make_figures.py`)

- **`fig1_radius_is_the_fix.png`** — best-fit χ²/dof across every forward-model lever
  (isothermal, +T-gradient, +clouds, +SO₂, +hi-fi ExoMolOP opacities all ≈2.5; forced-physical
  ≈38; **free-radius 0.76**, the only bar below χ²/dof=1). The core narrative in one panel.
- **`fig2_mirage_vs_ns.png`** — OT-calibrated MIRAGE posterior (radius/T/log H₂O) overlaid on
  the independent NS anchor; visual of the 7/7 coverage.
- **`fig3_spectrum_fit.png`** — real WASP-39b transit spectrum + MIRAGE best-fit model (χ²/dof≈0.03).
- **`fig4_corner.png`** — full 7-param corner, calibrated MIRAGE (blue) vs NS anchor (grey);
  contours track across all parameters.
- **`fig5_abc_ablation.png`** — the honest OTHER half: on synthetic ABC (truth known),
  noise-conditioning WINS — (a) no-cond under-covers (0.587<0.68 nominal), +σ/+cov restore to
  0.687; (b) the +σ+cov − no-cond log-density gap GROWS with injected σ (+0.64→+0.87), the
  adaptivity result. `scripts/run_noisecond_ablation.py --save` → `data/real_ess/abc_ablation.npz`.

---

## 4.5 — Robustness (`scripts/run_robustness.sh`)

OT-calibration is robust to its one hyperparameter (defensive inflation): sweeping inflate ∈
{1.0, 1.5, 2.0} gives radius 1.231/1.232/1.233, log H₂O −3.15/−3.09/−2.99, **7/7 coverage at all
three** (ESS 10.9–14.7). With the earlier error-floor stability (5%/2%/1% → radius+H₂O stable,
P3-D13), the calibrated result does not hinge on tuning.

---

## Conclusion

Phase 4 confirms and honestly bounds the real-data result. **What works:** the radius
parameterization (χ² 301→0.06, radius+water recovered) and OT calibration (7/7 vs an
independent NS retrieval) — MIRAGE produces a calibrated, physical, literature-consistent
retrieval of real JWST WASP-39b. **What does not transfer:** the learned covariance
noise-embedding, because real instrument noise is OOD — validated on synthetic ABC, not on the
single real target. The Aug-29 workshop paper reports both.

**For ICML 2027 (Phase 5):** multi-target real evaluation (WASP-96b/K2-18b) — the only setting
that can test whether noise-conditioning helps on real data at population scale — plus a
higher-resolution training grid to lift intrinsic IS efficiency, and a Zenodo release.

→ Decisions: P4-D1 (and P3-D11–D13) in `problems_and_decisions.md`.
