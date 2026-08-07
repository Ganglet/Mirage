# Phase 4 — Real WASP-39b benchmark (main results table)

Independent NS anchor: radius=1.23, T=606, logH₂O=−3.25 (truth radius=1.27, WASP-39b is water-rich). All metrics on the 47 real-covered bins, 5% forward floor.

| model | θ | best χ²/dof | radius(IS) | logH₂O(IS) | cover/7 | ESS |
|---|---|---|---|---|---|---|
| base FMPE, radius FIXED | θ=[T,5mol] | 301.0 | 1.27 (fixed) | −11→−3 | — | 1.0 |
| rad · nocond | θ=[rp,T,5mol] | 0.061 | 1.227 | -3.83 | 3/7 | 2.1 |
| rad · σ-only | θ=[rp,T,5mol] | 0.132 | 1.212 | -10.70 | 0/7 | 1.2 |
| rad · σ+cov | θ=[rp,T,5mol] | 0.116 | 1.203 | -4.91 | 3/7 | 7.4 |
| **rad · cov + OT-cal** | θ=[rp,T,5mol] | **0.028** | **1.227** | **-3.46** | **7/7** | 12.4 |
| NS anchor (fitrad) | reference | 0.76 | 1.229 | -3.25 | — | — |

**Reading:**
- **Radius fix** (P3-D11): adding radius to θ drops best-fit χ² from **301 → ~0.06** and recovers radius≈1.2 (≈truth) — the core real-data result. Robust across ALL arms.
- **OT calibration** (P3-D13): transports the cov posterior onto the NS anchor → **7/7 coverage**, physical/literature-consistent (radius 1.227, logH₂O −3.46).
- **Noise-conditioning ablation** (P4-D1, honest negative): σ+cov does NOT beat nocond on real data (cover 3/7 vs 3/7; JS→NS cov 0.43 > nocond 0.36). cov ingests OOD real OOT noise (P3-D1) → perturbed. Noise-conditioning is validated on synthetic ABC (Phase 2), not here.
- **ESS** is a poor yardstick here (peaked 47-bin likelihood); report coverage/JS-vs-NS.
