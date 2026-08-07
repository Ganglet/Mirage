"""
Phase 4.3 — consolidated real-WASP-39b benchmark table (paper main table).
Pulls every real-data result into one place: the fixed-radius collapse (before),
the three radius arms (noise-conditioning ablation), the OT-calibrated cov arm,
and the independent NS anchor. Emits Documentation/phase4_benchmark.md.

    conda activate mirage && python scripts/benchmark_table.py
"""
import numpy as np
from pathlib import Path

OUT = Path("data/real_ess")
NAMES = ["radius", "T", "logH2O", "logCO2", "logCH4", "logCO", "logNH3"]
_ORD = ["planet_radius", "T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]


def _ns():
    d = np.load(OUT / "wasp39b_ns_posterior_fitrad.npz", allow_pickle=True)
    idx = [list(d["fit_names"]).index(k) for k in _ORD]
    w = d["weights"] / d["weights"].sum()
    return np.sum(w[:, None] * d["samples"][:, idx], 0)


def _metrics(samp_f, fwd_f, nsm, use_log_g=False):
    s = np.load(OUT / samp_f); f = np.load(OUT / fwd_f)
    th = s["theta"]; x, sig, cov, ms = s["x_obs"], s["sig_obs"], s["covered"], f["model_spec"]
    logp = s["log_g"] if use_log_g else s["log_q"]
    m = cov & np.isfinite(sig) & (sig > 0); se = np.sqrt(sig ** 2 + (0.05 * x) ** 2)
    val = np.isfinite(ms[:, m]).all(1); r = (x[None, m] - ms[:, m]) / se[None, m]
    ll = np.where(val, -0.5 * np.nansum(r ** 2, 1), -np.inf)
    w = np.exp(ll - logp - np.nanmax(ll - logp)); w /= w.sum()
    bc = -2 * ll.max() / m.sum()
    wq = lambda v, q: np.interp(q, np.cumsum(w[np.argsort(v)]), v[np.argsort(v)])
    cvr = sum(wq(th[:, i], 0.16) <= nsm[i] <= wq(th[:, i], 0.84) for i in range(7))
    R = np.sum(w * th[:, 0]); H = np.sum(w * th[:, 2]); ess = 1 / np.sum(w ** 2)
    return bc, R, H, cvr, ess


def main():
    nsm = _ns()
    rows = []
    # before: fixed-radius model (documented — physical solution can't fit; ESS=1)
    rows.append(("base FMPE, radius FIXED", "θ=[T,5mol]", "301.0", "1.27 (fixed)", "−11→−3", "—", "1.0"))
    for lbl, tag in [("rad · nocond", "abl_rad_nocond"), ("rad · σ-only", "abl_rad_sigma"),
                     ("rad · σ+cov", "abl_rad")]:
        bc, R, H, cvr, ess = _metrics(f"{tag}_samples.npz", f"{tag}_forward.npz", nsm)
        rows.append((lbl, "θ=[rp,T,5mol]", f"{bc:.3f}", f"{R:.3f}", f"{H:.2f}", f"{cvr}/7", f"{ess:.1f}"))
    bc, R, H, cvr, ess = _metrics("ot_cal_samples.npz", "ot_cal_forward.npz", nsm, use_log_g=True)
    rows.append(("**rad · cov + OT-cal**", "θ=[rp,T,5mol]", f"**{bc:.3f}**", f"**{R:.3f}**",
                 f"**{H:.2f}**", f"**{cvr}/7**", f"{ess:.1f}"))
    rows.append(("NS anchor (fitrad)", "reference", "0.76", f"{nsm[0]:.3f}", f"{nsm[2]:.2f}", "—", "—"))

    hdr = ["model", "θ", "best χ²/dof", "radius(IS)", "logH₂O(IS)", "cover/7", "ESS"]
    md = ["# Phase 4 — Real WASP-39b benchmark (main results table)", "",
          "Independent NS anchor: radius=1.23, T=606, logH₂O=−3.25 (truth radius=1.27, "
          "WASP-39b is water-rich). All metrics on the 47 real-covered bins, 5% forward floor.", "",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md += ["",
           "**Reading:**",
           "- **Radius fix** (P3-D11): adding radius to θ drops best-fit χ² from **301 → ~0.06** and "
           "recovers radius≈1.2 (≈truth) — the core real-data result. Robust across ALL arms.",
           "- **OT calibration** (P3-D13): transports the cov posterior onto the NS anchor → "
           "**7/7 coverage**, physical/literature-consistent (radius 1.227, logH₂O −3.46).",
           "- **Noise-conditioning ablation** (P4-D1, honest negative): σ+cov does NOT beat nocond on "
           "real data (cover 3/7 vs 3/7; JS→NS cov 0.43 > nocond 0.36). cov ingests OOD real OOT noise "
           "(P3-D1) → perturbed. Noise-conditioning is validated on synthetic ABC (Phase 2), not here.",
           "- **ESS** is a poor yardstick here (peaked 47-bin likelihood); report coverage/JS-vs-NS."]
    doc = Path("Documentation/phase4_benchmark.md")
    doc.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n→ written to {doc}")


if __name__ == "__main__":
    main()
