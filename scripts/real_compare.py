"""
WI-5 (c) — fair real-data metric: compare MIRAGE's posterior on WASP-39b to the
nested-sampling ANCHOR (taurex_retrieve.py output), in the shared 6-param space.

Raw IS-ESS is destroyed by real JWST's tiny σ × the WI-1 5% forward systematic
(P3-D5), so it can't discriminate posterior quality. Comparing MIRAGE directly
to the trusted NS posterior (per-param intervals + 1-D Jensen–Shannon) is the
fair measure — and it's the natural "before calibration" baseline for RoPE-OT.

    python scripts/real_compare.py            # uses last real_ess_samples.npz (arm)
"""

from pathlib import Path
import numpy as np

OUT = Path("data/real_ess")
NAMES = ["T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]


def wquantile(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    c /= w.sum()
    return np.interp(q, c, x)


def js_1d(xa, wa, xb, wb, bins=40):
    lo = min(xa.min(), xb.min()); hi = max(xa.max(), xb.max())
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(xa, edges, weights=wa, density=True)
    pb, _ = np.histogram(xb, edges, weights=wb, density=True)
    pa = pa / pa.sum() + 1e-12; pb = pb / pb.sum() + 1e-12
    m = 0.5 * (pa + pb)
    kl = lambda p, q: np.sum(p * np.log(p / q))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)      # in nats, 0=identical, ln2=disjoint


def main():
    ns = np.load(OUT / "wasp39b_ns_posterior.npz", allow_pickle=True)
    mi = np.load(OUT / "real_ess_samples.npz")
    # fit_names == [T, log_H2O, log_CO2, log_CH4, log_CO, log_NH3] (θ order, log10)
    ns_th, ns_w = ns["samples"], ns["weights"] / ns["weights"].sum()
    mi_th = mi["theta"]; mi_w = np.full(len(mi_th), 1.0 / len(mi_th))

    print("Fair real-data comparison — MIRAGE posterior vs NS anchor (WASP-39b)\n")
    print(f"  {'param':<9} {'NS median [16,84]':>26} {'MIRAGE median [16,84]':>26} {'JS':>6}")
    print("  " + "-" * 70)
    jss = []
    for i, nm in enumerate(NAMES):
        a, b, c = wquantile(ns_th[:, i], ns_w, [0.16, 0.5, 0.84])
        d, e, f = wquantile(mi_th[:, i], mi_w, [0.16, 0.5, 0.84])
        js = js_1d(ns_th[:, i], ns_w, mi_th[:, i], mi_w)
        jss.append(js)
        print(f"  {nm:<9} {b:>10.2f} [{a:.2f},{c:.2f}] {e:>12.2f} [{d:.2f},{f:.2f}] {js:>6.2f}")
    print("  " + "-" * 70)
    print(f"  mean 1-D JS = {np.mean(jss):.3f} nats  (0 = identical, {np.log(2):.2f} = disjoint)")
    print("  → this is the fair 'distance to the trusted answer'; RoPE-OT should shrink it.")


if __name__ == "__main__":
    main()
