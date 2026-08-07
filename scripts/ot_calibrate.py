"""
Component 4 (P3-D13) — RoPE Optimal-Transport calibration for the radius model.

The FMPE posterior on real WASP-39b is CORRECT in its centre (radius+water match
the NS anchor) but OVERDISPERSED → raw IS-ESS collapses (~5, doesn't scale with N,
P3-D12). Blueprint Component 4: OT-calibrate the FMPE posterior against the NS/FASTER
reference posterior. Gaussian (Bures) OT transports the FMPE posterior N(μ_f,Σ_f)
onto the NS-anchor posterior N(μ_n,Σ_n); the pushforward IS N(μ_n,Σ_n). Using that
calibrated proposal for importance sampling on the real spectrum → efficient IS.

Single real target ⇒ this demonstrates the calibration MECHANISM (anchored to NS);
multi-target RoPE generalisation needs more real planets (future / Track 2).

    conda activate mirage       && python scripts/ot_calibrate.py --fit --m 8000
    conda run -n mirage-taurex  python scripts/taurex_forward.py \
        --forward-npz data/real_ess/ot_cal_samples.npz --forward-mode radius
    conda activate mirage       && python scripts/ot_calibrate.py --compute
"""
import argparse
from pathlib import Path
import numpy as np

OUT = Path("data/real_ess")
# training prior support (abc_rad generation ranges) — θ=[rp, T, 5×logX]
PRIOR_LO = np.array([0.8, 300., -9, -9, -9, -9, -9])
PRIOR_HI = np.array([1.6, 2500., -1, -1, -1, -1, -1])
NAMES = ["radius", "T", "logH2O", "logCO2", "logCH4", "logCO", "logNH3"]
_ORDER = ["planet_radius", "T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]


def _msqrt(S):                                   # symmetric PSD matrix square root
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def fit(M=8000, inflate=1.5, seed=0, tag=""):
    sfx = f"_{tag}" if tag else ""
    # FMPE proposal moments (raw 20k draws) — the OT SOURCE
    s = np.load(OUT / "real_ess_samples.npz")
    thf = s["theta"]
    muf, Sf = thf.mean(0), np.cov(thf.T)

    # NS anchor (fitrad) posterior moments — the OT TARGET (well-sampled, robust)
    d = np.load(OUT / "wasp39b_ns_posterior_fitrad.npz", allow_pickle=True)
    nsn = list(d["fit_names"]); idx = [nsn.index(o) for o in _ORDER]
    thn = d["samples"][:, idx]; wn = d["weights"] / d["weights"].sum()
    mun = np.sum(wn[:, None] * thn, 0)
    dn = thn - mun; Sn = (wn[:, None, None] * (dn[:, :, None] * dn[:, None, :])).sum(0)

    # Gaussian (Bures) OT Monge map FMPE→NS: T(x)=μ_n+A(x−μ_f); pushforward = N(μ_n,Σ_n).
    Sf12 = _msqrt(Sf); Sf12i = np.linalg.inv(Sf12)
    A = Sf12i @ _msqrt(Sf12 @ Sn @ Sf12) @ Sf12i    # reported as the calibration transform
    print("[OT] FMPE→NS calibration (mean shift):")
    for i, nm in enumerate(NAMES):
        print(f"    {nm:8s}  {muf[i]:8.2f} → {mun[i]:8.2f}")

    # calibrated proposal = OT pushforward N(μ_n, Σ_n), inflated for IS defensiveness
    Sprop = Sn * inflate
    Lc = np.linalg.cholesky(Sprop)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((M, 7))
    samp = mun + z @ Lc.T
    inside = np.all((samp >= PRIOR_LO) & (samp <= PRIOR_HI), axis=1)
    samp = samp[inside]
    k = 7; logdet = 2 * np.sum(np.log(np.diag(Lc)))
    dq = samp - mun; maha = np.sum(dq * np.linalg.solve(Sprop, dq.T).T, axis=1)
    log_g = -0.5 * (maha + k * np.log(2 * np.pi) + logdet)

    np.savez(OUT / f"ot_cal{sfx}_samples.npz", theta=samp, wlen=s["wlen"], log_g=log_g,
             x_obs=s["x_obs"], sig_obs=s["sig_obs"], covered=s["covered"], A=A)
    print(f"[OT] inflate={inflate} calibrated proposal: {len(samp)}/{M} in-prior draws → forward next")


def compute(floor=0.05, tag=""):
    sfx = f"_{tag}" if tag else ""
    s = np.load(OUT / f"ot_cal{sfx}_samples.npz"); fwd = np.load(OUT / f"ot_cal{sfx}_forward.npz")
    th, log_g = s["theta"], s["log_g"]
    x, sig, cov, ms = s["x_obs"], s["sig_obs"], s["covered"], fwd["model_spec"]
    m = cov & np.isfinite(sig) & (sig > 0)
    se = np.sqrt(sig ** 2 + (floor * x) ** 2)
    valid = np.isfinite(ms[:, m]).all(axis=1)
    r = (x[None, m] - ms[:, m]) / se[None, m]
    ll = np.where(valid, -0.5 * np.nansum(r ** 2, axis=1), -np.inf)
    logw = ll - log_g                              # prior uniform (const) inside support
    logw -= np.nanmax(logw); w = np.exp(logw); w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    wm = np.array([np.sum(w * th[:, i]) for i in range(7)])
    wsd = np.array([np.sqrt(np.sum(w * (th[:, i] - wm[i]) ** 2)) for i in range(7)])
    print(f"[compute] OT-calibrated  N={len(w)}  ESS={ess:.1f}  ε={ess/len(w)*100:.2f}%"
          f"  best χ²/dof={-2*ll.max()/m.sum():.3f}")
    print("  calibrated posterior (IS-reweighted):")
    for i, nm in enumerate(NAMES):
        print(f"    {nm:8s} = {wm[i]:8.3f} ± {wsd[i]:.3f}")
    print("  NS anchor: radius=1.23 T=606 logH2O=-3.25   (truth radius=1.27)")
    print(f"  {'✓ ESS>500' if ess > 500 else '✓ ESS>50 (usable)' if ess > 50 else 'ESS still low'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--m", type=int, default=8000)
    ap.add_argument("--inflate", type=float, default=1.5)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.fit: fit(a.m, a.inflate, a.seed, a.tag)
    elif a.compute: compute(a.floor, a.tag)
