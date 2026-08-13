"""
Phase 5 multi-target figure — one radius-inference method across three regimes.
Top row: real JWST transmission spectrum + MIRAGE posterior-predictive band.
Bottom row: MIRAGE radius posterior vs the independent NS anchor and the truth.
Spans hot Saturn (PRISM) -> hot Saturn (SOSS, self-reduced) -> cold sub-Neptune
(NIRISS), a ~6x radius range. All three use the SAME nocond radius model.

    conda run -n mirage python scripts/fig_multitarget.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("data/real_ess")
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
NS_ORDER = ["planet_radius", "T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]

# (label, subtitle, samples/forward prefix, NS anchor npz, truth radius RJ, colour)
PLANETS = [
    ("WASP-39b", "hot Saturn  ·  NIRSpec PRISM", "abl_rad_nocond",
     "wasp39b_ns_posterior_fitrad.npz", 1.279, "#c1440e"),
    ("WASP-96b", "hot Saturn  ·  NIRISS SOSS (self-reduced)", "wasp96",
     "wasp39b_ns_posterior_wasp96.npz", 1.20, "#1f6feb"),
    ("K2-18b", "cold sub-Neptune  ·  NIRISS", "k218cov",
     "wasp39b_ns_posterior_k218.npz", 0.2352, "#2ea043"),
]


def is_weights(prefix):
    s = np.load(OUT / f"{prefix}_samples.npz"); f = np.load(OUT / f"{prefix}_forward.npz")
    th, lq = s["theta"], s["log_q"]
    x, sig, cov, wl = s["x_obs"], s["sig_obs"], s["covered"], s["wlen"]
    ms = f["model_spec"]
    m = cov & np.isfinite(sig) & (sig > 0)
    se = np.sqrt(sig ** 2 + (0.05 * x) ** 2)
    val = np.isfinite(ms[:, m]).all(1)
    r = (x[None, m] - ms[:, m]) / se[None, m]
    ll = np.where(val, -0.5 * np.nansum(r ** 2, 1), -np.inf)
    w = np.exp(ll - lq - np.nanmax(ll - lq)); w /= w.sum()
    return th, w, x, sig, m, wl, ms


def wquant(v, w, q):
    i = np.argsort(v); c = np.cumsum(w[i])
    return np.interp(q, c, v[i])


def ns_radius(npz):
    d = np.load(OUT / npz, allow_pickle=True)
    fn = [str(x) for x in d["fit_names"]]
    wn = d["weights"] / d["weights"].sum()
    rv = d["samples"][:, fn.index("planet_radius")]
    return np.sum(wn * rv), wquant(rv, wn, 0.16), wquant(rv, wn, 0.84)


fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
for c, (name, sub, prefix, nsnpz, truth, col) in enumerate(PLANETS):
    th, w, x, sig, m, wl, ms = is_weights(prefix)
    wlc = wl[m]; o = np.argsort(wlc)
    # ---- top: spectrum + posterior-predictive band ----
    ax = axes[0, c]
    ax.errorbar(wlc[o], x[m][o] * 1e2, yerr=sig[m][o] * 1e2, fmt="o", ms=3, lw=0.8,
                color="0.25", ecolor="0.6", capsize=0, label="JWST (real)", zorder=3)
    lo = np.array([wquant(ms[:, m][:, k], w, 0.16) for k in range(m.sum())])
    hi = np.array([wquant(ms[:, m][:, k], w, 0.84) for k in range(m.sum())])
    md = np.array([wquant(ms[:, m][:, k], w, 0.50) for k in range(m.sum())])
    ax.fill_between(wlc[o], lo[o] * 1e2, hi[o] * 1e2, color=col, alpha=0.30, lw=0,
                    label="MIRAGE 68%", zorder=2)
    ax.plot(wlc[o], md[o] * 1e2, color=col, lw=1.2, zorder=2)
    ax.set_title(f"{name}\n{sub}", fontsize=10.5)
    ax.set_xlabel("wavelength [µm]", fontsize=9)
    if c == 0:
        ax.set_ylabel("transit depth  (Rp/R$_\\star$)$^2$  [%]", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.tick_params(labelsize=8)

    # ---- bottom: radius posterior vs NS vs truth ----
    ax = axes[1, c]
    rp = th[:, 0]
    lo_r, hi_r = wquant(rp, w, 0.005), wquant(rp, w, 0.995)
    bins = np.linspace(lo_r, hi_r, 60)
    ax.hist(rp, bins=bins, weights=w, color=col, alpha=0.55, density=True, label="MIRAGE posterior")
    r16, r84 = wquant(rp, w, 0.16), wquant(rp, w, 0.84)
    ax.axvspan(r16, r84, color=col, alpha=0.15)
    nsm, ns16, ns84 = ns_radius(nsnpz)
    ax.axvline(nsm, color="k", ls="--", lw=1.4, label=f"NS anchor ({nsm:.3f})")
    ax.axvline(truth, color="crimson", ls="-", lw=1.6, label=f"literature ({truth:.3f})")
    # x-range must bracket MIRAGE posterior + NS + literature radius with margin
    xlo = min(lo_r, nsm, truth); xhi = max(hi_r, nsm, truth); pad = 0.12 * (xhi - xlo)
    ax.set_xlim(xlo - pad, xhi + pad)
    ax.set_xlabel("planet radius  [R$_{\\rm Jup}$]", fontsize=9)
    if c == 0:
        ax.set_ylabel("posterior density", fontsize=9)
    ism = np.sum(w * rp)
    ax.set_title(f"radius: MIRAGE {ism:.3f}  ·  NS {nsm:.3f}  ·  lit {truth:.2f}", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=8)

fig.suptitle("One radius-inference method, three regimes — real JWST retrievals "
             "(hot Saturn → cold sub-Neptune, PRISM → SOSS, published → self-reduced)",
             fontsize=11.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = FIG / "fig6_multitarget.png"
fig.savefig(out, dpi=170); plt.close(fig)
print(f"  wrote {out}")

# quick console summary
print("\n  radius recovery (IS-reweighted MIRAGE vs truth):")
for name, sub, prefix, nsnpz, truth, col in PLANETS:
    th, w, *_ = is_weights(prefix)
    print(f"    {name:<9} MIRAGE {np.sum(w*th[:,0]):.3f}   truth {truth:.3f}   "
          f"NS {ns_radius(nsnpz)[0]:.3f}")
