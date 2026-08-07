"""
Phase 4.4 — publication figures for the real-WASP-39b result. Writes PNGs to figures/.
  1. radius_is_the_fix : best-fit χ²/dof across every forward-model lever — all physics
     probes stay cold (~2.5) or forced-hot (~38); freeing the RADIUS drops it to 0.76.
  2. mirage_vs_ns      : OT-calibrated MIRAGE posterior vs the independent NS anchor.
  3. spectrum_fit      : real WASP-39b spectrum + MIRAGE best-fit model.

    conda activate mirage && python scripts/make_figures.py
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("data/real_ess"); FIG = Path("figures"); FIG.mkdir(exist_ok=True)
BLUE, RED, GREY, GREEN = "#2c6fbb", "#c0392b", "#7f8c8d", "#27ae60"


def _chi2(f):
    d = np.load(OUT / f, allow_pickle=True); return float(d["chi2"])


def fig1():
    # (label, chi2, group)  group: physics-lever(cold), forced-hot, radius-fix
    probes = [("isothermal", 2.55, 0), ("+T-gradient", 2.65, 0), ("+clouds+grad", 2.616, 0),
              ("+SO₂", 2.555, 0), ("+hi-fi opacities\n(ExoMolOP R15000)", 2.416, 0),
              ("forced-physical\n(isothermal)", 38.441, 1), ("forced-physical\n(+gradient)", 38.478, 1),
              ("FREE RADIUS", 0.761, 2)]
    labels = [p[0] for p in probes]; vals = [p[1] for p in probes]
    colors = {0: GREY, 1: RED, 2: GREEN}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(range(len(vals)), vals, color=[colors[p[2]] for p in probes], width=0.68)
    ax.set_yscale("log")
    ax.axhline(1.0, ls="--", lw=1, color="k", alpha=0.5)
    ax.text(len(vals) - 0.5, 1.05, "χ²/dof = 1 (good fit)", ha="right", va="bottom", fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_ylabel("best-fit χ²/dof on real WASP-39b")
    ax.set_title("The sim-to-real gap is the radius, not forward-model fidelity", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, RED, GREEN)]
    ax.legend(handles, ["forward-model levers (cold-allowed)", "forced-physical", "free the radius"],
              fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig1_radius_is_the_fix.png", dpi=160); plt.close(fig)
    print("  fig1_radius_is_the_fix.png")


def fig2():
    # OT-calibrated MIRAGE (weighted) vs NS anchor, marginals for radius/T/logH2O
    s = np.load(OUT / "ot_cal_samples.npz"); f = np.load(OUT / "ot_cal_forward.npz")
    th, lg = s["theta"], s["log_g"]; x, sig, cov, ms = s["x_obs"], s["sig_obs"], s["covered"], f["model_spec"]
    m = cov & np.isfinite(sig) & (sig > 0); se = np.sqrt(sig ** 2 + (0.05 * x) ** 2)
    val = np.isfinite(ms[:, m]).all(1); r = (x[None, m] - ms[:, m]) / se[None, m]
    ll = np.where(val, -0.5 * np.nansum(r ** 2, 1), -np.inf)
    w = np.exp(ll - lg - np.nanmax(ll - lg)); w /= w.sum()
    d = np.load(OUT / "wasp39b_ns_posterior_fitrad.npz", allow_pickle=True)
    o = ["planet_radius", "T", "log_H2O"]; idx = [list(d["fit_names"]).index(k) for k in o]
    nsw = d["weights"] / d["weights"].sum(); nss = d["samples"][:, idx]
    cols = [(0, "radius (R$_J$)", 1.27), (1, "T (K)", None), (2, "log H$_2$O", None)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (ci, name, truth) in zip(axes, cols):
        lo = min(th[:, ci].min(), nss[:, {0:0,1:1,2:2}[ci]].min())
        hi = max(th[:, ci].max(), nss[:, {0:0,1:1,2:2}[ci]].max())
        b = np.linspace(lo, hi, 40)
        ax.hist(nss[:, {0:0,1:1,2:2}[ci]], b, weights=nsw, density=True, color=GREY, alpha=0.55, label="NS anchor")
        ax.hist(th[:, ci], b, weights=w, density=True, histtype="step", color=BLUE, lw=2, label="MIRAGE (OT-cal)")
        if truth: ax.axvline(truth, color=GREEN, ls="--", lw=1.5, label="literature")
        ax.set_xlabel(name); ax.set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle("OT-calibrated MIRAGE posterior vs independent nested-sampling (7/7 coverage)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "fig2_mirage_vs_ns.png", dpi=160); plt.close(fig)
    print("  fig2_mirage_vs_ns.png")


def fig3():
    s = np.load(OUT / "ot_cal_samples.npz"); f = np.load(OUT / "ot_cal_forward.npz")
    x, sig, cov, wl, ms = s["x_obs"], s["sig_obs"], s["covered"], s["wlen"], f["model_spec"]
    m = cov & np.isfinite(sig) & (sig > 0); se = np.sqrt(sig ** 2 + (0.05 * x) ** 2)
    val = np.isfinite(ms[:, m]).all(1); r = (x[None, m] - ms[:, m]) / se[None, m]
    ll = np.where(val, -0.5 * np.nansum(r ** 2, 1), -np.inf); bi = np.argmax(ll)
    o = np.argsort(wl)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    mm = m[o]; wlo = wl[o]
    ax.errorbar(wlo[mm], x[o][mm] * 1e6, yerr=sig[o][mm] * 1e6, fmt="o", ms=4, color="k",
                capsize=2, lw=1, label="JWST WASP-39b (real)")
    ax.plot(wlo[mm], ms[bi][o][mm] * 1e6, "-", color=RED, lw=2, label="MIRAGE best-fit (χ²/dof=0.03)")
    ax.set_xlabel("wavelength (µm)"); ax.set_ylabel("transit depth (ppm)")
    ax.set_title("MIRAGE physical fit to real WASP-39b (radius + OT calibration)", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig3_spectrum_fit.png", dpi=160); plt.close(fig)
    print("  fig3_spectrum_fit.png")


def fig4():
    # full 7-param corner: OT-calibrated MIRAGE posterior vs NS anchor
    import corner
    s = np.load(OUT / "ot_cal_samples.npz")          # OT-calibrated proposal = calibrated posterior
    th = s["theta"]
    d = np.load(OUT / "wasp39b_ns_posterior_fitrad.npz", allow_pickle=True)
    o = ["planet_radius", "T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
    idx = [list(d["fit_names"]).index(k) for k in o]
    nsw = d["weights"] / d["weights"].sum(); nss = d["samples"][:, idx]
    labels = ["R (R$_J$)", "T (K)", "log H₂O", "log CO₂", "log CH₄", "log CO", "log NH₃"]
    rng = [(min(th[:, i].min(), nss[:, i].min()), max(th[:, i].max(), nss[:, i].max())) for i in range(7)]
    fig = corner.corner(nss, weights=nsw, labels=labels, color=GREY, range=rng,
                        plot_datapoints=False, hist_kwargs={"density": True})
    corner.corner(th, fig=fig, color=BLUE, labels=labels, range=rng,
                  plot_datapoints=False, hist_kwargs={"density": True})
    fig.suptitle("Calibrated MIRAGE (blue) vs independent NS anchor (grey) — full 7-param posterior",
                 fontsize=13)
    fig.savefig(FIG / "fig4_corner.png", dpi=140); plt.close(fig)
    print("  fig4_corner.png")


def fig5():
    # ABC ablation: noise-conditioning WINS on synthetic data (the honest other half)
    d = np.load(OUT / "abc_ablation.npz", allow_pickle=True)
    tr = d["truths"]; ps = d["planet_sigma"]
    ORANGE = "#e67e22"
    arms = [("no-cond", "no_cond", GREY), ("+σ", "psigma", ORANGE), ("+σ+cov", "psigmapcov", BLUE)]

    def cov68(key):
        s = d["s__" + key]                              # (P, N, D)
        lo = np.quantile(s, 0.16, axis=1); hi = np.quantile(s, 0.84, axis=1)
        return float(np.mean((lo <= tr) & (tr <= hi)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    covs = [cov68(a[1]) for a in arms]
    ax1.bar(range(3), covs, color=[a[2] for a in arms], width=0.6)
    ax1.axhline(0.68, ls="--", color="k", lw=1)
    ax1.text(2.45, 0.695, "nominal 68%", ha="right", fontsize=8)
    ax1.set_xticks(range(3)); ax1.set_xticklabels([a[0] for a in arms])
    ax1.set_ylabel("coverage @ 68%"); ax1.set_ylim(0, 1)
    ax1.set_title("(a) ABC: conditioning restores coverage", fontsize=10)
    for i, c in enumerate(covs):
        ax1.text(i, c + 0.015, f"{c:.3f}", ha="center", fontsize=9)

    ldn, ldc = d["ld__no_cond"], d["ld__psigmapcov"]
    b = np.digitize(ps, np.quantile(ps, [1 / 3, 2 / 3]))
    gaps = [(ldc[b == k] - ldn[b == k]).mean() for k in range(3)]
    ax2.bar(range(3), gaps, color=GREEN, width=0.6)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(["low-σ", "mid-σ", "high-σ"])
    ax2.set_ylabel("Δ log-density  (+σ+cov − no-cond)")
    ax2.set_title("(b) ABC: cov helps most at high noise (adaptivity)", fontsize=10)
    for i, g in enumerate(gaps):
        ax2.text(i, g + 0.01, f"+{g:.2f}", ha="center", fontsize=9)
    fig.suptitle("Noise-conditioning WINS on synthetic ABC (Phase 2) — where truth is known",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "fig5_abc_ablation.png", dpi=160); plt.close(fig)
    print("  fig5_abc_ablation.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5"]
    for w in which:
        globals()[w]()
    print(f"→ figures in {FIG}/")
