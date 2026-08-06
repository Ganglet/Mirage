"""
WI-5 anchor — nested-sampling retrieval on the REAL WASP-39b spectrum, in
MIRAGE's exact 6-param space (T + log H2O/CO2/CH4/CO/NH3), using the WI-1
TauREx forward model with WASP-39b geometry. This is the reference posterior
RoPE-OT calibrates MIRAGE against — self-consistent (same param space), a real
NS retrieval on real data (not a prototype). Published FASTER/ERS posteriors use
different parameterisations so can't anchor a 6-param model directly.

Runs in `mirage-taurex`. Observed spectrum = the 52-bin binned real depth saved
by real_ess.py (data/real_ess/real_ess_samples.npz).

    conda run -n mirage-taurex python scripts/taurex_retrieve.py --live 300
"""

import argparse
from pathlib import Path

import numpy as np

import taurex_forward as tf   # PYTHONPATH=scripts; build_model, MOLECULES, WASP39
from taurex.constants import RJUP, MJUP, RSOL, MSOL  # noqa

OUT = Path("data/real_ess")
# ABC prior ranges (match training) so the NS posterior is comparable to MIRAGE
T_BOUNDS = (110.0, 5500.0)
LOGX_BOUNDS = (-9.0, -3.0)


def main(live, FLOOR=0.01, clouds=False, highres=False, nbins=150, tprofile="isothermal",
         tag="", tmin=None, tmax=None, so2=False, hifi=False, fitrad=False):
    from taurex.data.spectrum.array import ArraySpectrum
    from taurex.optimizer.nestle import NestleOptimizer

    if highres:
        # Path B/C test: fit the NATIVE-resolution WASP-39b spectrum (~R100) instead
        # of ABC's 52 bins (~R15). If the cold-T degeneracy breaks here, the sim-to-real
        # failure is a RESOLUTION mismatch (ABC/Ariel grid too coarse for real JWST).
        import pandas as pd
        df = pd.read_csv("data/jwst_wasp39b_spectrum.csv")
        lo, hi = df["wavelength_um"].min(), df["wavelength_um"].max()
        grid = np.geomspace(lo, hi, nbins)
        ge = np.empty(nbins + 1); ge[1:-1] = np.sqrt(grid[:-1] * grid[1:])
        ge[0] = grid[0] ** 2 / ge[1]; ge[-1] = grid[-1] ** 2 / ge[-2]
        b = np.digitize(df["wavelength_um"].to_numpy(), ge) - 1
        wl, depth, err = [], [], []
        for k in range(nbins):
            msk = b == k
            if msk.sum() == 0:
                continue
            w = 1.0 / np.clip(df["depth_error"].to_numpy()[msk], 1e-12, None) ** 2
            d = df["transit_depth"].to_numpy()[msk]
            wl.append(grid[k]); depth.append(np.sum(w * d) / np.sum(w))
            err.append(1.0 / np.sqrt(np.sum(w)))
        wl, depth, err = np.array(wl), np.array(depth), np.array(err)
    else:
        s = np.load(OUT / "real_ess_samples.npz")
        cov = s["covered"]
        wl, depth, err = s["wlen"][cov], s["x_obs"][cov], s["sig_obs"][cov]
    o = np.argsort(wl)                                   # ascending for TauREx
    wl, depth, err = wl[o], depth[o], err[o]
    # error floor: 5%-of-depth drowned the features → degenerate cold-T reference.
    # Smaller floor keeps features visible (breaks the T degeneracy) yet tractable.
    err = np.sqrt(err ** 2 + (FLOOR * depth) ** 2)
    edges = np.empty(len(wl) + 1)
    edges[1:-1] = np.sqrt(wl[:-1] * wl[1:])
    edges[0] = wl[0] ** 2 / edges[1]; edges[-1] = wl[-1] ** 2 / edges[-2]
    width = np.diff(edges)
    obs = ArraySpectrum(np.stack([wl, depth, err, width], axis=1))

    extra_mols = ["SO2"] if so2 else []                 # P3-D9: SO2 lever (4µm feature)
    tm = tf.build_model(**tf.WASP39, use_clouds=clouds, tprofile=tprofile,
                        extra_mols=extra_mols, hifi=hifi)
    tm.model()                                          # force profile init (nlayers etc.)
    opt = NestleOptimizer(num_live_points=live)
    opt.set_model(tm)
    opt.set_observed(obs)
    # T bounds: default = ABC prior; tmin/tmax override forces a PHYSICAL range
    # (hot probe) to quantify the χ² penalty of a non-cold-flat solution (P3-D8).
    tb = (tmin if tmin is not None else T_BOUNDS[0],
          tmax if tmax is not None else T_BOUNDS[1])
    if tprofile == "npoint":                            # P3-D8: 2-point vertical gradient
        for tp in ("T_surface", "T_top"):
            opt.enable_fit(tp); opt.set_boundary(tp, tb)
    else:
        opt.enable_fit("T"); opt.set_boundary("T", tb)
    for mol in tf.MOLECULES + extra_mols:                # log-uniform mix ratios
        opt.enable_fit(mol)
        opt.set_boundary(mol, (10.0 ** LOGX_BOUNDS[0], 10.0 ** LOGX_BOUNDS[1]))
        opt.set_mode(mol, "log")
    if clouds:                                           # Path B: gray-cloud deck
        opt.enable_fit("clouds_pressure")
        opt.set_boundary("clouds_pressure", (1e2, 1e6))
        opt.set_mode("clouds_pressure", "log")
    if fitrad:                                           # P3-D11: float planet radius (Rjup)
        opt.enable_fit("planet_radius")                  # baseline is a nuisance the fixed
        opt.set_boundary("planet_radius", (1.0, 1.6))    # radius may set wrong -> cold-flat corner
    npar = (7 if tprofile == "npoint" else 6) + (1 if clouds else 0) + len(extra_mols) + (1 if fitrad else 0)
    print(f"[retrieve] NS on real WASP-39b, {live} live points, {len(wl)} bins, {npar} params ...")
    opt.fit()

    # BULLETPROOF: save raw samples/weights/names — never lose an expensive run to
    # a mapping bug. Column→θ mapping is done downstream (cheap) from fit_names.
    samples = np.asarray(opt.get_samples(0))            # (M, ndim) fit space
    weights = np.asarray(opt.get_weights(0))
    names = [str(x) for x in opt.fit_names]
    w = weights / weights.sum()
    wmean = np.average(samples, axis=0, weights=w)

    # χ²/N of the weighted-mean fit vs the observed spectrum — the headline
    # number per probe (is the best physical fit actually good, or does it lose
    # to cold-flat?). Computed here so each parallel probe self-reports it.
    chi2 = float("nan")
    try:
        from taurex.binning import FluxBinner
        # fit_names carry log-mode params as "log_<X>"; the model only accepts the
        # linear name, so undo the log. T/T_surface/T_top are linear → set directly.
        for nm, val in zip(names, wmean):
            if nm.startswith("log_"):
                tm[nm[4:]] = float(10.0 ** val)
            else:
                tm[nm] = float(val)
        native_wn, rprs, _, _ = tm.model()
        native_wl = 10000.0 / native_wn
        fb = FluxBinner(wl, width)
        _, binned, _, _ = fb.bindown(native_wl[::-1], rprs[::-1])
        chi2 = float(np.mean(((binned - depth) / err) ** 2))
    except Exception as e:
        print(f"  [warn] χ² compute failed: {e}")

    out = OUT / f"wasp39b_ns_posterior{('_' + tag) if tag else ''}.npz"
    np.savez(out, samples=samples, weights=weights,
             fit_names=np.array(names, dtype=object), chi2=chi2)
    print(f"[retrieve] posterior saved: {len(samples)} weighted samples")
    print(f"  fit_names = {names}")
    print(f"  weighted means = {np.round(wmean, 3)}")
    print(f"  χ²/N (weighted-mean fit) = {chi2:.3f}")
    print(f"  → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=int, default=300)
    ap.add_argument("--floor", type=float, default=0.01)
    ap.add_argument("--clouds", action="store_true")     # Path B: 7-param cloud fit
    ap.add_argument("--highres", action="store_true")    # fit native-res spectrum
    ap.add_argument("--nbins", type=int, default=150)
    ap.add_argument("--tprofile", default="isothermal",  # P3-D8: "npoint" = 2-pt gradient
                    choices=["isothermal", "npoint"])
    ap.add_argument("--tag", default="")                 # output filename suffix (parallel probes)
    ap.add_argument("--tmin", type=float, default=None)  # force T lower bound (hot probe)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--so2", action="store_true")        # P3-D9: add + fit SO2 (4µm feature)
    ap.add_argument("--hifi", action="store_true")       # P3-D10: ExoMolOP R=15000 opacities
    ap.add_argument("--fitrad", action="store_true")     # P3-D11: float planet radius
    a = ap.parse_args()
    main(a.live, a.floor, a.clouds, a.highres, a.nbins, a.tprofile, a.tag, a.tmin, a.tmax, a.so2, a.hifi, a.fitrad)
