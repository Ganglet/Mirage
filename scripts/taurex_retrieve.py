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


def main(live, FLOOR=0.01, clouds=False, highres=False, nbins=150):
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

    tm = tf.build_model(**tf.WASP39, use_clouds=clouds)
    tm.model()                                          # force profile init (nlayers etc.)
    opt = NestleOptimizer(num_live_points=live)
    opt.set_model(tm)
    opt.set_observed(obs)
    opt.enable_fit("T"); opt.set_boundary("T", T_BOUNDS)
    for mol in tf.MOLECULES:                             # log-uniform mix ratios
        opt.enable_fit(mol)
        opt.set_boundary(mol, (10.0 ** LOGX_BOUNDS[0], 10.0 ** LOGX_BOUNDS[1]))
        opt.set_mode(mol, "log")
    if clouds:                                           # Path B: gray-cloud deck
        opt.enable_fit("clouds_pressure")
        opt.set_boundary("clouds_pressure", (1e2, 1e6))
        opt.set_mode("clouds_pressure", "log")
    npar = 7 if clouds else 6
    print(f"[retrieve] NS on real WASP-39b, {live} live points, {len(wl)} bins, {npar} params ...")
    opt.fit()

    # BULLETPROOF: save raw samples/weights/names — never lose an expensive run to
    # a mapping bug. Column→θ mapping is done downstream (cheap) from fit_names.
    samples = np.asarray(opt.get_samples(0))            # (M, ndim) fit space
    weights = np.asarray(opt.get_weights(0))
    names = [str(x) for x in opt.fit_names]
    np.savez(OUT / "wasp39b_ns_posterior.npz",
             samples=samples, weights=weights,
             fit_names=np.array(names, dtype=object))
    w = weights / weights.sum()
    print(f"[retrieve] posterior saved: {len(samples)} weighted samples")
    print(f"  fit_names = {names}")
    print(f"  weighted means = {np.round(np.average(samples, axis=0, weights=w), 3)}")
    print(f"  → {OUT/'wasp39b_ns_posterior.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=int, default=300)
    ap.add_argument("--floor", type=float, default=0.01)
    ap.add_argument("--clouds", action="store_true")     # Path B: 7-param cloud fit
    ap.add_argument("--highres", action="store_true")    # fit native-res spectrum
    ap.add_argument("--nbins", type=int, default=150)
    a = ap.parse_args()
    main(a.live, a.floor, a.clouds, a.highres, a.nbins)
