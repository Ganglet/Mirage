"""
Fix 1 — generate MIRAGE training data with EXTENDED coverage, using the WI-1
TauREx forward model. The ABC/Ariel training set tops out at log-abundance ≈ −3
and small planets (depth ≲ 0.009); WASP-39b sits AT that edge (log ≈ −3, depth
≈ 0.021), so the model extrapolates and fails (P3-D5). This regenerates spectra
over a prior that actually COVERS the WASP-39b regime:

  θ  = [T, log_H2O, log_CO2, log_CH4, log_CO, log_NH3]   (abundances → −1)
  geometry (Rp, Mp, R*, M*, T*) sampled to span depth ≈ 0.001–0.06

Runs in `mirage-taurex` (needs the forward model + numba). Chunkable via --seed
for parallel runs. Output = raw spectra HDF; normalise/split with a follow-up.

    conda run -n mirage-taurex python scripts/generate_training_data.py --n 200 --out data/abc_ext/chunk0.hdf --seed 0
"""

import argparse
from pathlib import Path
import time

import h5py
import numpy as np

import taurex_forward as tf   # PYTHONPATH=scripts

# extended prior (covers WASP-39b: high metallicity + inflated large planet)
T_RANGE = (300.0, 2500.0)
LOGX_RANGE = (-9.0, -1.0)                     # was ABC ~[-9,-3]; extended up
RP_RJ = (0.8, 1.6)                            # Jupiter radii
RS_RSUN = (0.60, 1.30)                        # solar radii  → depth 0.004–0.07
MP_MJ = (0.3, 3.0)
MS_MSUN = (0.70, 1.20)
TS = (4500.0, 6500.0)
NOISE_PPM = (30.0, 150.0)                     # per-bin depth noise (JWST-like)
PCLOUD_LOG = (2.0, 6.0)                        # Path B: log10 gray-cloud pressure [Pa]

# P5 multi-target: per-planet FIXED geometry + radius/T priors + grid range (from the
# planet's published spectrum). K2-18b is a cold sub-Neptune → far smaller Rp / cooler T
# than the hot-Jupiter default, so it needs its own priors, not just a geometry swap.
PLANETS = {
    "wasp39": dict(geom=tf.WASP39, rp=(0.8, 1.6), T=(300.0, 2500.0),
                   spec="data/jwst_wasp39b_spectrum.csv"),
    "wasp96": dict(geom=tf.WASP96, rp=(0.8, 1.6), T=(300.0, 2500.0),
                   spec="data/jwst_wasp96b_spectrum.csv"),
    "k218":   dict(geom=tf.K2_18b, rp=(0.10, 0.40), T=(100.0, 600.0),
                   spec="data/jwst_k2_18b_spectrum.csv"),
}


def main(n, out, seed, tprofile="isothermal", nbins=0, planet="wasp39"):
    grad = (tprofile == "npoint")                 # P3-D8: 2-point gradient, θ=[T_surf,T_top,5logX]
    radius_mode = (tprofile == "radius")          # P3-D11: θ=[rp_rj,T,5logX], per-planet geometry
    P = PLANETS[planet]
    pgeom = P["geom"]; rp_lo, rp_hi = P["rp"]; T_lo, T_hi = P["T"]   # pgeom ≠ the geom[] buffer below
    rng = np.random.default_rng(seed)
    if nbins:                                     # P5: hi-res grid over THIS planet's real range
        import pandas as pd
        df = pd.read_csv(P["spec"])
        lo, hi = float(df["wavelength_um"].min()), float(df["wavelength_um"].max())
        wlen = np.geomspace(lo, hi, nbins)
    else:                                         # default: ABC 52-bin grid
        with h5py.File("data/abc/abc_test.hdf") as f:
            wlen = f["wlen"][0].astype(float)
    nb = len(wlen)
    wl = np.sort(wlen)
    edges = np.empty(len(wl) + 1)
    edges[1:-1] = np.sqrt(wl[:-1] * wl[1:])
    edges[0] = wl[0] ** 2 / edges[1]; edges[-1] = wl[-1] ** 2 / edges[-2]
    width_asc = np.diff(edges); order = np.argsort(wlen)
    wlwidth = np.empty_like(width_asc); wlwidth[order] = width_asc

    theta = np.empty((n, 7), np.float32)          # Path B: + log10 P_cloud
    flux = np.empty((n, nb), np.float32)
    noise = np.empty((n, nb), np.float32)
    geom = np.empty((n, 5), np.float32)

    t0 = time.time()
    for i in range(n):
        T = rng.uniform(T_lo, T_hi) if radius_mode else rng.uniform(*T_RANGE)
        logx = rng.uniform(*LOGX_RANGE, size=5)
        if radius_mode:
            # P3-D11: mirror the NS that WORKED (χ²=0.76) — fix star/mass/geometry to the
            # TARGET planet, float ONLY radius + T + abundances. θ=[rp_rj,T,5logX]. Fixing
            # rs makes the baseline (rp/rs)² a bijection of rp, so radius is learnable.
            # Per-planet priors (rp,T) + geometry from PLANETS[planet].
            rp = rng.uniform(rp_lo, rp_hi)
            g = dict(rp_rj=rp, mp_mj=pgeom["mp_mj"], rs_rsun=pgeom["rs_rsun"],
                     ms_msun=pgeom["ms_msun"], ts=pgeom["ts"])
            tm = tf.build_model(**g)               # isothermal, target-planet star
            spec = tf.spectrum(tm, np.concatenate([[T], logx]), wlen, wlwidth)
            th = np.concatenate([[rp, T], logx]).astype(np.float64)
        else:
            g = dict(rp_rj=rng.uniform(*RP_RJ), mp_mj=rng.uniform(*MP_MJ),
                     rs_rsun=rng.uniform(*RS_RSUN), ms_msun=rng.uniform(*MS_MSUN),
                     ts=rng.uniform(*TS))
            if grad:                               # θ = [T_surface, T_top, 5×logX]; T_top ≤ T_surface
                T_top = rng.uniform(T_RANGE[0], T)
                tm = tf.build_model(**g, tprofile="npoint")
                th = np.concatenate([[T, T_top], logx]).astype(np.float64)
                spec = tf.spectrum(tm, th, wlen, wlwidth, tprofile="npoint")
            else:                                  # Path B: θ = [T, 5×logX, log P_cloud]
                logpc = rng.uniform(*PCLOUD_LOG)
                tm = tf.build_model(**g, use_clouds=True)
                th = np.concatenate([[T], logx, [logpc]]).astype(np.float64)
                spec = tf.spectrum(tm, th, wlen, wlwidth)
        sig = spec.mean() * rng.uniform(*NOISE_PPM) * 1e-6   # per-planet noise scale
        theta[i] = th; flux[i] = spec
        noise[i] = np.full(nb, max(sig, 1e-7))
        geom[i] = [g["rp_rj"], g["mp_mj"], g["rs_rsun"], g["ms_msun"], g["ts"]]
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n}  ({(time.time()-t0)/(i+1):.2f}s/planet)")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("theta", data=theta)
        f.create_dataset("flux", data=flux)
        f.create_dataset("noise", data=noise)
        f.create_dataset("geometry", data=geom)
        f.create_dataset("wlen", data=wlen[None, :].astype(np.float32))
    dt = time.time() - t0
    print(f"[gen] {n} planets → {out}  ({dt:.0f}s, {dt/n:.2f}s/planet)")
    print(f"  depth range [{flux.min():.5f}, {flux.max():.5f}]  "
          f"(WASP-39b ≈ 0.021 should be well inside)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=str, default="data/abc_ext/chunk0.hdf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tprofile", default="isothermal",  # npoint=P3-D8 gradient; radius=P3-D11
                    choices=["isothermal", "npoint", "radius"])
    ap.add_argument("--nbins", type=int, default=0)      # P5: >0 = hi-res geomspace grid over real range
    ap.add_argument("--planet", default="wasp39", choices=["wasp39", "wasp96", "k218"])
    a = ap.parse_args()
    main(a.n, a.out, a.seed, a.tprofile, a.nbins, a.planet)
