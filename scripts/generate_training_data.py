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


def main(n, out, seed):
    rng = np.random.default_rng(seed)
    # grid from ABC (wlen + widths), shared
    with h5py.File("data/abc/abc_test.hdf") as f:
        wlen = f["wlen"][0].astype(float)
    wl = np.sort(wlen)
    edges = np.empty(len(wl) + 1)
    edges[1:-1] = np.sqrt(wl[:-1] * wl[1:])
    edges[0] = wl[0] ** 2 / edges[1]; edges[-1] = wl[-1] ** 2 / edges[-2]
    width_asc = np.diff(edges); order = np.argsort(wlen)
    wlwidth = np.empty_like(width_asc); wlwidth[order] = width_asc

    theta = np.empty((n, 7), np.float32)          # Path B: + log10 P_cloud
    flux = np.empty((n, 52), np.float32)
    noise = np.empty((n, 52), np.float32)
    geom = np.empty((n, 5), np.float32)

    t0 = time.time()
    for i in range(n):
        T = rng.uniform(*T_RANGE)
        logx = rng.uniform(*LOGX_RANGE, size=5)
        logpc = rng.uniform(*PCLOUD_LOG)
        g = dict(rp_rj=rng.uniform(*RP_RJ), mp_mj=rng.uniform(*MP_MJ),
                 rs_rsun=rng.uniform(*RS_RSUN), ms_msun=rng.uniform(*MS_MSUN),
                 ts=rng.uniform(*TS))
        tm = tf.build_model(**g, use_clouds=True)
        th = np.concatenate([[T], logx, [logpc]]).astype(np.float64)
        spec = tf.spectrum(tm, th, wlen, wlwidth)
        sig = spec.mean() * rng.uniform(*NOISE_PPM) * 1e-6   # per-planet noise scale
        theta[i] = th; flux[i] = spec
        noise[i] = np.full(52, max(sig, 1e-7))
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
    a = ap.parse_args()
    main(a.n, a.out, a.seed)
