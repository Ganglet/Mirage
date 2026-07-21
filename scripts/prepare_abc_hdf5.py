"""
Convert ABC Zenodo dataset (Level2Data) to HDF5 files for fm4ar and sbi-ear.

Uses Level2Data because it has matched (spectrum, ground-truth theta) pairs
for all 91,392 planets. FM_Parameter_Table.csv gives the true theta;
SpectralData.hdf5 gives the forward-model spectrum.

Cleaning + normalisation (folded in — see problems_and_decisions): the raw ABC
flux is ~500x smaller than the O(1) scale fm4ar/Gebhard were built for and is
polluted by corrupt spectra (values to 4304). Left as-is the transformer learns
almost nothing (near-prior posteriors). So we:
  1. drop planets with any |flux| bin > FLUX_CUTOFF (unphysical transit depth)
  2. per-bin standardise flux using TRAIN-split stats → zero-mean, unit-var per
     wavelength (isolates each planet's deviation from the mean spectrum);
     scale `noise` by the same per-bin std so error bars stay consistent.

Output: data/abc/abc_{train,valid,test}.hdf
Keys per file:
  theta      (N, 6)   float32  — [T, log_H2O, log_CO2, log_CH4, log_CO, log_NH3]
  flux       (N, 52)  float32  — NORMALISED transmission spectrum
  wlen       (1, 52)  float32  — wavelength grid in microns (shared)
  noise      (N, 52)  float32  — per-wavelength 1-sigma noise (scaled by flux_std)
  planet_id  (N,)     int64    — for Tracedata lookup
  flux_mean  (52,)    float32  — per-bin normalisation stats (for inversion)
  flux_std   (52,)    float32

Split: 80 / 10 / 10
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path

ABC_DIR = Path("data/abc")
LEVEL2 = ABC_DIR / "Level2Data"
GT_DIR = LEVEL2 / "Ground Truth Package"

PARAM_COLS = ["planet_temp", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
PARAM_NAMES = ["T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]

FLUX_CUTOFF = 0.5   # transit depths are <<1; anything above is corrupt
EPS = 1e-8


def main() -> None:
    print("Loading FM_Parameter_Table.csv ...")
    gt = pd.read_csv(GT_DIR / "FM_Parameter_Table.csv")
    planet_ids = gt["planet_ID"].values  # keep as array for indexing
    theta_all = gt[PARAM_COLS].values.astype(np.float32)
    n_total = len(planet_ids)
    print(f"  {n_total} planets with ground-truth parameters")

    spectra = np.full((n_total, 52), np.nan, dtype=np.float32)
    noise_all = np.full((n_total, 52), np.nan, dtype=np.float32)
    wlen = None

    print("Loading SpectralData.hdf5 ...")
    with h5py.File(LEVEL2 / "SpectralData.hdf5", "r") as f:
        for i, pid in enumerate(planet_ids):
            if i % 10_000 == 0:
                print(f"  {i}/{n_total}")
            key = f"Planet_{pid}"
            if key not in f:
                continue
            spectra[i] = f[key]["instrument_spectrum"][:]
            noise_all[i] = f[key]["instrument_noise"][:]
            if wlen is None:
                wlen = f[key]["instrument_wlgrid"][:].astype(np.float32)

    # Clean: drop NaN spectra and corrupt (out-of-physical-range) spectra
    valid = ~np.any(np.isnan(spectra), axis=1)
    n_nan = int((~valid).sum())
    valid &= ~(np.abs(spectra) > FLUX_CUTOFF).any(axis=1)
    n_corrupt = int((~valid).sum()) - n_nan
    spectra = spectra[valid]
    noise_all = noise_all[valid]
    theta_all = theta_all[valid]
    planet_ids = planet_ids[valid]
    n = len(theta_all)
    print(f"  Dropped {n_nan} NaN + {n_corrupt} corrupt (|flux|>{FLUX_CUTOFF}) planets")
    print(f"  Valid planets: {n}")
    print(f"  Wavelength range: {wlen.min():.3f} – {wlen.max():.3f} μm")

    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_train = int(0.80 * n)
    n_valid = int(0.10 * n)

    splits = {
        "train": idx[:n_train],
        "valid": idx[n_train : n_train + n_valid],
        "test":  idx[n_train + n_valid :],
    }

    # Per-bin standardisation stats from the TRAIN split only (no leakage)
    train_flux = spectra[splits["train"]].astype(np.float64)
    flux_mean = train_flux.mean(axis=0).astype(np.float32)          # (52,)
    flux_std = (train_flux.std(axis=0) + EPS).astype(np.float32)    # (52,)
    spectra = ((spectra - flux_mean) / flux_std).astype(np.float32)
    noise_all = (noise_all / flux_std).astype(np.float32)
    tf = spectra[splits["train"]]
    print(f"  Normalised flux (train): mean={tf.mean():.4f} std={tf.std():.4f} "
          f"range=[{tf.min():.2f}, {tf.max():.2f}]")

    for name, sel in splits.items():
        out = ABC_DIR / f"abc_{name}.hdf"
        with h5py.File(out, "w") as f:
            f.create_dataset("theta",     data=theta_all[sel])
            f.create_dataset("flux",      data=spectra[sel])
            f.create_dataset("wlen",      data=wlen[None, :])   # (1, 52)
            f.create_dataset("noise",     data=noise_all[sel])
            f.create_dataset("planet_id", data=planet_ids[sel])  # for Tracedata lookup
            f.create_dataset("flux_mean", data=flux_mean)
            f.create_dataset("flux_std",  data=flux_std)
        print(f"  {name}: {len(sel)} planets → {out}")

    print("\nParameter ranges in training set:")
    train_theta = theta_all[splits["train"]]
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name}: [{train_theta[:, i].min():.2f}, {train_theta[:, i].max():.2f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
