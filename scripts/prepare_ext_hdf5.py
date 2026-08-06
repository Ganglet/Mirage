"""
Fix 1 (step 2) — merge the generated chunks, clean, per-bin standardise, split.
Produces data/abc_ext/abc_ext_{train,valid,test}.hdf in the SAME format the
model trains on (identical to prepare_abc_hdf5), so retraining is just a config
file_path swap. Normalisation stats are recomputed on the EXTENDED train split.

    python scripts/prepare_ext_hdf5.py
"""

from pathlib import Path
import argparse
import glob

import h5py
import numpy as np

EPS = 1e-8


def main(remove_baseline=False, prefix="abc_ext", indir="data/abc_ext"):
    EXT = Path(indir)
    chunks = sorted(glob.glob(str(EXT / "chunk*.hdf")))
    print(f"Merging {len(chunks)} chunks ...")
    theta = np.concatenate([h5py.File(c)["theta"][:] for c in chunks])
    flux = np.concatenate([h5py.File(c)["flux"][:] for c in chunks])
    noise = np.concatenate([h5py.File(c)["noise"][:] for c in chunks])
    with h5py.File(chunks[0]) as f:
        wlen = f["wlen"][:]
    n = len(theta)
    ok = np.isfinite(flux).all(1) & np.isfinite(theta).all(1)
    theta, flux, noise = theta[ok], flux[ok], noise[ok]
    print(f"  {n} planets, {int((~ok).sum())} dropped → {len(theta)} valid")
    print(f"  depth range [{flux.min():.5f}, {flux.max():.5f}]")

    if remove_baseline:
        # Fix 2 — condition on SHAPE not level: subtract each spectrum's own
        # mean, removing the radius-set baseline (a nuisance the model can't
        # infer). WASP-39b's shape is then in-distribution regardless of its
        # large radius. (Same "structure not scale" idea as P3-D1 whitening.)
        flux = flux - flux.mean(axis=1, keepdims=True)
        print(f"  baseline removed → shape space, range [{flux.min():.5f}, {flux.max():.5f}]")

    n = len(theta)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    splits = {"train": idx[:n_tr], "valid": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}

    tr = flux[splits["train"]].astype(np.float64)
    flux_mean = tr.mean(0).astype(np.float32)
    flux_std = (tr.std(0) + EPS).astype(np.float32)
    flux_n = ((flux - flux_mean) / flux_std).astype(np.float32)
    noise_n = (noise / flux_std).astype(np.float32)

    for name, sel in splits.items():
        out = EXT / f"{prefix}_{name}.hdf"
        with h5py.File(out, "w") as f:
            f.create_dataset("theta", data=theta[sel])
            f.create_dataset("flux", data=flux_n[sel])
            f.create_dataset("wlen", data=wlen)
            f.create_dataset("noise", data=noise_n[sel])
            f.create_dataset("planet_id", data=sel.astype(np.int64))
            f.create_dataset("flux_mean", data=flux_mean)
            f.create_dataset("flux_std", data=flux_std)
        print(f"  {name}: {len(sel)} → {out}")
    print("\nParameter ranges (train):")
    mols = ["log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
    if theta.shape[1] == 7 and prefix.startswith("abc_grad"):
        names = ["T_surface", "T_top"] + mols            # P3-D8 gradient θ
    elif theta.shape[1] == 7 and prefix.startswith("abc_rad"):
        names = ["planet_radius", "T"] + mols            # P3-D11 radius θ
    elif theta.shape[1] == 7:
        names = ["T"] + mols + ["log_Pcloud"]            # Path B cloud θ
    else:
        names = ["T"] + mols
    for i, nm in enumerate(names):
        t = theta[splits["train"], i]
        print(f"  {nm}: [{t.min():.2f}, {t.max():.2f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove-baseline", action="store_true")
    ap.add_argument("--prefix", default="abc_ext")
    ap.add_argument("--dir", default="data/abc_ext")
    a = ap.parse_args()
    main(a.remove_baseline, a.prefix, a.dir)
