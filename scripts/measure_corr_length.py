"""
Phase 3 P3-D1 refinement — measure the TRUE real correlation length.

The 52-bin envelope check found real correlation "sub-resolution", but that is
partly an artifact: rebinning PRISM's 409 native points down to 47 bins averages
within bins and decorrelates. Here I estimate the correlation length at a ladder
of resolutions (52 → native 409) so the length stops shrinking once the grid is
fine enough to resolve it. That asymptotic value is what the D4 training kernel
must contain.

Run from Project/ with PYTHONPATH=scripts:
    PYTHONPATH=scripts python scripts/measure_corr_length.py
"""

import numpy as np
import pandas as pd

from build_real_covariance import OOT
from validate_whitening import corr_1e_distance

PRISM = "NIRSPEC_CLEAR_PRISM_out_of_transit_full.csv"


def log_edges(grid):
    g = np.sort(grid)
    e = np.empty(len(g) + 1)
    e[1:-1] = np.sqrt(g[:-1] * g[1:])
    e[0] = g[0] ** 2 / e[1]
    e[-1] = g[-1] ** 2 / e[-2]
    return e


def rebin(df, grid):
    edges = log_edges(grid)
    nb = len(grid)
    ints = np.sort(df["integration_num"].unique())
    M = np.full((len(ints), nb), np.nan)
    b = np.digitize(df["wavelength_um"].to_numpy(), edges) - 1
    d = df.assign(_b=b)
    d = d[(d._b >= 0) & (d._b < nb)]
    row = {ii: r for r, ii in enumerate(ints)}
    for (ii, bb), v in d.groupby(["integration_num", "_b"])["flux_jy"].mean().items():
        M[row[ii], bb] = v
    return M, np.sort(grid)


def main():
    df = pd.read_csv(OOT / PRISM)
    df = df[df["dq_flag"] == 0]
    lo, hi = df["wavelength_um"].min(), df["wavelength_um"].max()
    native = np.sort(df["wavelength_um"].unique())
    n_frames = df["integration_num"].nunique()
    print(f"PRISM: {len(native)} native λ over {lo:.2f}-{hi:.2f} µm, {n_frames} frames\n")

    grids = [("model-52", np.geomspace(lo, hi, 52)),
             ("104", np.geomspace(lo, hi, 104)),
             ("156", np.geomspace(lo, hi, 156)),
             ("208", np.geomspace(lo, hi, 208)),
             ("native", native)]

    print(f"{'grid':>9} {'bins':>5} {'Δλ_min µm':>10} {'nn-corr':>8} {'corr-L µm':>11}")
    print("-" * 48)
    for name, grid in grids:
        M, g = rebin(df, grid)
        covered = np.isfinite(M).all(axis=0)
        F, wl = M[:, covered], g[covered]
        rel = F / F.mean(axis=0) - 1.0
        corr = np.corrcoef(rel, rowvar=False)
        nn = np.median(np.diag(corr, k=1))
        L = corr_1e_distance(corr, wl)
        dmin = np.diff(wl).min()
        rank = "" if F.shape[0] > F.shape[1] else "  (rank-def)"
        Ls = f"{L:>9.3f}" if np.isfinite(L) else f"  <{dmin:>5.3f}"
        print(f"{name:>9} {covered.sum():>5} {dmin:>10.3f} {nn:>+8.2f} {Ls:>11}{rank}")

    print("\nRead: corr-L stops shrinking once bins are finer than the true")
    print("length. Set the D4 kernel's short edge below that asymptote.")


if __name__ == "__main__":
    main()
