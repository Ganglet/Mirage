"""
Phase 3 WI-2/WI-3 — real-JWST noise covariance from out-of-transit frames.

First concrete Phase-3 step: prove the real data feeds the Phase-2 covariance
machinery. Loads Vedanth's WASP-39b OOT frames, rebins each integration onto
the model's 52-bin ABC grid, converts to RELATIVE flux (so the covariance is
dimensionless and lands in the same standardized space the embedding trained
on, σ≈0.05-0.3), and estimates the empirical Σ̂.

Run from Project/:
    python scripts/build_real_covariance.py
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ABC = Path("data/abc/abc_test.hdf")
OOT = Path("data/jwst_wasp39b_oot")   # gitignored (real JWST data, D11); Zenodo at release
INSTRUMENTS = {
    "NIRISS":       "NIRISS_CLEAR_N_A_out_of_transit_full.csv",
    "NIRCam":       "NIRCAM_F322W2_N_A_out_of_transit_full.csv",
    "NIRSpec-G395H": "NIRSPEC_F290LP_G395H_out_of_transit_full.csv",
    "NIRSpec-PRISM": "NIRSPEC_CLEAR_PRISM_out_of_transit_full.csv",
}


def abc_grid():
    with h5py.File(ABC) as f:
        wl = f["wlen"][0].astype(float)
    wl = np.sort(wl)                      # ascending for binning
    edges = np.empty(len(wl) + 1)
    edges[1:-1] = np.sqrt(wl[:-1] * wl[1:])        # geometric midpoints (log grid)
    edges[0] = wl[0] ** 2 / edges[1]
    edges[-1] = wl[-1] ** 2 / edges[-2]
    return wl, edges


def rebin_frames(df, edges):
    """(long OOT df) -> (n_int, n_bins) relative-flux matrix, bin-averaged."""
    n_bins = len(edges) - 1
    ints = np.sort(df["integration_num"].unique())
    M = np.full((len(ints), n_bins), np.nan)
    bin_of = np.digitize(df["wavelength_um"].to_numpy(), edges) - 1
    df = df.assign(_bin=bin_of)
    df = df[(df._bin >= 0) & (df._bin < n_bins)]
    row_of = {ii: r for r, ii in enumerate(ints)}
    g = df.groupby(["integration_num", "_bin"])["flux_jy"].mean()
    for (ii, b), v in g.items():
        M[row_of[ii], b] = v
    return M, ints


def main():
    wl, edges = abc_grid()
    print(f"Model grid: {len(wl)} bins, {wl.min():.3f}-{wl.max():.3f} µm\n")

    # ---- alignment check: which model bins does each instrument cover? ----
    print(f"{'instrument':<14} {'λ-range (µm)':>16} {'frames':>7} {'bins hit':>9}")
    print("-" * 52)
    prism = None
    for name, fn in INSTRUMENTS.items():
        df = pd.read_csv(OOT / fn)
        df = df[df["dq_flag"] == 0]
        lo, hi = df["wavelength_um"].min(), df["wavelength_um"].max()
        M, ints = rebin_frames(df, edges)
        hit = int(np.isfinite(M).any(axis=0).sum())
        print(f"{name:<14} {lo:7.2f}-{hi:5.2f} {len(ints):>7} {hit:>9}/{len(wl)}")
        if name == "NIRSpec-PRISM":
            prism = (M, ints)

    # ---- covariance from PRISM (broadest + best-sampled) ----
    M, ints = prism
    covered = np.isfinite(M).all(axis=0)          # bins present in every frame
    idx = np.where(covered)[0]
    F = M[:, idx]                                  # (n_int, n_covered), absolute Jy
    rel = F / F.mean(axis=0) - 1.0                 # dimensionless residuals

    Sigma = np.cov(rel, rowvar=False)              # (n_covered, n_covered)
    d = np.sqrt(np.diag(Sigma))
    corr = Sigma / np.outer(d, d)
    evals = np.linalg.eigvalsh(Sigma)

    print(f"\nPRISM covariance on model grid")
    print(f"  covered bins:        {len(idx)}/{len(wl)}  "
          f"(λ {wl[idx].min():.2f}-{wl[idx].max():.2f} µm)")
    print(f"  frames used:         {F.shape[0]}")
    print(f"  PD (min eigval>0):   {evals.min():.2e}  ->  {'PASS' if evals.min() > 0 else 'FAIL'}")
    print(f"  condition number:    {evals.max()/evals.min():.1f}")
    print(f"  rel-noise σ (diag):  {d.min():.3f}-{d.max():.3f}  (median {np.median(d):.3f})")
    print(f"  training σ regime:   0.05-0.30  (embedding was trained here)")
    # nearest-neighbour correlation = is there real off-diagonal structure?
    nn = np.diag(corr, k=1)
    print(f"  nn off-diag corr:    median {np.median(nn):+.2f}  "
          f"(range {nn.min():+.2f}..{nn.max():+.2f})")
    print(f"  mean |off-diag|:     {np.abs(corr[~np.eye(len(idx), dtype=bool)]).mean():.3f}")

    outp = OOT / "prism_Sigma_modelgrid.npz"
    np.savez(outp, Sigma=Sigma, corr=corr, bin_index=idx, wlen=wl[idx],
             rel_sigma=d)
    print(f"\nSaved -> {outp}")


if __name__ == "__main__":
    main()
