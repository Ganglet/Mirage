"""
WI-2 — real JWST observation adapter (Phase 3).

Turns a published multi-instrument transmission spectrum (transit depths) into
the exact model-input the ABC-trained flow expects: a 52-bin, ABC-normalised
depth vector + per-λ σ, on the model's own wavelength grid.

Canonical input (one long CSV):
    instrument, wavelength_um, transit_depth, depth_error
with transit_depth = (Rp/Rs)^2 (dimensionless, ~0.005–0.02). Massage whatever
the published source provides into these four columns.

The critical step is the ABC-space normalisation: the model trained on flux
standardised per bin with the mean/std stored in `abc_*.hdf` (P2-D10), so the
real depth MUST be pushed through the SAME (x − mean)/std to be in-distribution.
Real coverage (0.6–5.29 µm) fills 47/52 bins; the rest are masked (P3-D1).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ABC = Path("data/abc/abc_test.hdf")


def abc_grid_and_norm(stats_hdf=ABC):
    """
    Model grid + per-bin normalisation stats, in the HDF's stored order.
    `stats_hdf` selects which training set's flux_mean/std to use — this MUST
    match the model being fed (e.g. data/abc_ext/abc_ext_train.hdf for the
    extended-coverage retrain, whose mean depth ≈ WASP-39b's, so the real
    spectrum lands in-distribution instead of at +2σ).
    """
    with h5py.File(stats_hdf) as f:
        wl = f["wlen"][0].astype(float)
        mean = f["flux_mean"][:].astype(float)
        std = f["flux_std"][:].astype(float)
    return wl, mean, std          # stored (descending) order, mutually aligned


def _log_edges(wl_asc):
    e = np.empty(len(wl_asc) + 1)
    e[1:-1] = np.sqrt(wl_asc[:-1] * wl_asc[1:])
    e[0] = wl_asc[0] ** 2 / e[1]
    e[-1] = wl_asc[-1] ** 2 / e[-2]
    return e


def rebin_spectrum(df, wl_asc):
    """
    Long depth df → (depth, err, covered) on the ascending grid `wl_asc`.
    Inverse-variance combine of every point in a bin (so overlapping
    instruments merge by weight, and errors propagate correctly).
    """
    edges = _log_edges(wl_asc)
    nb = len(wl_asc)
    b = np.digitize(df["wavelength_um"].to_numpy(), edges) - 1
    d = df.assign(_b=b)
    d = d[(d._b >= 0) & (d._b < nb)]
    w = 1.0 / np.clip(d["depth_error"].to_numpy(), 1e-12, None) ** 2
    d = d.assign(_w=w)

    depth = np.full(nb, np.nan)
    err = np.full(nb, np.nan)
    for bb, grp in d.groupby("_b"):
        ww, dd = grp["_w"].to_numpy(), grp["transit_depth"].to_numpy()
        depth[bb] = np.sum(ww * dd) / np.sum(ww)
        err[bb] = 1.0 / np.sqrt(np.sum(ww))
    return depth, err, np.isfinite(depth)


def build_observation(spectrum_csv, stats_hdf=ABC, remove_baseline=False):
    """
    Published transit-depth spectrum → model-input context dict.

    `stats_hdf` = the training set whose normalisation the target model expects.
    `remove_baseline` (Fix 2) = subtract the spectrum's own mean first, matching
    a shape-conditioned model (radius/level nuisance removed).

    Returns (context, covered):
      context = {"flux": (52,), "wlen": (52,), "error_bars": (52,)}  float32,
                in the model's stored bin order and ABC-normalised space.
      covered = (52,) bool — bins with real data (masked bins filled neutrally).
    """
    wl_s, mean_s, std_s = abc_grid_and_norm(stats_hdf)
    order = np.argsort(wl_s)                 # to ascending for binning
    wl_a, mean_a, std_a = wl_s[order], mean_s[order], std_s[order]

    df = pd.read_csv(spectrum_csv)
    depth_a, err_a, cov_a = rebin_spectrum(df, wl_a)

    if remove_baseline:                      # Fix 2: shape space (match training)
        depth_a = depth_a - np.nanmean(depth_a[cov_a])

    # ABC-space normalisation — the sim-to-real bridge (same transform as training)
    flux_a = (depth_a - mean_a) / std_a
    ebar_a = err_a / std_a

    # masked bins: neutral fill (0 = ABC mean; σ = median observed → "no strong info")
    med_sigma = np.nanmedian(ebar_a[cov_a]) if cov_a.any() else 1.0
    flux_a = np.where(cov_a, flux_a, 0.0)
    ebar_a = np.where(cov_a, ebar_a, med_sigma)

    inv = np.argsort(order)                   # back to stored order
    context = {
        "flux": flux_a[inv].astype(np.float32),
        "wlen": wl_s.astype(np.float32),
        "error_bars": ebar_a[inv].astype(np.float32),
    }
    return context, cov_a[inv]
