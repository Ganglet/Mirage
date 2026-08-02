"""
WI-2 driver / self-test — build the model-input context from a published
WASP-39b transit-depth spectrum.

Until the real published spectrum is in hand, run with --stub to validate the
whole transform end-to-end on a fabricated but physically-plausible WASP-39b
depth spectrum (baseline ~0.021 with H2O/CO2 features). Swap in the real file
with --spectrum path/to/published_depths.csv once sourced (same 4 columns:
instrument, wavelength_um, transit_depth, depth_error).

    PYTHONPATH=scripts python scripts/build_real_observation.py --stub
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import mirage  # noqa: F401
from mirage.datasets.real_jwst import build_observation

# real per-instrument wavelength coverage (from the OOT summary)
BANDS = {
    "NIRISS":       (0.85, 2.83, 400),
    "NIRCam":       (2.40, 4.39, 300),
    "NIRSpec-G395H": (2.88, 3.72, 250),
    "NIRSpec-PRISM": (0.60, 5.29, 400),
}


def make_stub() -> Path:
    """Plausible WASP-39b depth spectrum: baseline (Rp/Rs)^2≈0.0211 + features."""
    rng = np.random.default_rng(0)
    rows = []
    for inst, (lo, hi, n) in BANDS.items():
        wl = np.linspace(lo, hi, n)
        depth = np.full_like(wl, 0.0211)
        depth += 0.0011 * np.exp(-0.5 * ((wl - 1.4) / 0.08) ** 2)   # H2O
        depth += 0.0014 * np.exp(-0.5 * ((wl - 4.3) / 0.10) ** 2)   # CO2
        depth += 0.0007 * np.exp(-0.5 * ((wl - 2.7) / 0.10) ** 2)   # H2O/CO2
        err = np.full_like(wl, 1.2e-4) * (1 + 0.5 * rng.random(n))
        depth = depth + rng.normal(0, err)
        rows.append(pd.DataFrame({"instrument": inst, "wavelength_um": wl,
                                  "transit_depth": depth, "depth_error": err}))
    p = Path(tempfile.mkdtemp()) / "stub_depths.csv"
    pd.concat(rows).to_csv(p, index=False)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectrum", type=str, help="published depth CSV")
    ap.add_argument("--stub", action="store_true", help="use fabricated WASP-39b depths")
    args = ap.parse_args()

    if args.stub or not args.spectrum:
        csv = make_stub()
        print(f"[STUB] fabricated depths → {csv}\n"
              f"       (real numbers pending the published WASP-39b spectrum)\n")
    else:
        csv = Path(args.spectrum)

    ctx, covered = build_observation(csv)
    flux, ebar = ctx["flux"], ctx["error_bars"]

    print(f"Model-input context built:")
    print(f"  grid:            {len(flux)} bins, {ctx['wlen'].min():.2f}–{ctx['wlen'].max():.2f} µm")
    print(f"  covered (real):  {covered.sum()}/{len(flux)}  "
          f"(masked bins filled neutrally)")
    print(f"  flux (ABC-norm): mean {flux[covered].mean():+.2f}σ  "
          f"range [{flux[covered].min():+.2f}, {flux[covered].max():+.2f}]")
    print(f"  σ  (ABC-norm):   median {np.median(ebar[covered]):.3f}  "
          f"range [{ebar[covered].min():.3f}, {ebar[covered].max():.3f}]")
    print(f"\n  Sim-to-real read: ABC per-bin space is N(0,1). WASP-39b sits at "
          f"~{flux[covered].mean():+.1f}σ")
    print(f"  → it lives in the {'TAIL' if abs(flux[covered].mean()) > 1.5 else 'body'} "
          f"of ABC's training depth distribution (distribution shift to watch).")


if __name__ == "__main__":
    main()
