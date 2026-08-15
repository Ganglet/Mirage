"""
Extract out-of-transit frames + per-wavelength sigma for WASP-96b and K2-18b.

Produces the same CSV format as WASP-39b (WASP39b_out_of_transit_data/):
  - wavelength_um
  - flux_jy
  - flux_error_jy   <- per-wavelength sigma, NOT normalised
  - dq_flag
  - integration_num
  - instrument

One file per instrument/mode, time axis preserved (not collapsed).

Transit parameters used for OOT frame selection:
  WASP-96b:  T14 ~ 2.32 h,  total obs ~ 6.4 h  → OOT = first+last ~30%
  K2-18b:    T14 ~ 2.86 h,  total obs ~ 8.0 h  → OOT = first+last ~30%

Usage:
    python scripts/extract_wasp96b_k2_18b_oot.py

Reads from:  MAST_downloads/
Writes to:   output/wasp96b_oot_data/   and   output/k2_18b_oot_data/
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from astropy.io import fits
    from astropy.table import Table
except ImportError:
    raise ImportError("pip install astropy")


# ── Transit timing parameters ─────────────────────────────────────────────

TRANSIT_PARAMS = {
    "WASP-96b": {
        "oot_fraction": 0.30,   # first 30% + last 30% = out-of-transit
        "instruments": {
            "NIRISS_SOSS": "0.6–2.8 µm, R~700",
        }
    },
    "K2-18b": {
        "oot_fraction": 0.30,
        "instruments": {
            "NIRSpec_G395H":  "2.87–5.14 µm, R~2700",
            "NIRCam_F322W2":  "2.40–4.00 µm, R~1600",
        }
    },
}


def load_x1dints(filepath: Path) -> dict:
    """Load JWST x1dints FITS → arrays in native byte order."""
    with fits.open(filepath) as hdul:
        hdr = hdul[0].header
        tbl = Table(hdul["EXTRACT1D"].data)

        def to_f64(col):
            arr = np.array(tbl[col])
            return arr.astype(np.float64)

        def to_i32(col):
            arr = np.array(tbl[col])
            return arr.astype(np.int32)

        return {
            "wavelength": to_f64("WAVELENGTH"),
            "flux":       to_f64("FLUX"),
            "flux_error": to_f64("FLUX_ERROR"),
            "dq":         to_i32("DQ"),
            "int_num":    to_i32("INT_NUM"),
            "instrument": hdr.get("INSTRUME", "UNKNOWN"),
            "filter":     hdr.get("FILTER",   "N/A"),
            "grating":    hdr.get("GRATING",  "N/A"),
            "target":     hdr.get("TARGPROP", "UNKNOWN"),
            "n_ints":     int(hdr.get("NINTS", 0)),
        }


def get_oot_mask(int_nums: np.ndarray, oot_fraction: float):
    """Return boolean mask for out-of-transit integrations."""
    unique = np.unique(int_nums)
    n = len(unique)
    n_oot = int(n * oot_fraction)
    pre_end  = unique[n_oot - 1]
    post_start = unique[n - n_oot]
    return (int_nums <= pre_end) | (int_nums >= post_start)


def extract_to_df(data: dict, oot_mask: np.ndarray,
                  instrument_label: str) -> pd.DataFrame:
    """Extract masked data into a tidy DataFrame."""
    wl  = data["wavelength"][oot_mask].astype(np.float64)
    fl  = data["flux"][oot_mask].astype(np.float64)
    fe  = data["flux_error"][oot_mask].astype(np.float64)
    dq  = data["dq"][oot_mask].astype(np.int32)
    ii  = data["int_num"][oot_mask].astype(np.int32)

    # Flatten if 2-D (some instruments return (n_ints, n_lambda))
    if wl.ndim == 2:
        n_i, n_w = wl.shape
        wl = wl.flatten()
        fl = fl.flatten()
        fe = fe.flatten()
        dq = dq.flatten()
        ii = np.repeat(ii[:, 0] if ii.ndim == 2 else ii, n_w)

    df = pd.DataFrame({
        "wavelength_um":  wl,
        "flux_jy":        fl,
        "flux_error_jy":  fe,   # per-lambda sigma, NOT normalised
        "dq_flag":        dq,
        "integration_num": ii,
        "instrument":     instrument_label,
    })
    df = df[np.isfinite(df["wavelength_um"]) &
            np.isfinite(df["flux_jy"]) &
            np.isfinite(df["flux_error_jy"])]
    return df.sort_values(["integration_num", "wavelength_um"]).reset_index(drop=True)


def process_planet(planet_name: str, fits_dir: Path, out_dir: Path):
    """Process all x1dints files for one planet and write CSVs."""
    fits_files = list(fits_dir.rglob("*x1dints.fits"))
    if not fits_files:
        print(f"  WARNING: No x1dints.fits files found in {fits_dir}")
        print(f"     Run: python scripts/download_wasp96b_k2_18b.py first")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    params = TRANSIT_PARAMS[planet_name]
    summary_rows = []

    print(f"\n  {'─'*58}")
    print(f"  {planet_name}: {len(fits_files)} x1dints files")

    for fpath in fits_files:
        print(f"\n    Loading: {fpath.name}")
        data = load_x1dints(fpath)

        inst  = data["instrument"]
        filt  = data["filter"]
        grat  = data["grating"]
        label = f"{inst}_{filt}_{grat}".replace("/", "_")

        oot_fraction = params["oot_fraction"]
        oot_mask = get_oot_mask(data["int_num"], oot_fraction)

        n_total = len(np.unique(data["int_num"]))
        n_oot   = len(np.unique(data["int_num"][oot_mask]))

        print(f"    Instrument: {inst} | Filter: {filt} | Grating: {grat}")
        print(f"    Total integrations: {n_total} | OOT: {n_oot}")

        df = extract_to_df(data, oot_mask, label)

        if len(df) == 0:
            print(f"    WARNING: No valid data after filtering")
            continue

        wl_min = df["wavelength_um"].min()
        wl_max = df["wavelength_um"].max()
        n_uniq_wl = df["wavelength_um"].nunique()

        print(f"    Wavelength range: {wl_min:.3f}–{wl_max:.3f} µm")
        print(f"    Unique wavelengths: {n_uniq_wl}")
        print(f"    Total data points: {len(df)}")

        # ── Write full OOT ──
        oot_file = out_dir / f"{label}_out_of_transit_full.csv"
        df.to_csv(oot_file, index=False)
        print(f"    ✓ {oot_file.name}")

        # ── Write pre-transit only ──
        unique_ints = np.unique(df["integration_num"])
        n_pre = int(len(unique_ints) * 0.5)
        pre_ints = unique_ints[:n_pre]
        pre_df = df[df["integration_num"].isin(pre_ints)]
        pre_file = out_dir / f"{label}_pre_transit.csv"
        pre_df.to_csv(pre_file, index=False)

        # ── Write post-transit only ──
        post_ints = unique_ints[n_pre:]
        post_df = df[df["integration_num"].isin(post_ints)]
        post_file = out_dir / f"{label}_post_transit.csv"
        post_df.to_csv(post_file, index=False)

        summary_rows.append({
            "planet":                   planet_name,
            "instrument":               inst,
            "filter":                   filt,
            "grating":                  grat,
            "wavelength_min_um":        round(wl_min, 4),
            "wavelength_max_um":        round(wl_max, 4),
            "n_unique_wavelengths":     n_uniq_wl,
            "n_total_integrations":     n_total,
            "n_oot_integrations":       n_oot,
            "total_oot_data_points":    len(df),
            "oot_fraction_used":        oot_fraction,
            "source_file":              fpath.name,
        })

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_file = out_dir / "summary.csv"
        summary.to_csv(summary_file, index=False)
        print(f"\n    ✓ Summary: {summary_file}")


def write_readme(planet_name: str, out_dir: Path):
    """Write a README for the data package."""
    readme = f"""# {planet_name} — Out-of-Transit Data Package

Same format as WASP-39b (WASP39b_out_of_transit_data/).

## Column Format

| Column | Units | Description |
|--------|-------|-------------|
| wavelength_um | µm | Native instrument wavelength grid |
| flux_jy | Jy | Spectral flux (NOT normalised) |
| flux_error_jy | Jy | Per-wavelength sigma (1σ, as observed) |
| dq_flag | — | JWST pipeline data quality flag (0=good) |
| integration_num | — | Time-series integration index |
| instrument | — | Instrument/mode identifier |

## Files

- `*_out_of_transit_full.csv` — all OOT integrations (pre + post transit)
- `*_pre_transit.csv` — pre-transit frames only
- `*_post_transit.csv` — post-transit frames only
- `summary.csv` — per-instrument statistics

## OOT Selection

First 30% + last 30% of total integrations = out-of-transit baseline.

## Notes

- flux_error_jy is the raw pipeline uncertainty — not continuum-normalised
- Filter dq_flag == 0 for clean data
- Time axis is preserved — one row per (integration × wavelength) pixel
"""
    (out_dir / "README.md").write_text(readme)


def main():
    print("="*62)
    print("  WASP-96b + K2-18b Out-of-Transit Extraction")
    print("="*62)

    planets = {
        "WASP-96b": (
            Path("MAST_downloads/WASP_96b"),
            Path("output/wasp96b_oot_data"),
        ),
        "K2-18b": (
            Path("MAST_downloads/K2_18b"),
            Path("output/k2_18b_oot_data"),
        ),
    }

    for planet_name, (fits_dir, out_dir) in planets.items():
        process_planet(planet_name, fits_dir, out_dir)
        write_readme(planet_name, out_dir)

    print(f"\n{'='*62}")
    print("  Extraction complete.")
    print("  Packages ready in:")
    print("    output/wasp96b_oot_data/")
    print("    output/k2_18b_oot_data/")
    print("\n  To zip and share:")
    print("    python scripts/zip_oot_packages.py")
    print("="*62)


if __name__ == "__main__":
    main()