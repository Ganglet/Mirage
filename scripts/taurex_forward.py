"""
WI-1 — ABC forward model (TauREx3), the likelihood for real-data importance
sampling. Reproduces the Ariel Data Challenge simulator MIRAGE trained on
(the setup in `data/abc/NeurIPS taurex tutorial/`): transmission, isothermal T,
H2/He (ratio 0.17) fill + constant H2O/CO2/CH4/CO/NH3, CIA + Rayleigh, no clouds.

Runs in the SEPARATE `mirage-taurex` env (numpy>=2, incompatible with fm4ar's
numpy 1.26) — so it is standalone, imports no mirage/torch code. θ = 6 params
[T, log_H2O, log_CO2, log_CH4, log_CO, log_NH3] → (Rp/Rs)^2 on any wl grid.

    conda run -n mirage-taurex python scripts/taurex_forward.py --validate 8
"""

import argparse
import glob
from pathlib import Path

import numpy as np

TUT = Path("data/abc/NeurIPS taurex tutorial")
XSEC = (TUT / "inputs/xsec").resolve()            # ABC ExoTransmit .dat (R~1000, older lists)
HIFI_XSEC = Path("data/opacity_hifi").resolve()   # P3-D10 ExoMolOP R=15000 (latest lists)
CIA = (TUT / "inputs/cia").resolve()
MOLECULES = ["H2O", "CO2", "CH4", "CO", "NH3"]   # order matches θ[1:]


_CACHE_SET = False


def build_model(rp_rj=1.0, mp_mj=1.0, rs_rsun=1.0, ms_msun=1.0, ts=6000.0,
                use_clouds=False, tprofile="isothermal", extra_mols=None, hifi=False):
    """Forward model for ONE planet's fixed geometry (Rp, Mp, R*, T*).
    Atmospheric θ (T + abundances [+ log P_cloud]) is set later via `spectrum`.
    use_clouds=True adds a gray SimpleClouds deck (Path B, 7th param) so the
    model can represent real WASP-39b's cloud muting.
    tprofile="npoint" replaces the isothermal profile with a 2-point vertical
    gradient (fit params T_surface, T_top) — the minimal test of whether real
    WASP-39b needs a P-T structure the isothermal model cannot represent
    (Phase-3 forward-model-misspecification probe, P3-D8)."""
    global _CACHE_SET
    import taurex.log
    taurex.log.disableLogging()
    from taurex.cache import OpacityCache, CIACache
    if not _CACHE_SET:
        OpacityCache().clear_cache()
        OpacityCache().set_opacity_path(str(HIFI_XSEC if hifi else XSEC))  # P3-D10 hi-fi swap
        CIACache().set_cia_path(str(CIA))
        _CACHE_SET = True

    from taurex.planet import Planet
    from taurex.stellar import BlackbodyStar
    from taurex.temperature import Isothermal, NPoint
    from taurex.chemistry import TaurexChemistry, ConstantGas
    from taurex.model import TransmissionModel
    from taurex.contributions import (
        AbsorptionContribution, CIAContribution, RayleighContribution,
        SimpleCloudsContribution)

    chemistry = TaurexChemistry(fill_gases=["H2", "He"], ratio=[0.17])
    for mol in MOLECULES:
        chemistry.addGas(ConstantGas(mol, mix_ratio=1e-7))
    for mol in (extra_mols or []):        # P3-D9: SO2 (opacSO2.dat, same ExoTransmit grid)
        chemistry.addGas(ConstantGas(mol, mix_ratio=1e-7))

    # isothermal (ABC default) vs a 2-point gradient. NPoint with no interior
    # points = a smooth T_surface(bottom)→T_top(top) ramp across the full column;
    # P_surface/P_top default to the atmosphere's max/min pressure.
    if tprofile == "npoint":
        temp_prof = NPoint(T_surface=1200.0, T_top=400.0)
    else:
        temp_prof = Isothermal(T=1200.0)

    tm = TransmissionModel(
        planet=Planet(planet_radius=rp_rj, planet_mass=mp_mj, albedo=0),
        temperature_profile=temp_prof,
        chemistry=chemistry,
        star=BlackbodyStar(temperature=ts, radius=rs_rsun, mass=ms_msun),
        atm_min_pressure=1e-1, atm_max_pressure=1e6, nlayers=100,
    )
    tm.add_contribution(AbsorptionContribution())
    tm.add_contribution(CIAContribution(cia_pairs=["H2-H2", "H2-He"]))
    tm.add_contribution(RayleighContribution())
    if use_clouds:
        tm.add_contribution(SimpleCloudsContribution(clouds_pressure=1e5))
    tm.build()
    return tm


def spectrum(tm, theta, wlgrid, wlwidth, tprofile="isothermal"):
    """θ → (Rp/Rs)^2 binned onto wlgrid (µm).
    isothermal: θ = [T, log_H2O, log_CO2, log_CH4, log_CO, log_NH3] (+ log10 P_cloud).
    npoint (P3-D8): θ = [T_surface, T_top, log_H2O, log_CO2, log_CH4, log_CO, log_NH3]."""
    from taurex.binning import FluxBinner
    if tprofile == "npoint":
        tm["T_surface"] = float(theta[0])
        tm["T_top"] = float(theta[1])
        for mol, logx in zip(MOLECULES, theta[2:7]):
            tm[mol] = float(10.0 ** logx)
    elif tprofile == "radius":                           # P3-D11: θ=[rp_rj, T, 5×logX]
        tm["planet_radius"] = float(theta[0])            # per-sample radius = the fix
        tm["T"] = float(theta[1])
        for mol, logx in zip(MOLECULES, theta[2:7]):
            tm[mol] = float(10.0 ** logx)
    else:
        tm["T"] = float(theta[0])
        for mol, logx in zip(MOLECULES, theta[1:6]):
            tm[mol] = float(10.0 ** logx)
        if len(theta) > 6:                               # Path B cloud param
            tm["clouds_pressure"] = float(10.0 ** theta[6])
    native_wn, rprs, _, _ = tm.model()
    native_wl = 10000.0 / native_wn                  # µm, descending
    order = np.argsort(wlgrid)
    fb = FluxBinner(np.asarray(wlgrid)[order], np.asarray(wlwidth)[order])
    _, binned, _, _ = fb.bindown(native_wl[::-1], rprs[::-1])
    out = np.empty_like(binned)
    out[order] = binned
    return out


def validate(n):
    import h5py, pandas as pd
    from taurex.constants import RJUP, MJUP, RSOL, MSOL
    base = TUT.parent / "Level2Data"
    gt = pd.read_csv(base / "Ground Truth Package/FM_Parameter_Table.csv")
    aux = pd.read_csv(base / "AuxillaryTable.csv").set_index("planet_ID")
    cols = ["planet_temp", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
    spec_h5 = glob.glob("data/abc/Level2Data/SpectralData.hdf5")[0]

    print(f"Validating forward model on {n} planets "
          f"(TauREx vs stored challenge spectrum; per-planet Rp/R*/M* from AuxillaryTable)\n")
    print(f"  {'planet':>8} {'T (K)':>7} {'mean|Δ|/depth':>13} {'mean|Δ|/σ_noise':>15}")
    print("  " + "-" * 56)
    errs, chis = [], []
    with h5py.File(spec_h5) as f:
        for pid in gt["planet_ID"].values[:n]:
            key = f"Planet_{pid}"
            if key not in f or pid not in aux.index:
                continue
            g = f[key]
            wl, ww, obs = g["instrument_wlgrid"][:], g["instrument_width"][:], g["instrument_spectrum"][:]
            noise = g["instrument_noise"][:]
            a = aux.loc[pid]
            tm = build_model(
                rp_rj=a["planet_radius_m"] / RJUP, mp_mj=a["planet_mass_kg"] / MJUP,
                rs_rsun=a["star_radius_m"] / RSOL, ms_msun=a["star_mass_kg"] / MSOL,
                ts=a["star_temperature"],
            )
            row = gt.loc[gt["planet_ID"] == pid, cols].values[0]
            model = spectrum(tm, row, wl, ww)
            rel = np.abs(model - obs) / np.abs(obs).mean()
            chi = np.abs(model - obs) / noise            # residual in units of obs noise
            errs.append(rel.mean()); chis.append(chi.mean())
            print(f"  {pid:>8} {row[0]:>7.0f} {rel.mean():>13.4f} {chi.mean():>15.2f}")
    print("  " + "-" * 56)
    print(f"  median mean-rel-error: {np.median(errs):.4f}   median |Δ|/σ_noise: {np.median(chis):.2f}")
    ok = np.median(chis) < 2.0
    print(f"  {'PASS — matches ABC to within the observational noise' if ok else 'residual exceeds noise — refine (Rp ref-pressure / binning / clouds)'}")


# WASP-39b system params (Mancini 2018 / ERS); Rp/Rs≈0.146 → depth≈0.0213 (matches data)
WASP39 = dict(rp_rj=1.27, mp_mj=0.28, rs_rsun=0.895, ms_msun=0.913, ts=5400.0)
# P5 multi-target geometries (fixed except radius, per the P3-D11 radius-model recipe):
# WASP-96b (Hellier 2014): hot Saturn, G8 star. K2-18b (Benneke 2019): cool sub-Neptune, M dwarf.
WASP96 = dict(rp_rj=1.20, mp_mj=0.49, rs_rsun=1.05, ms_msun=1.06, ts=5540.0)
K2_18b = dict(rp_rj=0.2352, mp_mj=0.0272, rs_rsun=0.4445, ms_msun=0.495, ts=3457.0)


def forward_npz(npz_path, mode="auto", planet="wasp39"):
    """WI-4 step 2: run the forward model on MIRAGE's θ samples with WASP-39b
    geometry → model spectra on the sample grid. Writes real_ess_forward.npz.
    mode="radius" (P3-D11): θ=[rp_rj,T,5×logX], isothermal, radius applied PER SAMPLE."""
    s = np.load(npz_path)
    theta, wlen = s["theta"], s["wlen"]
    wl = np.sort(wlen)                              # ascending, for bin widths
    edges = np.empty(len(wl) + 1)
    edges[1:-1] = np.sqrt(wl[:-1] * wl[1:])
    edges[0] = wl[0] ** 2 / edges[1]; edges[-1] = wl[-1] ** 2 / edges[-2]
    width_asc = np.diff(edges)
    order = np.argsort(wlen)
    wlwidth = np.empty_like(width_asc); wlwidth[order] = width_asc   # back to stored order

    radius_mode = (mode == "radius")
    use_clouds = (not radius_mode) and (theta.shape[1] > 6)          # Path B auto-detect
    geom = {"wasp39": WASP39, "wasp96": WASP96, "k218": K2_18b}[planet]   # P5 per-planet geometry
    tm = build_model(**geom, use_clouds=use_clouds)                 # radius set per-sample below
    tprof = "radius" if radius_mode else "isothermal"
    specs = np.full((len(theta), len(wlen)), np.nan)     # NaN = unphysical → 0 weight
    n_bad = 0
    for i, th in enumerate(theta):
        try:
            specs[i] = spectrum(tm, th, wlen, wlwidth, tprofile=tprof)
        except Exception:                                # e.g. abundances sum > 1
            n_bad += 1
        if (i + 1) % 100 == 0:
            print(f"  forward {i+1}/{len(theta)}")
    if n_bad:
        print(f"  {n_bad}/{len(theta)} samples unphysical (abundances sum>1) → 0 weight")
    out = npz_path.rsplit("samples", 1)[0] + "forward.npz"
    np.savez(out, model_spec=specs)
    print(f"[forward] {len(theta)} spectra (WASP-39b geometry) → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, default=0)
    ap.add_argument("--forward-npz", type=str, default=None)
    ap.add_argument("--forward-mode", default="auto", choices=["auto", "radius"])
    ap.add_argument("--forward-planet", default="wasp39", choices=["wasp39", "wasp96", "k218"])
    args = ap.parse_args()
    if args.forward_npz:
        forward_npz(args.forward_npz, args.forward_mode, args.forward_planet)
    else:
        validate(args.validate or 8)
