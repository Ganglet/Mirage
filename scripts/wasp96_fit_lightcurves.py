"""
WASP-96b Stage 4 -- transmission spectrum from exoTEDRF box order-1 light curves.

Reads <target>_box_spectra_fullres.fits (Wave O1 / Flux O1 / Flux Err O1 / Time).
White-light fit: free [Rp/Rs, t0, a/Rs, inc, q1, q2, quad-baseline] with physical
Kipping (2013) limb darkening -> the system solution + LD. Per bin: geometry, t0
and LD fixed to the white-light values; free [Rp/Rs, quad-baseline]. Errors are
inflated to red-chi2=1 (photon-only bars underestimate real scatter) for honest depths.

    conda run -n mirage-reduce python scripts/wasp96_fit_lightcurves.py --checkpoint  # white-light only
    conda run -n mirage-reduce python scripts/wasp96_fit_lightcurves.py               # full spectrum

WASP-96b ephemeris (ERO 2734): P=3.42525650 d; a/Rs~8.84, i~85.6, e=0 (fit here).
"""
import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.optimize import least_squares
import batman

OUTDIR = Path("pipeline_outputs_directory_wasp96b")
P_ORB, ECC, WW = 3.42525650, 0.0, 90.0
ARS0, INC0 = 8.84, 85.6                       # literature initial geometry
DEPTH_LIT = 0.0142                            # (Rp/Rs)^2 white-light truth for the checkpoint
T14_HALF = 0.0554                             # half transit duration (d), for in/out masking


def find_fullres():
    hits = list(OUTDIR.rglob("*box_spectra_fullres.fits"))
    if not hits:
        raise FileNotFoundError(f"no *box_spectra_fullres.fits under {OUTDIR}")
    return sorted(hits)[-1]


def load():
    f = find_fullres()
    print(f"[load] {f}")
    with fits.open(f) as h:
        print("  extensions:", [e.name for e in h if e.name])
        wave = np.asarray(h["Wave O1"].data, float)
        flux = np.asarray(h["Flux O1"].data, float)
        ferr = np.asarray(h["Flux Err O1"].data, float)
        t = np.asarray(h["Time"].data, float)
    o = np.argsort(wave)
    return wave[o], flux[:, o], ferr[:, o], t


def kipping_u(q1, q2):
    """Kipping (2013) q1,q2 in [0,1] -> physical quadratic u1,u2."""
    sq = np.sqrt(q1)
    return 2 * sq * q2, sq * (1 - 2 * q2)


def lc(t, rp, t0, ars, inc, u1, u2, base):
    pm = batman.TransitParams()
    pm.t0, pm.per, pm.rp, pm.a, pm.inc, pm.ecc, pm.w = t0, P_ORB, rp, ars, inc, ECC, WW
    pm.limb_dark, pm.u = "quadratic", [u1, u2]
    return batman.TransitModel(pm, t).light_curve(pm) * base


def _inflate_refit(resid_fn, x, err, dof, nfev):
    """Two-pass: fit, inflate err by sqrt(red-chi2), refit; return result + inflated err."""
    r = least_squares(resid_fn, x, args=(err,), bounds=resid_fn.bounds,
                      method="trf", max_nfev=nfev)
    red = float(np.sum(r.fun ** 2)) / dof
    err_inf = err * np.sqrt(max(red, 1.0))
    r = least_squares(resid_fn, r.x, args=(err_inf,), bounds=resid_fn.bounds,
                      method="trf", max_nfev=nfev)
    return r, err_inf, red


def fit_white(t, y, yerr):
    dt = t - t.mean()
    tc0 = t[np.argmin(y)]

    def resid(p, e):
        rp, t0, ars, inc, q1, q2, b0, b1, b2 = p
        u1, u2 = kipping_u(q1, q2)
        return (y - lc(t, rp, t0, ars, inc, u1, u2, b0 + b1 * dt + b2 * dt ** 2)) / e
    resid.bounds = ([0.05, tc0 - 0.05, 6.0, 82.0, 0, 0, 0.5 * np.median(y), -np.inf, -np.inf],
                    [0.20, tc0 + 0.05, 12.0, 89.9, 1, 1, 1.5 * np.median(y), np.inf, np.inf])
    p0 = [0.119, tc0, ARS0, INC0, 0.3, 0.3, np.median(y), 0.0, 0.0]
    r, err_inf, red = _inflate_refit(resid, p0, yerr, len(t) - 9, 20000)
    rp, t0, ars, inc, q1, q2, b0, b1, b2 = r.x
    u1, u2 = kipping_u(q1, q2)
    try:
        rp_err = float(np.sqrt(np.linalg.inv(r.jac.T @ r.jac)[0, 0]))
    except np.linalg.LinAlgError:
        rp_err = np.nan
    res = r.fun * err_inf
    oot = np.abs(((t - t0 + 0.5 * P_ORB) % P_ORB) - 0.5 * P_ORB) > T14_HALF
    return dict(rp=rp, rp_err=rp_err, t0=t0, ars=ars, inc=inc, q1=q1, q2=q2, u1=u1, u2=u2,
                b0=b0, b1=b1, b2=b2, depth=rp ** 2, depth_err=2 * rp * rp_err, redchi2=red,
                rms_ppm=np.std(res) * 1e6, oot_ppm=np.std(res[oot]) * 1e6,
                in_ppm=np.std(res[~oot]) * 1e6)


def fit_bin(t, y, yerr, sol, rp0=0.119):
    """Geometry (a/Rs, inc, t0) + LD fixed to white-light; free [rp, b0, b1, b2]."""
    dt = t - t.mean()

    def resid(p, e):
        rp, b0, b1, b2 = p
        return (y - lc(t, rp, sol["t0"], sol["ars"], sol["inc"], sol["u1"], sol["u2"],
                       b0 + b1 * dt + b2 * dt ** 2)) / e
    resid.bounds = ([0.02, 0.5 * np.median(y), -np.inf, -np.inf],
                    [0.25, 1.5 * np.median(y), np.inf, np.inf])
    p0 = [rp0, np.median(y), 0.0, 0.0]
    r, _, _ = _inflate_refit(resid, p0, yerr, len(t) - 4, 5000)
    try:
        rp_err = float(np.sqrt(np.linalg.inv(r.jac.T @ r.jac)[0, 0]))
    except np.linalg.LinAlgError:
        rp_err = np.nan
    return r.x[0], rp_err


def whitelight(flux, ferr, good):
    y = np.nansum(flux[:, good], axis=1)
    yn = y / np.nanmedian(y)
    ye = np.sqrt(np.nansum(ferr[:, good] ** 2, axis=1)) / np.nanmedian(y)
    return yn, np.where(np.isfinite(ye) & (ye > 0), ye, np.nanmedian(ye))


def main(checkpoint, nbins):
    wave, flux, ferr, t = load()
    print(f"[data] {flux.shape[0]} integrations x {flux.shape[1]} wavelengths   "
          f"{wave.min():.3f}-{wave.max():.3f}um")
    good = np.isfinite(flux).all(axis=0) & (np.nanmedian(flux, axis=0) > 0)
    yn, ye = whitelight(flux, ferr, good)
    sol = fit_white(t, yn, ye)
    print("\n=== WHITE-LIGHT FIT (Kipping LD + free geometry) ===")
    print(f"  Rp/Rs = {sol['rp']:.4f} +/- {sol['rp_err']:.4f}   depth = {sol['depth']:.5f} "
          f"({sol['depth']*1e2:.3f} %)   truth ~{DEPTH_LIT:.4f}")
    print(f"  a/Rs  = {sol['ars']:.2f}   inc = {sol['inc']:.2f}   u1={sol['u1']:.3f} u2={sol['u2']:.3f}")
    print(f"  RMS = {sol['rms_ppm']:.0f} ppm   OOT = {sol['oot_ppm']:.0f} ppm   "
          f"in-transit = {sol['in_ppm']:.0f} ppm   red-chi2(photon)={sol['redchi2']:.1f}")
    off = abs(sol["depth"] - DEPTH_LIT)
    print(f"  |depth - truth| = {off:.5f}   {'PASS' if off < 0.004 else 'CHECK'}")
    np.save(OUTDIR / "white_light_fit.npy", sol, allow_pickle=True)
    if checkpoint:
        print("\n[checkpoint] white-light only; stopping before per-bin spectrum.")
        return

    print(f"\n=== PER-BIN SPECTRUM ({nbins} bins, geometry+LD fixed to white-light) ===")
    edges = np.geomspace(wave.min(), wave.max(), nbins + 1)
    ctr, depth, depth_err = [], [], []
    for k in range(nbins):
        m = good & (wave >= edges[k]) & (wave < edges[k + 1])
        if m.sum() < 3:
            continue
        yb, yeb = whitelight(flux, ferr, m)
        rp, rp_err = fit_bin(t, yb, yeb, sol)
        if not np.isfinite(rp_err):
            continue
        ctr.append(np.sqrt(edges[k] * edges[k + 1]))
        depth.append(rp ** 2)
        depth_err.append(2 * rp * rp_err)
    ctr, depth, depth_err = map(np.asarray, (ctr, depth, depth_err))
    print(f"  fitted {len(ctr)} bins   depth median={np.median(depth):.5f}   "
          f"median err={np.median(depth_err)*1e6:.0f} ppm")

    import pandas as pd
    out = Path("data/jwst_wasp96b_spectrum.csv")
    pd.DataFrame({"wavelength_um": ctr, "transit_depth": depth,
                  "depth_error": depth_err}).to_csv(out, index=False)
    print(f"  wrote {out}  ({len(ctr)} bins)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="store_true")
    ap.add_argument("--nbins", type=int, default=90)
    a = ap.parse_args()
    main(a.checkpoint, a.nbins)
