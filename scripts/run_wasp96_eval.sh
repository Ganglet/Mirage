#!/bin/bash
# P5 multi-target — evaluate the trained WASP-96b radius model on the REAL,
# self-reduced NIRISS spectrum (data/jwst_wasp96b_spectrum.csv, 90 bins).
# sample (mirage) -> forward per-sample radius, WASP-96b geometry (mirage-taurex)
# -> best-fit chi2 + ESS + IS-reweighted radius/T/abundances. Expected: low chi2,
# radius IS ~1.20 RJ (WASP96 truth), water-rich (the 5.8-sigma 1.4um feature).
set -u
cd "/Users/angshumansmac/Desktop/Actual Projects/Exoplanet/Project"
PYM=/Users/angshumansmac/anaconda3/envs/mirage/bin/python
PYT=/Users/angshumansmac/anaconda3/envs/mirage-taurex/bin/python
L=data/real_ess; N=${1:-2000}

echo "[wasp96] sample $(date '+%T')"
$PYM scripts/real_ess.py --sample --arm wasp96 --n $N 2>&1 | grep -aviE "anaconda-cloud|Warning|warn|enable_nested" | tail -3
echo "[wasp96] forward"
PYTHONPATH=scripts $PYT scripts/taurex_forward.py --forward-npz $L/real_ess_samples.npz \
    --forward-mode radius --forward-planet wasp96 2>&1 | grep -avE "anaconda-cloud|forward [0-9]" | tail -2
cp $L/real_ess_samples.npz $L/wasp96_samples.npz
cp $L/real_ess_forward.npz $L/wasp96_forward.npz

echo "[wasp96] compute $(date '+%T')"
$PYM - <<'PYEOF' 2>&1 | grep -avE "anaconda-cloud|Warning|warn"
import numpy as np
from pathlib import Path
OUT = Path("data/real_ess")
names = ["radius","T","logH2O","logCO2","logCH4","logCO","logNH3"]
s = np.load(OUT/"wasp96_samples.npz"); f = np.load(OUT/"wasp96_forward.npz")
th, lq = s["theta"], s["log_q"]
x, sig, cov = s["x_obs"], s["sig_obs"], s["covered"]
ms = f["model_spec"]
m = cov & np.isfinite(sig) & (sig > 0)
se = np.sqrt(sig**2 + (0.05*x)**2)                 # + forward-model systematic floor
val = np.isfinite(ms[:, m]).all(1)
r = (x[None, m] - ms[:, m]) / se[None, m]
ll = np.where(val, -0.5*np.nansum(r**2, 1), -np.inf)
logw = ll - lq; logw -= np.nanmax(logw)
w = np.exp(logw); w /= w.sum()
ess = w.sum()**2 / np.sum(w**2)
bc = -2*ll.max() / m.sum()
print(f"\n=== WASP-96b real-data retrieval ===")
print(f"  covered bins = {m.sum()}/{len(cov)}   N = {len(th)}")
print(f"  best-fit chi2/dof = {bc:.3f}    ESS = {ess:.1f}")
def wq(v, q): i = np.argsort(v); return np.interp(q, np.cumsum(w[i]), v[i])
print(f"\n  {'param':<8}{'IS-mean':>10}{'IS-16%':>10}{'IS-84%':>10}   (proposal mean)")
for i, nm in enumerate(names):
    ismean = np.sum(w*th[:, i])
    print(f"  {nm:<8}{ismean:>10.3f}{wq(th[:,i],0.16):>10.3f}{wq(th[:,i],0.84):>10.3f}   {th[:,i].mean():>8.3f}")
print(f"\n  WASP-96b expected: radius~1.20 RJ (geometry truth), water-rich (1.4um at 5.8 sigma).")
print(f"  READ: low chi2 + radius IS near 1.20 + H2O elevated => method transfers to a")
print(f"        2nd hot Saturn on an INDEPENDENT instrument (NIRISS SOSS) + self-reduction.")
PYEOF
echo "[wasp96] done $(date '+%T')"
