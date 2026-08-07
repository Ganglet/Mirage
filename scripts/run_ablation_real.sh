#!/bin/bash
# Phase 4.1/4.2 — noise-conditioning ablation on the RADIUS model, real WASP-39b.
# For each arm (nocond/sigma/cov): sample from the trained model on the real input,
# forward-model (per-sample radius), save per-arm. Then compare best-fit χ² + how
# well each arm's posterior matches the independent NS anchor (coverage 7/7).
# Tests whether covariance noise-conditioning helps the REAL retrieval (project thesis).
set -u
cd "/Users/angshumansmac/Desktop/Actual Projects/Exoplanet/Project"
PYM=/Users/angshumansmac/anaconda3/envs/mirage/bin/python
PYT=/Users/angshumansmac/anaconda3/envs/mirage-taurex/bin/python
L=data/real_ess; N=1000

for arm in rad_nocond rad_sigma rad; do
  echo "[$arm] sample $(date '+%T')"
  $PYM scripts/real_ess.py --sample --arm $arm --n $N 2>&1 | grep -aviE "anaconda-cloud|Warning|warn|enable_nested" | tail -2
  echo "[$arm] forward"
  $PYT scripts/taurex_forward.py --forward-npz $L/real_ess_samples.npz --forward-mode radius 2>&1 | grep -avE "anaconda-cloud|forward [0-9]" | tail -1
  cp $L/real_ess_samples.npz $L/abl_${arm}_samples.npz
  cp $L/real_ess_forward.npz $L/abl_${arm}_forward.npz
done

echo "[compare] $(date '+%T')"
$PYM - <<'PYEOF' 2>&1 | grep -avE "anaconda-cloud|Warning|warn"
import numpy as np
from pathlib import Path
OUT = Path("data/real_ess")
names = ["radius","T","logH2O","logCO2","logCH4","logCO","logNH3"]
# NS anchor means (fitrad)
d = np.load(OUT/"wasp39b_ns_posterior_fitrad.npz", allow_pickle=True)
o = ["planet_radius","T","log_H2O","log_CO2","log_CH4","log_CO","log_NH3"]
idx = [list(d["fit_names"]).index(k) for k in o]
wn = d["weights"]/d["weights"].sum(); nsm = np.sum(wn[:,None]*d["samples"][:,idx],0)
def wq(v,w,q): i=np.argsort(v); return np.interp(q,np.cumsum(w[i]),v[i])
print(f"\n{'arm':<12}{'best χ²/dof':>12}{'cover/7':>9}{'radius(IS)':>12}{'logH2O(IS)':>12}")
print("-"*57)
for arm in ["rad_nocond","rad_sigma","rad"]:
    s=np.load(OUT/f"abl_{arm}_samples.npz"); f=np.load(OUT/f"abl_{arm}_forward.npz")
    th,lq,x,sig,cov,ms = s["theta"],s["log_q"],s["x_obs"],s["sig_obs"],s["covered"],f["model_spec"]
    m=cov&np.isfinite(sig)&(sig>0); se=np.sqrt(sig**2+(0.05*x)**2)
    val=np.isfinite(ms[:,m]).all(1); r=(x[None,m]-ms[:,m])/se[None,m]
    ll=np.where(val,-0.5*np.nansum(r**2,1),-np.inf)
    w=np.exp(ll-lq-np.nanmax(ll-lq)); w/=w.sum()
    bc=-2*ll.max()/m.sum()
    cvr=sum(wq(th[:,i],w,0.16)<=nsm[i]<=wq(th[:,i],w,0.84) for i in range(7))
    Rw=np.sum(w*th[:,0]); Hw=np.sum(w*th[:,2])
    lbl={"rad_nocond":"nocond","rad_sigma":"sigma","rad":"cov"}[arm]
    print(f"{lbl:<12}{bc:>12.3f}{cvr:>7}/7{Rw:>12.3f}{Hw:>12.2f}")
print("-"*57)
print("NS anchor: radius=1.23  logH2O=-3.25   (truth radius=1.27)")
print("READ: higher cover/7 + lower best-χ² for cov vs sigma/nocond => noise-conditioning")
print("      helps the REAL retrieval (project thesis holds end-to-end on real data).")
PYEOF
echo "[done] $(date '+%T')"
