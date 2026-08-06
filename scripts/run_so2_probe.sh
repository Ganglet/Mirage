#!/bin/bash
# P3-D9 — SO2 lever test. opacSO2.dat (ExoTransmit, same grid as the other 5) now
# in the xsec cache. Two parallel NS probes decide if SO2 is the missing physics:
#   so2      : isothermal + SO2, cold-ALLOWED  -> does NS spontaneously go physical?
#   so2_hot  : isothermal + SO2, T forced hot  -> does SO2 rescue the physical fit
#              (compare χ² to no-SO2 hot_iso = 38.4)?
# Refs: cold_iso χ²=2.55, cold_grad 2.65, grad_cloud 2.62, hot_iso 38.4, hot_grad 38.5.
set -u
cd "/Users/angshumansmac/Desktop/Actual Projects/Exoplanet/Project"
PY=/Users/angshumansmac/anaconda3/envs/mirage-taurex/bin/python
export PYTHONPATH=scripts
L=data/real_ess
run() { OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 "$PY" scripts/taurex_retrieve.py "$@"; }

echo "[so2] launching 2 parallel probes $(date '+%T')"
run --live 300 --so2 --tag so2                        > "$L/ns_so2.log"     2>&1 &
run --live 300 --so2 --tmin 700 --tmax 1800 --tag so2_hot > "$L/ns_so2_hot.log" 2>&1 &
wait
echo "[so2] done $(date '+%T')"

"$PY" - <<'PYEOF' 2>&1 | grep -v anaconda-cloud
import numpy as np
from pathlib import Path
OUT = Path("data/real_ess")
def load(tag):
    f = OUT / f"wasp39b_ns_posterior_{tag}.npz"
    if not f.exists(): return None
    d = np.load(f, allow_pickle=True)
    w = d["weights"]/d["weights"].sum()
    wm = np.average(d["samples"],axis=0,weights=w)
    names = list(d["fit_names"])
    T = wm[names.index("T")] if "T" in names else wm[names.index("T_surface")]
    so2 = wm[names.index("log_SO2")] if "log_SO2" in names else float("nan")
    return float(d["chi2"]), float(T), float(so2), names, np.round(wm,2)
print("\n================ SO2 LEVER TEST ================")
print(f"{'probe':<10}{'T':>7}{'log_SO2':>9}{'chi2/N':>9}")
print(f"{'cold_iso':<10}{114:>7.0f}{'--':>9}{2.55:>9.2f}   ref (no SO2)")
print(f"{'hot_iso':<10}{700:>7.0f}{'--':>9}{38.44:>9.2f}   ref (no SO2, forced hot)")
for tag in ("so2","so2_hot"):
    r = load(tag)
    if r is None: print(f"{tag:<10} FAILED — see ns_{tag}.log"); continue
    chi2,T,so2,names,wm = r
    print(f"{tag:<10}{T:>7.0f}{so2:>9.2f}{chi2:>9.2f}")
    print(f"           params {names}")
    print(f"           wmean  {list(wm)}")
print("===============================================")
print("READ:")
print(" so2 (cold-allowed): T>600 & chi2~cold  => SO2 FLIPS it physical = GREEN, retrain")
print("                     T~114 still cold    => SO2 alone insufficient")
print(" so2_hot vs hot_iso: chi2 << 38.4        => SO2 helps the physical fit (partial win)")
print("                     chi2 ~ 38.4         => SO2 does not close the gap")
PYEOF
echo "[so2] summary complete $(date '+%T')"
