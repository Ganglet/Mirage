#!/bin/bash
# P3-D10 — hi-fi forward-model NS probe (GREEN GATE before any retrain).
# ExoMolOP R=15000 opacities (latest line lists) for all 6 molecules incl SO2 —
# the fullest, highest-fidelity forward model. Two parallel probes, 300 live
# (matches the lo-fi references for a clean χ² comparison):
#   hifi      : isothermal + SO2, cold-ALLOWED  -> does hi-fi spontaneously go physical?
#   hifi_hot  : isothermal + SO2, T forced hot  -> best physical χ² vs lo-fi hot_iso=38.4
# ~1.9s/eval at R15000 => a few hours each. Refs (ExoTransmit lo-fi):
#   cold_iso 2.55, hot_iso 38.44, so2 2.56, so2_hot 38.46.
set -u
cd "/Users/angshumansmac/Desktop/Actual Projects/Exoplanet/Project"
PY=/Users/angshumansmac/anaconda3/envs/mirage-taurex/bin/python
export PYTHONPATH=scripts
L=data/real_ess
run() { OMP_NUM_THREADS=5 NUMBA_NUM_THREADS=5 "$PY" scripts/taurex_retrieve.py "$@"; }

echo "[hifi] launching 2 parallel probes $(date '+%T')"
run --live 300 --hifi --so2 --tag hifi                         > "$L/ns_hifi.log"     2>&1 &
run --live 300 --hifi --so2 --tmin 700 --tmax 1800 --tag hifi_hot > "$L/ns_hifi_hot.log" 2>&1 &
wait
echo "[hifi] done $(date '+%T')"

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
    return float(d["chi2"]), float(T), names, np.round(wm,2)
print("\n============ HI-FI FORWARD MODEL TEST (ExoMolOP R=15000) ============")
print(f"{'probe':<10}{'T':>7}{'chi2/N':>9}   note")
print(f"{'cold_iso':<10}{114:>7.0f}{2.55:>9.2f}   lo-fi ref")
print(f"{'hot_iso':<10}{700:>7.0f}{38.44:>9.2f}   lo-fi ref (forced)")
print(f"{'so2':<10}{114:>7.0f}{2.56:>9.2f}   lo-fi+SO2 ref")
for tag in ("hifi","hifi_hot"):
    r = load(tag)
    if r is None: print(f"{tag:<10} FAILED — see ns_{tag}.log"); continue
    chi2,T,names,wm = r
    print(f"{tag:<10}{T:>7.0f}{chi2:>9.2f}   HI-FI")
    print(f"           params {names}")
    print(f"           wmean  {list(wm)}")
print("====================================================================")
print("READ:")
print(" hifi (cold-allowed): T>600 & chi2 low   => HI-FI FLIPS physical = GREEN, retrain")
print("                      T~114 still cold    => fidelity is NOT the fix either")
print(" hifi_hot vs hot_iso: chi2 << 38.4        => hi-fi opacities help the physical fit")
print("                      chi2 ~ 38.4         => no improvement from better line lists")
PYEOF
echo "[hifi] summary complete $(date '+%T')"
