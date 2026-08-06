#!/bin/bash
# P3-D8 overnight NS ruling-out matrix. Runs the local-data forward-model probes
# in PARALLEL (each caps threads so 3×3=9 ≤ 10 cores), then prints a χ² summary.
# The two "hot" probes FORCE a physical T range to quantify how badly a
# non-cold-flat solution fits (the core misspecification number). grad_cloud
# tests the one physics combination not yet tried. Cold refs already computed
# (cold_iso χ²≈2.55, cold_grad χ²=2.65).
set -u
cd "/Users/angshumansmac/Desktop/Actual Projects/Exoplanet/Project"
PY=/Users/angshumansmac/anaconda3/envs/mirage-taurex/bin/python
export PYTHONPATH=scripts
L=data/real_ess
mkdir -p "$L"

run() { OMP_NUM_THREADS=3 NUMBA_NUM_THREADS=3 "$PY" scripts/taurex_retrieve.py "$@"; }

echo "[matrix] launching 3 parallel probes $(date '+%T')"
run --live 300 --tmin 700 --tmax 1800 --tag hot_iso             > "$L/ns_hot_iso.log"    2>&1 &
run --live 300 --tprofile npoint --tmin 700 --tmax 1800 --tag hot_grad > "$L/ns_hot_grad.log" 2>&1 &
run --live 300 --tprofile npoint --clouds --tag grad_cloud      > "$L/ns_grad_cloud.log" 2>&1 &
wait
echo "[matrix] all probes done $(date '+%T')"

"$PY" - <<'PYEOF' 2>&1 | grep -v anaconda-cloud
import numpy as np
from pathlib import Path
OUT = Path("data/real_ess")
rows = [("cold_iso","(prior)","~2.55 (documented)"),
        ("cold_grad","(prior)", None)]
def load(tag):
    f = OUT / f"wasp39b_ns_posterior_{tag}.npz"
    if not f.exists(): return None
    d = np.load(f, allow_pickle=True)
    w = d["weights"]/d["weights"].sum()
    wm = np.average(d["samples"],axis=0,weights=w)
    names = list(d["fit_names"])
    tsurf = wm[names.index("T_surface")] if "T_surface" in names else wm[names.index("T")]
    return float(d["chi2"]), float(tsurf), names, np.round(wm,2)
print("\n================ NS RULING-OUT MATRIX ================")
print(f"{'probe':<12}{'T(_surface)':>12}{'chi2/N':>10}")
print(f"{'cold_iso':<12}{114:>12.0f}{2.55:>10.2f}   (documented reference)")
print(f"{'cold_grad':<12}{117:>12.0f}{2.65:>10.2f}   (this session)")
for tag in ("hot_iso","hot_grad","grad_cloud"):
    r = load(tag)
    if r is None: print(f"{tag:<12}{'FAILED — see ns_'+tag+'.log':>12}"); continue
    chi2, tsurf, names, wm = r
    print(f"{tag:<12}{tsurf:>12.0f}{chi2:>10.2f}")
    print(f"             params {names}")
    print(f"             wmean  {list(wm)}")
print("=====================================================")
print("READ: if hot_* chi2 >> cold_* chi2, physical fits lose to cold-flat")
print("      => misspecification confirmed; forward model needs SO2/opacities,")
print("      not P-T structure or clouds. If any hot_* chi2 ~ cold, that lever helps.")
PYEOF
echo "[matrix] summary complete $(date '+%T')"
