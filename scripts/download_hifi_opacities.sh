#!/bin/bash
# P3-D10 — fetch ExoMolOP R=15000 TauREx cross-sections for all 6 molecules
# (latest ExoMol/HITEMP line lists), replacing ABC's older ExoTransmit .dat set.
# Resumable (-C -) + retries; verifies each file's size against the server.
set -u
cd "/Users/angshumansmac/Desktop/Actual Projects/Exoplanet/Project"
DEST=data/opacity_hifi
mkdir -p "$DEST"
B=https://www.exomol.com/db

declare -a F=(
"H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL__R15000_0.3-50mu.xsec.TauREx.h5"
"CO2/12C-16O2/UCL-4000/12C-16O2__UCL-4000.R15000_0.3-50mu.xsec.TauREx.h5"
"CH4/12C-1H4/YT34to10/12C-1H4__YT34to10.R15000_0.3-50mu.xsec.TauREx.h5"
"CO/12C-16O/Li2015/12C-16O__Li2015.R15000_0.3-50mu.xsec.TauREx.h5"
"NH3/14N-1H3/CoYuTe/14N-1H3__CoYuTe.R15000_0.3-50mu.xsec.TauREx.h5"
"SO2/32S-16O2/ExoAmes/32S-16O2__ExoAmes.R15000_0.3-50mu.xsec.TauREx.h5"
)

for path in "${F[@]}"; do
  url="$B/$path"
  fname=$(basename "$path")
  out="$DEST/$fname"
  echo "[dl] $(date '+%T') $fname"
  curl -sL -C - --retry 8 --retry-delay 4 --max-time 1200 -o "$out" "$url"
  remote=$(curl -sIL --max-time 30 "$url" | grep -i content-length | tail -1 | tr -dc '0-9')
  local=$(stat -f%z "$out" 2>/dev/null || echo 0)
  if [ "$local" = "$remote" ] && [ -n "$remote" ]; then
    echo "    OK $((local/1048576)) MB (verified)"
  else
    echo "    !! SIZE MISMATCH local=$local remote=$remote — re-run to resume"
  fi
done
echo "[dl] all done $(date '+%T')"
ls -la "$DEST"
