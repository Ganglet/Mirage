"""
Zip the WASP-96b and K2-18b OOT data packages for sharing.
Produces:
  Downloads/WASP96b_out_of_transit_data.zip
  Downloads/K2_18b_out_of_transit_data.zip
"""
import zipfile
from pathlib import Path

PACKAGES = {
    "WASP96b_out_of_transit_data": Path("output/wasp96b_oot_data"),
    "K2_18b_out_of_transit_data":  Path("output/k2_18b_oot_data"),
}

for zip_name, pkg_dir in PACKAGES.items():
    if not pkg_dir.exists():
        print(f"WARNING: {pkg_dir} not found — run extract_wasp96b_k2_18b_oot.py first")
        continue
    out_zip = Path.home() / "Downloads" / f"{zip_name}.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(pkg_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(pkg_dir))
    size_mb = round(out_zip.stat().st_size / 1e6, 1)
    print(f"✓ {out_zip.name}  ({size_mb} MB)")