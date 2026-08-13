"""
WASP-96b reduction setup: inspect the raw NIRISS SOSS uncal headers, isolate the
science TSO segments (drop the short 02101 exposures), stage a clean input dir,
and emit the exoTEDRF run_DMS config with the correct integration counts.

    conda run -n mirage-reduce python scripts/wasp96_setup_reduction.py
"""
from pathlib import Path
import shutil
import numpy as np
from astropy.io import fits

RAW = Path("data/jwst_wasp96b_raw")
SCI = Path("data/jwst_wasp96b_sci")
EXOTEDRF_FILES = Path("/Users/angshumansmac/anaconda3/envs/mirage-reduce/lib/python3.11/site-packages/exotedrf/files")
CFG = Path("configs_reduce/wasp96b_dms.yaml")


def scan():
    rows = []
    for f in sorted(RAW.glob("*_uncal.fits")):
        h = fits.getheader(f, 0)
        rows.append(dict(file=f.name,
                         exp=f.name.split("_")[1],       # 02101 / 04101
                         segn=f.name.split("-")[-1].split("_")[0],
                         filt=h.get("FILTER"), pupil=h.get("PUPIL"),
                         exptype=h.get("EXP_TYPE"), subarray=h.get("SUBARRAY"),
                         nints=h.get("NINTS"), ngroups=h.get("NGROUPS"),
                         tframe=h.get("TFRAME"), effinttm=h.get("EFFINTTM")))
    return rows


def main():
    rows = scan()
    print("=== all uncal segments ===")
    for r in rows:
        print(f"  {r['file']}  exp={r['exp']} {r['segn']}  FILTER={r['filt']} PUPIL={r['pupil']}"
              f"  {r['exptype']} {r['subarray']}  NINTS={r['nints']} NGROUPS={r['ngroups']} EFFINTTM={r['effinttm']}")

    # science TSO = the exposure group with the most total integrations (the long transit series)
    from collections import defaultdict
    by_exp = defaultdict(list)
    for r in rows:
        by_exp[r["exp"]].append(r)
    exp_nints = {e: sum(int(r["nints"] or 0) for r in v) for e, v in by_exp.items()}
    sci_exp = max(exp_nints, key=exp_nints.get)
    print(f"\nscience exposure = {sci_exp}  (total NINTS across exposures: {exp_nints})")

    sci_rows = sorted(by_exp[sci_exp], key=lambda r: r["segn"])
    total_nints = sum(int(r["nints"]) for r in sci_rows)
    subarray = sci_rows[0]["subarray"]
    print(f"science segments ({len(sci_rows)}): total NINTS={total_nints}  SUBARRAY={subarray}")

    # stage a clean input dir with only the science segments
    if SCI.exists():
        shutil.rmtree(SCI)
    SCI.mkdir(parents=True)
    for r in sci_rows:
        (SCI / r["file"]).symlink_to((RAW / r["file"]).resolve())

    # baseline (out-of-transit) ints: WASP-96b transit ~2.4h in a ~6h series -> generous
    # ~30% pre + ~30% post as OOT baseline for normalisation / 1/f scaling.
    nb = int(round(0.30 * total_nints))
    baseline = [nb, -nb]
    bg = EXOTEDRF_FILES / ("model_background256.npy" if "256" in (subarray or "") else "model_background96.npy")

    CFG.parent.mkdir(parents=True, exist_ok=True)
    cfg = f"""# exoTEDRF run_DMS config -- WASP-96b NIRISS/SOSS (JWST ERO 2734), box order-1
crds_cache_path : '{RAW.resolve()}/crds_cache/'
input_dir : '{SCI.resolve()}/'
input_filetag : 'uncal'
observing_mode : 'NIRISS/SOSS'
filter_detector : 'CLEAR'

DQInitStep : 'run'
EmiCorrStep : 'skip'
SuperBiasStep : 'run'
RefPixStep : 'run'
DarkCurrentStep : 'skip'
OneOverFStep_grp : 'run'
LinearityStep : 'run'
JumpStep : 'run'
RampFitStep : 'run'
GainScaleStep : 'run'
hot_pixel_map : None
saturation_threshold : 80
superbias_method : 'crds'
soss_background_file : '{bg}'
oof_method : 'scale-achromatic'
f277w : None
soss_timeseries : None
soss_timeseries_o2 : None
outlier_maps : None
soss_inner_mask_width : 40
soss_outer_mask_width : 70
nirspec_mask_width : 16
miri_drop_groups : 12
flag_up_ramp : False
jump_threshold : 15
flag_in_time : True
time_jump_threshold : 10
stage1_kwargs : {{}}

AssignWCSStep : 'run'
FlatFieldStep : 'run'
BackgroundStep : 'run'
OneOverFStep_int : 'skip'
BadPixStep : 'run'
PCAReconstructStep : 'run'
miri_trace_width : 20
miri_background_width : 14
miri_background_method : 'median'
space_outlier_threshold : 15
time_outlier_threshold : 10
pca_components : 10
remove_components : None
generate_lc : True
stage2_kwargs : {{}}

extract_method : 'box'
extract_width : 30
extract_width_soss2 : None
soss_specprofile : None
centroids : None
deepframe : None
st_teff : 5540
st_logg : 4.4
st_met : 0.14
planet_letter : 'b'
stage3_kwargs : {{}}

output_tag : 'wasp96b'
run_stages : [1, 2, 3]
save_results : True
force_redo : False
baseline_ints : {baseline}
do_plots : False
"""
    CFG.write_text(cfg)
    print(f"\nbaseline_ints = {baseline}  (30% pre/post of {total_nints})")
    print(f"background model = {bg.name}")
    print(f"wrote config -> {CFG}")
    print(f"staged science input -> {SCI}/ ({len(sci_rows)} segments)")


if __name__ == "__main__":
    main()
