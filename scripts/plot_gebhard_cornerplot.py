import h5py
import numpy as np
import corner
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/gebhard")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

PARAM_NAMES = [
    "C/O", "Fe/H", "log_P_quench", "S_eq_Fe", "S_eq_MgSiO3",
    "f_sed", "log_K_zz", "sigma_g", "log_g", "R_P",
    "T_int", "T_3", "T_2", "T_1", "alpha", "log_delta"
]

# Ground truth (theta_0 from Vasist 2023)
THETA_0 = np.array([
    0.55, 0.0, -5.0, -0.86, -0.86, 3.0, 8.5, 2.0,
    3.75, 1.0, 1000.0, 0.59, 0.29, 0.18, 1.39, 0.32
])

# Subset to match Fig. 3 in Gebhard 2025
SUBSET_IDX = [0, 2, 3, 5, 6, 12]
SUBSET_NAMES = [PARAM_NAMES[i] for i in SUBSET_IDX]
SUBSET_TRUTHS = [THETA_0[i] for i in SUBSET_IDX]

print("Loading fmpe.hdf...")
with h5py.File(DATA_DIR / "fmpe.hdf", "r") as f:
    samples = f["samples"][:]        # (1048576, 16)
    weights = f["weights"][:]        # (1048576,)
    n_eff = float(f["n_eff"][()])
    eff = float(f["sampling_efficiency"][()])

print(f"  Samples: {samples.shape}")
print(f"  ESS:     {n_eff:.1f}")
print(f"  ε:       {100*eff:.2f}%")

# Subsample using IS weights for the corner plot (corner doesn't support weights natively)
rng = np.random.default_rng(42)
idx = rng.choice(len(samples), size=10_000, replace=True, p=weights / weights.sum())
samples_is = samples[idx][:, SUBSET_IDX]

print("\nPlotting corner plot...")
fig = corner.corner(
    samples_is,
    labels=SUBSET_NAMES,
    truths=SUBSET_TRUTHS,
    truth_color="black",
    color="#033696",
    show_titles=True,
    title_kwargs={"fontsize": 10},
    label_kwargs={"fontsize": 10},
    smooth=1.0,
    bins=50,
)

fig.suptitle(
    f"FMPE with IS — noise-free, σ=0.125754\nESS={n_eff:.0f}, ε={100*eff:.1f}%",
    y=1.01, fontsize=11
)

out_path = OUT_DIR / "gebhard_fmpe_cornerplot.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
