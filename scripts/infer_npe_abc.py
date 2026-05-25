"""
Run NPE inference on a held-out ABC test spectrum and produce a corner plot.

Loads checkpoints/abc_npe_best.pt, samples 10,000 posterior samples for one
test planet, and overlays the ground-truth nested sampling posterior from
Level2Data/Ground Truth Package/Tracedata.hdf5.

Usage (run from Project/):
    python scripts/infer_npe_abc.py [--planet-idx N]

--planet-idx: index into abc_test.hdf (default 0).
"""

import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import corner
import matplotlib.pyplot as plt
from pathlib import Path
from lampe.inference import NPE

ABC_DIR = Path("data/abc")
TRACEDATA = ABC_DIR / "Level2Data/Ground Truth Package/Tracedata.hdf5"
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

PARAM_NAMES = ["T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
DIM_THETA = 6
DIM_X = 52
DIM_EMBED = 64
N_SAMPLES = 10_000


class NPEModel(nn.Module):
    """Must match architecture in train_npe_abc.py exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(DIM_X, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, DIM_EMBED), nn.ELU(),
        )
        self.npe = NPE(DIM_THETA, DIM_EMBED)

    def forward(self, theta, x):
        return self.npe(theta, self.embedding(x))

    def flow(self, x):
        return self.npe.flow(self.embedding(x))


def main(planet_idx: int = 0) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test spectrum
    print(f"Loading test planet index {planet_idx} from abc_test.hdf ...")
    with h5py.File(ABC_DIR / "abc_test.hdf", "r") as f:
        flux = torch.tensor(f["flux"][planet_idx], dtype=torch.float32)
        theta_true = f["theta"][planet_idx]
        planet_id = int(f["planet_id"][planet_idx])

    print(f"  Planet ID: {planet_id}")
    print(f"  True theta: {dict(zip(PARAM_NAMES, theta_true))}")

    # Load NPE checkpoint
    ckpt_path = Path("checkpoints/abc_npe_best.pt")
    print(f"Loading NPE checkpoint from {ckpt_path} ...")
    model = NPEModel().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Sample posterior
    print(f"Sampling {N_SAMPLES} posterior samples ...")
    x = flux.unsqueeze(0).to(device)   # (1, 52)
    with torch.no_grad():
        flow = model.flow(x)
        samples_npe = flow.sample((N_SAMPLES,)).squeeze(1).cpu().numpy()  # (N_SAMPLES, 6)

    print(f"  Samples shape: {samples_npe.shape}")

    # Load ground-truth nested sampling posterior
    print(f"Loading ground-truth posterior for Planet_{planet_id} ...")
    with h5py.File(TRACEDATA, "r") as f:
        key = f"Planet_{planet_id}"
        if key in f:
            td = f[key]["tracedata"][:]
            wt = f[key]["weights"][:]
            rng = np.random.default_rng(42)
            idx = rng.choice(len(td), size=10_000, replace=True, p=wt / wt.sum())
            samples_ns = td[idx]
            has_gt = True
        else:
            print(f"  Warning: {key} not found in Tracedata.hdf5")
            has_gt = False

    # Corner plot
    print("Plotting ...")
    fig = corner.corner(
        samples_npe,
        labels=PARAM_NAMES,
        truths=theta_true,
        truth_color="black",
        color="#1E7B3E",
        show_titles=True,
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 9},
        smooth=1.0,
        bins=40,
    )

    if has_gt:
        corner.corner(
            samples_ns,
            labels=PARAM_NAMES,
            color="#F3AC3F",
            smooth=1.0,
            bins=40,
            fig=fig,
        )

    legend_items = [
        plt.Line2D([0], [0], color="#1E7B3E", lw=2, label="NPE (this model)"),
    ]
    if has_gt:
        legend_items.append(
            plt.Line2D([0], [0], color="#F3AC3F", lw=2, label="Nested sampling (ground truth)")
        )
    fig.legend(handles=legend_items, loc="upper right", fontsize=9)

    fig.suptitle(
        f"NPE on ABC — Planet_{planet_id}\n(smoke-test checkpoint, not converged)",
        y=1.01, fontsize=10,
    )

    out_path = OUT_DIR / f"abc_npe_planet{planet_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planet-idx", type=int, default=0)
    args = parser.parse_args()
    main(planet_idx=args.planet_idx)
