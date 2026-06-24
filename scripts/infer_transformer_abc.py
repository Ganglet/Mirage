"""
Run Transformer-FMPE inference on a held-out ABC test spectrum.

Loads configs/transformer_abc/model__best.pt, samples 10,000 posterior
samples for one test planet, and overlays the ground-truth NS posterior.

Usage (run from Project/):
    python scripts/infer_transformer_abc.py [--planet-idx N]
"""

import argparse
import h5py
import numpy as np
import torch
import yaml
import corner
import matplotlib.pyplot as plt
from pathlib import Path

import mirage  # noqa: F401 — registers SpectraEncoder block into fm4ar
from fm4ar.models.build_model import build_model
from fm4ar.datasets.theta_scalers import get_theta_scaler

ABC_DIR = Path("data/abc")
TRACEDATA = ABC_DIR / "Level2Data/Ground Truth Package/Tracedata.hdf5"
CKPT_DIR = Path("configs/transformer_abc")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

PARAM_NAMES = ["T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
N_SAMPLES = 10_000


def main(planet_idx: int = 0) -> None:
    # torchdiffeq ODE solver requires float64, which MPS doesn't support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Loading test planet index {planet_idx} from abc_test.hdf ...")
    with h5py.File(ABC_DIR / "abc_test.hdf", "r") as f:
        flux = torch.tensor(f["flux"][planet_idx], dtype=torch.float32)
        wlen = torch.tensor(f["wlen"][0], dtype=torch.float32)
        theta_true = f["theta"][planet_idx]
        planet_id = int(f["planet_id"][planet_idx])

    print(f"  Planet ID: {planet_id}")
    print(f"  True theta: {dict(zip(PARAM_NAMES, theta_true))}")

    ckpt_path = CKPT_DIR / "model__best.pt"
    print(f"Loading Transformer-FMPE checkpoint from {ckpt_path} ...")
    model = build_model(
        file_path=ckpt_path,
        experiment_dir=CKPT_DIR,
        device="cpu",
    )
    model.network.eval()

    with open(CKPT_DIR / "config.yaml") as fh:
        config = yaml.safe_load(fh)
    theta_scaler = get_theta_scaler(config.get("theta_scaler", {}))

    print(f"Sampling {N_SAMPLES} posterior samples ...")
    context = {
        "flux": flux.unsqueeze(0).expand(N_SAMPLES, -1).to(device),
        "wlen": wlen.unsqueeze(0).expand(N_SAMPLES, -1).to(device),
    }

    with torch.no_grad():
        samples_scaled = model.sample_batch(context=context).cpu().numpy()

    samples = theta_scaler.inverse_array(samples_scaled)
    print(f"  Samples shape: {samples.shape}")

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

    print("Plotting ...")
    fig = corner.corner(
        samples,
        labels=PARAM_NAMES,
        truths=theta_true,
        truth_color="black",
        color="#8B1A1A",
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
        plt.Line2D([0], [0], color="#8B1A1A", lw=2, label="Transformer-FMPE (Phase 1)"),
    ]
    if has_gt:
        legend_items.append(
            plt.Line2D([0], [0], color="#F3AC3F", lw=2, label="Nested sampling (ground truth)")
        )
    fig.legend(handles=legend_items, loc="upper right", fontsize=9)

    fig.suptitle(
        f"Transformer-FMPE on ABC — Planet_{planet_id}\n"
        f"(d_model=128, 4 heads, 4 layers, output_dim=256)",
        y=1.01, fontsize=10,
    )

    out_path = OUT_DIR / f"abc_transformer_planet{planet_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planet-idx", type=int, default=0)
    args = parser.parse_args()
    main(planet_idx=args.planet_idx)
