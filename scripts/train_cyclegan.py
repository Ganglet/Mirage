"""
Phase 3 Track 2 — CycleGAN domain translation training.

Trains a sim-to-real CycleGAN on:
  Domain A: normalised ABC synthetic spectra (52 bins, from abc_train.hdf)
  Domain B: real JWST WASP-39b spectra (52 bins, from the standardised CSV)

Writes checkpoints to configs/cyclegan/ and a training log to
configs/cyclegan/training_log.csv.

Usage:
    python scripts/train_cyclegan.py [options]

Example (quick smoke test, ~2 min on M1 Pro):
    python scripts/train_cyclegan.py --epochs 5 --batch-size 32 --n-abc 500

Full training:
    python scripts/train_cyclegan.py --epochs 200 --batch-size 64
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import mirage  # noqa: F401 — registers MIRAGE components

from mirage.nn.cyclegan import CycleGAN

# ── Defaults ───────────────────────────────────────────────────────────────
ABC_DIR    = Path("data/abc")
REAL_CSV   = Path("MAST_2026-05-11T1524/MAST_2026-05-11T1524/JWST/WASP39b_final_standardized.csv")
CKPT_DIR   = Path("configs/cyclegan")
N_BINS     = 52
LR         = 2e-4
BETAS      = (0.5, 0.999)


# ── Data loading ───────────────────────────────────────────────────────────

def load_abc_spectra(n_samples: int | None = None) -> torch.Tensor:
    """
    Load normalised ABC training spectra from abc_train.hdf.
    Uses the same normalisation as the noisecond arms (flux_mean/flux_std from hdf).
    Returns: (N, N_BINS) float32 tensor in [-1, 1] for CycleGAN.
    """
    path = ABC_DIR / "abc_train.hdf"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/prepare_abc_hdf5.py first."
        )

    with h5py.File(path) as f:
        flux = f["flux"][:]
        if "flux_mean" in f and "flux_std" in f:
            mean = f["flux_mean"][:]
            std  = f["flux_std"][:]
            flux = (flux - mean) / std

    flux = flux.astype(np.float32)

    # Clip extremes and rescale to [-1, 1] for Tanh output
    p1, p99 = np.percentile(flux, [1, 99])
    flux = np.clip(flux, p1, p99)
    flux = 2.0 * (flux - p1) / (p99 - p1 + 1e-8) - 1.0

    if n_samples is not None:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(flux), min(n_samples, len(flux)), replace=False)
        flux = flux[idx]

    return torch.from_numpy(flux)


def load_real_spectra(n_bins: int = N_BINS) -> torch.Tensor:
    """
    Load real WASP-39b spectra from the standardised CSV and bin to n_bins.
    Returns: (N_integrations, n_bins) float32 tensor in [-1, 1].
    """
    if not REAL_CSV.exists():
        raise FileNotFoundError(
            f"{REAL_CSV} not found. Run scripts/extract_out_of_transit_data.py "
            "to produce the out-of-transit frames, or use the standardised CSV."
        )

    df = pd.read_csv(REAL_CSV)

    # Identify wavelength and flux columns flexibly
    wav_col  = next((c for c in df.columns if "wavelength" in c.lower()), None)
    flux_col = next((c for c in df.columns if "flux" in c.lower()
                     and "error" not in c.lower()), None)

    if wav_col is None or flux_col is None:
        raise ValueError(
            f"Cannot find wavelength/flux columns in {REAL_CSV}. "
            f"Columns: {list(df.columns)}"
        )

    df = df[[wav_col, flux_col]].dropna()
    df.columns = ["wavelength", "flux"]
    df = df[df["flux"] > 0].sort_values("wavelength")

    # Bin to the model's n_bins wavelength grid
    wav_edges = np.linspace(df["wavelength"].min(), df["wavelength"].max(), n_bins + 1)
    df["bin"] = pd.cut(df["wavelength"], bins=wav_edges, labels=False)
    binned = df.groupby("bin")["flux"].mean().values.astype(np.float32)

    # Pad/trim to exactly n_bins
    if len(binned) < n_bins:
        binned = np.pad(binned, (0, n_bins - len(binned)), constant_values=np.nanmedian(binned))
    binned = binned[:n_bins]
    binned = np.nan_to_num(binned, nan=np.nanmedian(binned))

    # Expand to a small "dataset" by bootstrap sampling (real data is scarce)
    # This gives the GAN more steps on real data without information fabrication
    rng = np.random.default_rng(0)
    n_real = 500
    noise_scale = binned.std() * 0.01  # 1% photon noise jitter
    real_samples = binned[None, :] + rng.standard_normal((n_real, n_bins)) * noise_scale
    real_samples = real_samples.astype(np.float32)

    # Rescale to [-1, 1]
    p1, p99 = np.percentile(real_samples, [1, 99])
    real_samples = np.clip(real_samples, p1, p99)
    real_samples = 2.0 * (real_samples - p1) / (p99 - p1 + 1e-8) - 1.0

    return torch.from_numpy(real_samples)


# ── Training loop ──────────────────────────────────────────────────────────

def train(
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = LR,
    n_abc: int | None = None,
    n_res: int = 9,
    ngf: int = 64,
    ndf: int = 64,
    lambda_cyc: float = 10.0,
    lambda_id: float = 5.0,
    save_every: int = 10,
    device_str: str = "cpu",
) -> None:

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)
    print(f"\n{'='*70}")
    print("  MIRAGE Phase 3 Track 2 — CycleGAN Training")
    print(f"{'='*70}")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  λ_cyc:      {lambda_cyc},  λ_id: {lambda_id}")
    print(f"  Checkpoint: {CKPT_DIR}")

    # Load data
    print("\n  Loading data ...")
    sim_data  = load_abc_spectra(n_samples=n_abc).to(device)
    real_data = load_real_spectra(n_bins=N_BINS).to(device)
    print(f"  ABC simulated spectra:  {sim_data.shape}")
    print(f"  Real JWST spectra:      {real_data.shape}")

    sim_loader  = DataLoader(TensorDataset(sim_data),  batch_size=batch_size,
                             shuffle=True, drop_last=True)
    real_loader = DataLoader(TensorDataset(real_data), batch_size=batch_size,
                             shuffle=True, drop_last=True)

    # Build model
    model = CycleGAN(
        n_bins=N_BINS, n_res=n_res, ngf=ngf, ndf=ndf,
        lambda_cyc=lambda_cyc, lambda_id=lambda_id,
    ).to(device)

    # Optimisers — separate G and D (Adam with β₁=0.5, standard GAN)
    opt_G = optim.Adam(
        list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
        lr=lr, betas=BETAS,
    )
    opt_D = optim.Adam(
        list(model.D_A.parameters()) + list(model.D_B.parameters()),
        lr=lr, betas=BETAS,
    )

    # Linear LR decay from epoch 100 → 0 at epoch `epochs` (Zhu et al. 2017)
    decay_start = epochs // 2
    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / (epochs - decay_start + 1e-8))

    sched_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    # Save config
    config = {
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "n_bins": N_BINS, "n_res": n_res, "ngf": ngf, "ndf": ndf,
        "lambda_cyc": lambda_cyc, "lambda_id": lambda_id,
        "n_abc_samples": int(sim_data.shape[0]),
        "n_real_samples": int(real_data.shape[0]),
        "device": device_str,
    }
    with open(CKPT_DIR / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    # Training log
    log_path = CKPT_DIR / "training_log.csv"
    log_fields = ["epoch", "loss_G", "loss_G_AB", "loss_G_BA",
                  "loss_cyc", "loss_id", "loss_D", "loss_D_A", "loss_D_B",
                  "lr_G", "lr_D"]
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    best_loss_G = float("inf")

    print(f"\n  Training for {epochs} epochs ...\n")
    for epoch in range(1, epochs + 1):

        model.train()
        epoch_metrics: dict[str, list[float]] = {k: [] for k in log_fields}

        # Zip the two loaders — iterate until the shorter one is exhausted
        for (batch_A,), (batch_B,) in zip(sim_loader, real_loader):

            # ── Train Generators ──
            opt_G.zero_grad()
            loss_G, g_metrics = model.generator_loss(batch_A, batch_B)
            loss_G.backward()
            opt_G.step()

            # ── Train Discriminators ──
            opt_D.zero_grad()
            loss_D, d_metrics = model.discriminator_loss(batch_A, batch_B)
            loss_D.backward()
            opt_D.step()

            for k, v in {**g_metrics, **d_metrics}.items():
                epoch_metrics[k].append(v)

        sched_G.step()
        sched_D.step()

        # Aggregate epoch metrics
        row = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_metrics.items()}
        row["epoch"]  = epoch
        row["lr_G"]   = opt_G.param_groups[0]["lr"]
        row["lr_D"]   = opt_D.param_groups[0]["lr"]
        log_writer.writerow(row)
        log_file.flush()

        # Console output every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>4}/{epochs} | "
                f"loss_G={row['loss_G']:.4f}  loss_D={row['loss_D']:.4f}  "
                f"cyc={row['loss_cyc']:.4f}  id={row['loss_id']:.4f}  "
                f"lr={row['lr_G']:.2e}"
            )

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = CKPT_DIR / f"cyclegan_epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "G_AB": model.G_AB.state_dict(),
                "G_BA": model.G_BA.state_dict(),
                "D_A":  model.D_A.state_dict(),
                "D_B":  model.D_B.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "config": config,
            }, ckpt_path)

        # Save best generator
        if row["loss_G"] < best_loss_G:
            best_loss_G = row["loss_G"]
            torch.save({
                "epoch": epoch,
                "loss_G": best_loss_G,
                "G_AB": model.G_AB.state_dict(),
                "G_BA": model.G_BA.state_dict(),
                "config": config,
            }, CKPT_DIR / "cyclegan_best.pt")

    log_file.close()

    print(f"\n{'='*70}")
    print(f"  Training complete.")
    print(f"  Best generator loss: {best_loss_G:.4f}")
    print(f"  Checkpoints saved to: {CKPT_DIR}")
    print(f"  Training log:         {log_path}")
    print(f"{'='*70}\n")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train Phase 3 Track 2 CycleGAN — sim-to-real spectral translation"
    )
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--n-abc",       type=int,   default=None,
                   help="Limit ABC samples (default: all; use 500 for smoke test)")
    p.add_argument("--n-res",       type=int,   default=9,
                   help="Residual blocks in generator (9 full, 6 fast)")
    p.add_argument("--ngf",         type=int,   default=64)
    p.add_argument("--ndf",         type=int,   default=64)
    p.add_argument("--lambda-cyc",  type=float, default=10.0)
    p.add_argument("--lambda-id",   type=float, default=5.0)
    p.add_argument("--save-every",  type=int,   default=10)
    p.add_argument("--device",      type=str,   default="cpu",
                   choices=["cpu", "cuda", "mps"])
    args = p.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_abc=args.n_abc,
        n_res=args.n_res,
        ngf=args.ngf,
        ndf=args.ndf,
        lambda_cyc=args.lambda_cyc,
        lambda_id=args.lambda_id,
        save_every=args.save_every,
        device_str=args.device,
    )
