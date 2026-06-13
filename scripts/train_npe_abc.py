"""
Train NPE on ABC dataset. Adapted from Vasist 2023 (sbi-ear) for:
  - 52 spectral bins (vs 379 in sbi-ear)
  - 6 atmospheric parameters (vs 16)
  - lampe 0.9 API (vs 0.6 in sbi-ear)

CPU smoke test: N_EPOCHS=5, BATCH_SIZE=128, loads abc_train.hdf
Cluster run:   N_EPOCHS=512, BATCH_SIZE=1024, same files
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from lampe.inference import NPE, NPELoss
from pathlib import Path
from tqdm import trange

ABC_DIR = Path("data/abc")
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

PARAM_NAMES = ["T", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
DIM_THETA = 6
DIM_X = 52
DIM_EMBED = 256

N_EPOCHS = 512
BATCH_SIZE = 1024
LR = 1e-3


class ABCDataset(Dataset):
    def __init__(self, hdf_path: Path) -> None:
        with h5py.File(hdf_path, "r") as f:
            self.theta = torch.tensor(f["theta"][:], dtype=torch.float32)
            self.flux = torch.tensor(f["flux"][:], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.theta)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.theta[idx], self.flux[idx]


class NPEModel(nn.Module):
    """NPE with MLP spectrum embedding. Adapted from sbi-ear for ABC dims."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(DIM_X, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, DIM_EMBED), nn.ELU(),
        )
        self.npe = NPE(DIM_THETA, DIM_EMBED, transforms=3)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        return self.npe(theta, self.embedding(x))

    def flow(self, x: Tensor):
        return self.npe.flow(self.embedding(x))


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    train_ds = ABCDataset(ABC_DIR / "abc_train.hdf")
    valid_ds = ABCDataset(ABC_DIR / "abc_valid.hdf")
    print(f"Train: {len(train_ds)}  Valid: {len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = NPEModel().to(device)
    loss_fn = NPELoss(model)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    best_valid_loss = float("inf")

    for epoch in trange(N_EPOCHS, desc="Training"):
        model.train()
        train_losses = []
        for theta, x in train_loader:
            theta, x = theta.to(device), x.to(device)
            loss = loss_fn(theta, x)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for theta, x in valid_loader:
                theta, x = theta.to(device), x.to(device)
                valid_losses.append(loss_fn(theta, x).item())

        train_loss = float(np.mean(train_losses))
        valid_loss = float(np.mean(valid_losses))
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), CKPT_DIR / "abc_npe_best.pt")

        print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | train={train_loss:.4f}  val={valid_loss:.4f}")

    print(f"\nBest valid loss: {best_valid_loss:.4f}")
    print(f"Checkpoint: {CKPT_DIR / 'abc_npe_best.pt'}")


if __name__ == "__main__":
    main()
