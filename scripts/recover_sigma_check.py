"""
Recover-Σ sanity check (WI-5).

Confirms the trained covariance embedding actually encodes the noise structure:
draw OOT frames from CorrelatedNoiseGenerator with KNOWN, randomised kernel
hyperparameters, push them through the model's `cov_embedding`, and fit a
held-out linear probe from the embedding to each hyperparameter. High R² for
the correlation length scales means the embedding carries Σ — i.e. it is not a
dead branch and the network has something real to condition on (P2-D3).

Run from Project/ after training the +cov arm:
  python scripts/recover_sigma_check.py [--n 2000] [--k 100]
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

import mirage  # noqa: F401
from mirage.datasets.noise import CorrelatedNoiseGenerator

from fm4ar.models.build_model import build_model

COV_DIR = Path("configs/noisecond_cov")
ABC_DIR = Path("data/abc")
PARAMS = ["sigma", "rho", "se_length", "ou_length"]


def load_cov_embedding():
    ckpt = COV_DIR / "model__best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"{ckpt} not found — train the +cov arm first:\n"
            f"    python scripts/train.py --experiment-dir {COV_DIR}"
        )
    model = build_model(file_path=ckpt, experiment_dir=COV_DIR, device="cpu")
    model.network.eval()
    enc = model.network.context_embedding_net[0]
    if not getattr(enc, "use_covariance", False):
        raise SystemExit("Loaded model has no covariance embedding.")
    return enc.cov_embedding


def probe_r2(X: np.ndarray, y: np.ndarray, frac_train: float = 0.7) -> float:
    """Held-out OLS R² for predicting y from embedding X (with bias)."""
    n_tr = int(frac_train * len(y))
    Xtr, Xte, ytr, yte = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]
    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pred = np.hstack([Xte, np.ones((len(Xte), 1))]) @ coef
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


@torch.no_grad()
def main(n: int = 2000, k: int = 100, seed: int = 7) -> None:
    cov = load_cov_embedding()

    with h5py.File(ABC_DIR / "abc_train.hdf") as f:
        wlen = f["wlen"][0].astype(np.float64)

    gen = CorrelatedNoiseGenerator(random_seed=seed)
    embeds, labels = [], []
    print(f"Sampling {n} covariances × {k} OOT frames, embedding each ...")
    for _ in tqdm(range(n)):
        sigma_mat, p = gen.sample_covariance(wlen, return_params=True)
        frames = gen.sample_oot_frames(sigma_mat, n_frames=k)        # (k, 52)
        emb = cov(torch.from_numpy(frames).float().unsqueeze(0))      # (1, embed_dim)
        embeds.append(emb.squeeze(0).numpy())
        labels.append([p[name] for name in PARAMS])

    X = np.asarray(embeds)
    Y = np.asarray(labels)

    print(f"\n{'─'*46}")
    print(f"  Recover-Σ: held-out R² of embedding → kernel param")
    print(f"  (embedding dim {X.shape[1]}, n={n}, K={k})")
    print(f"  {'-'*30}")
    for j, name in enumerate(PARAMS):
        print(f"  {name:<12} R² = {probe_r2(X, Y[:, j]):.3f}")
    print(f"{'─'*46}")
    print("  High R² for se_length / ou_length ⇒ the embedding encodes the")
    print("  correlation structure (not a dead branch).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    main(n=args.n, k=args.k, seed=args.seed)
