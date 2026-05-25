"""
Compute IS-efficiency of NPE on ABC test set.

Method: IS against nested-sampling reference posterior (no TauREx3 needed).
  θ_i ~ q(θ|x)            [NPE samples]
  log w_i = log p_NS(θ_i)  [scipy KDE on weighted NS samples from Tracedata.hdf5]
           - log q(θ_i|x)  [NPE log_prob]
  ε = ESS / N,  ESS = (Σ w_i)² / Σ w_i²

The log p_NS Jacobian constant (from standardising the KDE) cancels in the
normalised weights, so it does not affect ε.

Run from Project/:
  python scripts/compute_is_efficiency_abc.py [--n-planets 20]
"""

import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import gaussian_kde
from tqdm import tqdm
from lampe.inference import NPE

ABC_DIR = Path("data/abc")
TRACEDATA = ABC_DIR / "Level2Data/Ground Truth Package/Tracedata.hdf5"

DIM_THETA = 6
DIM_X    = 52
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

    def flow(self, x: torch.Tensor):
        return self.npe.flow(self.embedding(x))


def compute_ess(log_w: np.ndarray) -> tuple[float, float]:
    log_w = log_w - log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    ess = float(1.0 / (w ** 2).sum())
    return ess, ess / len(log_w)


def planet_efficiency(
    model: nn.Module,
    flux: torch.Tensor,
    ns_samples: np.ndarray,
    ns_weights: np.ndarray,
    device: torch.device,
) -> tuple[float, float] | None:
    """
    Returns (ESS, ε) for one planet, or None if KDE fails.
    ns_weights should be raw (unnormalized) weights.
    """
    ns_w = ns_weights / ns_weights.sum()

    # Standardise parameter space for KDE numerical stability
    ns_mean = ns_samples.mean(axis=0)
    ns_std  = ns_samples.std(axis=0) + 1e-8
    ns_norm = (ns_samples - ns_mean) / ns_std

    try:
        kde = gaussian_kde(ns_norm.T, weights=ns_w, bw_method="scott")
    except Exception as e:
        print(f"    KDE error: {e}")
        return None

    x = flux.unsqueeze(0).to(device)  # (1, 52)
    flow = model.flow(x)

    with torch.no_grad():
        theta_t = flow.sample((N_SAMPLES,))          # (N_SAMPLES, 1, 6)
        log_q   = flow.log_prob(theta_t).squeeze(-1) # (N_SAMPLES,)

    theta_np   = theta_t.squeeze(1).cpu().numpy()    # (N_SAMPLES, 6)
    theta_norm = (theta_np - ns_mean) / ns_std

    log_p_ns = np.log(kde.evaluate(theta_norm.T) + 1e-300)  # (N_SAMPLES,)
    log_q_np = log_q.cpu().numpy()

    log_w = log_p_ns - log_q_np
    # clip extreme outliers to avoid numerical collapse
    log_w = np.clip(log_w, log_w.max() - 50, np.inf)

    return compute_ess(log_w)


def main(n_planets: int = 20) -> None:
    device = torch.device("cpu")

    model = NPEModel().to(device)
    model.load_state_dict(
        torch.load("checkpoints/abc_npe_best.pt", map_location=device)
    )
    model.eval()

    ess_list: list[float] = []
    eps_list: list[float] = []

    with h5py.File(ABC_DIR / "abc_test.hdf", "r") as f, \
         h5py.File(TRACEDATA, "r") as gt:

        # Pre-filter to test planets that have valid NS posteriors in Tracedata.hdf5.
        # 76% of ABC planets have malformed/empty NS runs — skip them upfront.
        all_planet_ids = f["planet_id"][:]
        valid_indices = [
            i for i, pid in enumerate(all_planet_ids)
            if (key := f"Planet_{int(pid)}") in gt
            and hasattr(gt[key].get("tracedata"), "ndim")
            and gt[key]["tracedata"].ndim == 2
            and gt[key]["tracedata"].shape[0] >= 10
        ]
        indices = valid_indices[:n_planets]
        print(f"Valid test planets: {len(valid_indices)} / {len(all_planet_ids)}")
        print(f"Evaluating {len(indices)} (N_SAMPLES={N_SAMPLES} each) ...")

        for idx in tqdm(indices):
            flux      = torch.tensor(f["flux"][idx], dtype=torch.float32)
            planet_id = int(f["planet_id"][idx])

            key = f"Planet_{planet_id}"
            if key not in gt:
                print(f"  Planet_{planet_id}: not in Tracedata, skipping")
                continue

            try:
                ns_samples = gt[key]["tracedata"][:]  # (N_ns, 6)
                ns_weights = gt[key]["weights"][:]    # (N_ns,)
            except ValueError:
                tqdm.write(f"  Planet_{planet_id}: malformed tracedata, skipping")
                continue
            if ns_samples.ndim != 2 or ns_samples.shape[0] < 10:
                tqdm.write(f"  Planet_{planet_id}: too few NS samples, skipping")
                continue

            result = planet_efficiency(model, flux, ns_samples, ns_weights, device)
            if result is None:
                continue

            ess, eps = result
            ess_list.append(ess)
            eps_list.append(eps)
            tqdm.write(f"  Planet_{planet_id:>6d}: ESS={ess:>9.1f}  ε={eps*100:.3f}%")

    if not eps_list:
        print("No valid results.")
        return

    print(f"\n{'─'*45}")
    print(f"  Planets evaluated : {len(eps_list)}")
    print(f"  Mean ESS          : {np.mean(ess_list):.1f}")
    print(f"  Mean ε            : {np.mean(eps_list)*100:.3f}%")
    print(f"  Median ε          : {np.median(eps_list)*100:.3f}%")
    print(f"  Min / Max ε       : {np.min(eps_list)*100:.3f}% / {np.max(eps_list)*100:.3f}%")
    print(f"{'─'*45}")
    print("Note: ε here is NPE vs NS posterior reference, not NPE vs TauREx3 likelihood.")
    print("Cluster-trained checkpoint will give meaningfully higher ε.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-planets", type=int, default=20,
                        help="Number of test planets to evaluate (default 20)")
    args = parser.parse_args()
    main(n_planets=args.n_planets)
