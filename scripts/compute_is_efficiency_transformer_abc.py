"""
Compute IS-efficiency of Transformer-FMPE on ABC test set.

Method: IS against nested-sampling reference posterior.
  theta_i, log_q_i = sample_and_log_prob_batch(context)   [single ODE pass]
  log_p_NS_i       = log KDE(theta_i)                      [scipy KDE on NS]
  log w_i          = log_p_NS_i - log_q_i
  epsilon          = ESS / N,  ESS = (sum w)^2 / sum w^2

N_SAMPLES is kept at 2000 per planet (vs 10k for NPE) because each FMPE
evaluation requires an ODE solve — ~30–60s per planet on CPU.

Run from Project/:
  python scripts/compute_is_efficiency_transformer_abc.py [--n-planets 20]
"""

import argparse
import h5py
import numpy as np
import torch
import yaml
from pathlib import Path
from scipy.stats import gaussian_kde
from tqdm import tqdm

from fm4ar.models.build_model import build_model
from fm4ar.datasets.theta_scalers import get_theta_scaler

ABC_DIR  = Path("data/abc")
TRACEDATA = ABC_DIR / "Level2Data/Ground Truth Package/Tracedata.hdf5"
CKPT_DIR = Path("configs/transformer_abc")

N_SAMPLES = 2_000   # per planet — ODE solve limits throughput on CPU


def compute_ess(log_w: np.ndarray) -> tuple[float, float]:
    log_w = log_w - log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    ess = float(1.0 / (w ** 2).sum())
    return ess, ess / len(log_w)


def planet_efficiency(
    model,
    theta_scaler,
    flux: torch.Tensor,
    wlen: torch.Tensor,
    ns_samples: np.ndarray,
    ns_weights: np.ndarray,
) -> tuple[float, float] | None:
    ns_w = ns_weights / ns_weights.sum()

    # Normalise NS samples for KDE numerical stability
    ns_mean = ns_samples.mean(axis=0)
    ns_std  = ns_samples.std(axis=0) + 1e-8
    ns_norm = (ns_samples - ns_mean) / ns_std

    try:
        kde = gaussian_kde(ns_norm.T, weights=ns_w, bw_method="scott")
    except Exception as e:
        print(f"    KDE error: {e}")
        return None

    context = {
        "flux": flux.unsqueeze(0).expand(N_SAMPLES, -1),
        "wlen": wlen.unsqueeze(0).expand(N_SAMPLES, -1),
    }

    with torch.no_grad():
        # single ODE pass → samples (scaled) + log q (in scaled space)
        theta_scaled, log_q = model.sample_and_log_prob_batch(
            context=context,
            tolerance=1e-3,   # looser tolerance for speed; tighten for paper results
        )

    # Inverse-scale samples to physical space for KDE comparison
    theta_np = theta_scaler.inverse_array(theta_scaled.cpu().numpy())

    # Evaluate KDE in normalised NS space
    theta_norm = (theta_np - ns_mean) / ns_std
    log_p_ns = np.log(kde.evaluate(theta_norm.T) + 1e-300)
    log_q_np = log_q.cpu().numpy()

    log_w = log_p_ns - log_q_np
    log_w = np.clip(log_w, log_w.max() - 50, np.inf)

    return compute_ess(log_w)


def main(n_planets: int = 20) -> None:
    # ODE solver requires float64 → CPU only (MPS doesn't support float64)
    device = torch.device("cpu")

    print(f"Loading Transformer-FMPE from {CKPT_DIR / 'model__best.pt'} ...")
    model = build_model(
        file_path=CKPT_DIR / "model__best.pt",
        experiment_dir=CKPT_DIR,
        device="cpu",
    )
    model.network.eval()

    with open(CKPT_DIR / "config.yaml") as fh:
        config = yaml.safe_load(fh)
    theta_scaler = get_theta_scaler(config.get("theta_scaler", {}))

    ess_list: list[float] = []
    eps_list: list[float] = []

    with h5py.File(ABC_DIR / "abc_test.hdf", "r") as f, \
         h5py.File(TRACEDATA, "r") as gt:

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
        print(f"Evaluating {len(indices)} planets (N_SAMPLES={N_SAMPLES} each) ...")
        print(f"Note: each planet requires an ODE solve — expect ~30–60s/planet on CPU.\n")

        wlen_global = torch.tensor(f["wlen"][0], dtype=torch.float32)

        for idx in tqdm(indices):
            flux      = torch.tensor(f["flux"][idx], dtype=torch.float32)
            planet_id = int(f["planet_id"][idx])

            key = f"Planet_{planet_id}"
            try:
                ns_samples = gt[key]["tracedata"][:]
                ns_weights = gt[key]["weights"][:]
            except ValueError:
                tqdm.write(f"  Planet_{planet_id}: malformed tracedata, skipping")
                continue
            if ns_samples.ndim != 2 or ns_samples.shape[0] < 10:
                tqdm.write(f"  Planet_{planet_id}: too few NS samples, skipping")
                continue

            result = planet_efficiency(
                model, theta_scaler, flux, wlen_global, ns_samples, ns_weights
            )
            if result is None:
                continue

            ess, eps = result
            ess_list.append(ess)
            eps_list.append(eps)
            tqdm.write(f"  Planet_{planet_id:>6d}: ESS={ess:>7.1f}  ε={eps*100:.3f}%")

    if not eps_list:
        print("No valid results.")
        return

    print(f"\n{'─'*50}")
    print(f"  Model             : Transformer-FMPE (Phase 1)")
    print(f"  Planets evaluated : {len(eps_list)}")
    print(f"  N_SAMPLES/planet  : {N_SAMPLES}")
    print(f"  Mean ESS          : {np.mean(ess_list):.1f}")
    print(f"  Mean ε            : {np.mean(eps_list)*100:.3f}%")
    print(f"  Median ε          : {np.median(eps_list)*100:.3f}%")
    print(f"  Min / Max ε       : {np.min(eps_list)*100:.3f}% / {np.max(eps_list)*100:.3f}%")
    print(f"{'─'*50}")
    print(f"  Phase 0 NPE baseline (N=10k):  mean ε = 0.025%")
    print(f"  Any improvement validates transformer context encoding.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-planets", type=int, default=20)
    args = parser.parse_args()
    main(n_planets=args.n_planets)
