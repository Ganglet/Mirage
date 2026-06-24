"""
Phase 2 Track 2 posterior diagnostics.

Computes:
  - IS-efficiency from log importance weights or q-vs-reference KDE weights
  - empirical 68% and 95% credible-interval coverage
  - posterior corner plots
  - deep-ensemble KL diagnostics following the Alvey et al. 2025 use case:
    ensemble disagreement is summarized with pairwise Gaussian KL values.

Expected NPZ inputs are intentionally simple. Provide any subset of:
  samples:            (n_samples, n_dim) posterior samples
  log_weights:        (n_samples,) precomputed log importance weights
  log_q:              (n_samples,) proposal/model log density for samples
  reference_samples:  (n_ref, n_dim) reference posterior samples for KDE
  reference_weights:  (n_ref,) optional reference posterior weights
  theta_true:         (n_dim,) ground-truth or reference parameter vector
  ensemble_samples:   (n_members, n_samples, n_dim) deep-ensemble samples
  parameter_names:    (n_dim,) optional string labels

Run:
  python scripts/evaluate_phase2_track2_posteriors.py path/to/posterior.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde


def stable_ess(log_weights: np.ndarray) -> tuple[float, float]:
    log_weights = np.asarray(log_weights, dtype=float)
    log_weights = log_weights[np.isfinite(log_weights)]
    if log_weights.size == 0:
        return float("nan"), float("nan")
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    ess = float(1.0 / np.sum(weights**2))
    return ess, ess / float(weights.size)


def kde_log_prob(
    samples: np.ndarray,
    reference_samples: np.ndarray,
    reference_weights: np.ndarray | None = None,
) -> np.ndarray:
    ref = np.asarray(reference_samples, dtype=float)
    samples = np.asarray(samples, dtype=float)
    mean = ref.mean(axis=0)
    std = ref.std(axis=0) + 1e-8
    ref_norm = (ref - mean) / std
    samples_norm = (samples - mean) / std
    weights = None if reference_weights is None else reference_weights / np.sum(reference_weights)
    kde = gaussian_kde(ref_norm.T, weights=weights, bw_method="scott")
    return np.log(kde.evaluate(samples_norm.T) + 1e-300)


def central_coverage(
    samples: np.ndarray,
    theta_true: np.ndarray,
    levels: tuple[float, ...] = (0.68, 0.95),
) -> dict[str, float]:
    samples = np.asarray(samples, dtype=float)
    theta_true = np.asarray(theta_true, dtype=float)
    results: dict[str, float] = {}
    for level in levels:
        alpha = (1.0 - level) / 2.0
        lo = np.quantile(samples, alpha, axis=0)
        hi = np.quantile(samples, 1.0 - alpha, axis=0)
        covered = (theta_true >= lo) & (theta_true <= hi)
        results[f"coverage_{int(level * 100)}"] = float(np.mean(covered))
    return results


def gaussian_kl(mean_p: np.ndarray, cov_p: np.ndarray, mean_q: np.ndarray, cov_q: np.ndarray) -> float:
    dim = mean_p.size
    jitter = 1e-6 * np.eye(dim)
    cov_p = cov_p + jitter
    cov_q = cov_q + jitter
    inv_q = np.linalg.inv(cov_q)
    diff = mean_q - mean_p
    sign_p, logdet_p = np.linalg.slogdet(cov_p)
    sign_q, logdet_q = np.linalg.slogdet(cov_q)
    if sign_p <= 0 or sign_q <= 0:
        return float("nan")
    trace = np.trace(inv_q @ cov_p)
    quad = float(diff.T @ inv_q @ diff)
    return 0.5 * (trace + quad - dim + logdet_q - logdet_p)


def ensemble_kl_diagnostics(ensemble_samples: np.ndarray) -> dict[str, float]:
    ensemble_samples = np.asarray(ensemble_samples, dtype=float)
    means = ensemble_samples.mean(axis=1)
    covs = np.array([np.cov(member, rowvar=False) for member in ensemble_samples])

    pairwise: list[float] = []
    symmetric: list[float] = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            kl_ij = gaussian_kl(means[i], covs[i], means[j], covs[j])
            kl_ji = gaussian_kl(means[j], covs[j], means[i], covs[i])
            pairwise.extend([kl_ij, kl_ji])
            symmetric.append(0.5 * (kl_ij + kl_ji))

    if not pairwise:
        return {}
    return {
        "ensemble_pairwise_kl_mean": float(np.nanmean(pairwise)),
        "ensemble_pairwise_kl_median": float(np.nanmedian(pairwise)),
        "ensemble_pairwise_kl_max": float(np.nanmax(pairwise)),
        "ensemble_symmetric_kl_mean": float(np.nanmean(symmetric)),
        "ensemble_symmetric_kl_max": float(np.nanmax(symmetric)),
    }


def save_corner_plot(
    samples: np.ndarray,
    labels: list[str],
    truths: np.ndarray | None,
    output: Path,
) -> None:
    import corner
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        show_titles=True,
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 9},
        smooth=1.0,
    )
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def load_labels(npz: np.lib.npyio.NpzFile, n_dim: int) -> list[str]:
    if "parameter_names" not in npz:
        return [f"theta_{i}" for i in range(n_dim)]
    return [str(item) for item in npz["parameter_names"]]


def evaluate(npz_path: Path, output_dir: Path, max_corner_samples: int) -> dict[str, float | str]:
    data = np.load(npz_path, allow_pickle=True)
    if "samples" not in data:
        raise ValueError("NPZ must contain a 'samples' array.")

    samples = np.asarray(data["samples"], dtype=float)
    labels = load_labels(data, samples.shape[1])
    metrics: dict[str, float | str] = {
        "input": str(npz_path),
        "n_samples": float(samples.shape[0]),
        "n_dim": float(samples.shape[1]),
    }

    if "log_weights" in data:
        ess, epsilon = stable_ess(data["log_weights"])
        metrics["ess"] = ess
        metrics["is_efficiency"] = epsilon
    elif "log_q" in data and "reference_samples" in data:
        reference_weights = data["reference_weights"] if "reference_weights" in data else None
        log_p_ref = kde_log_prob(samples, data["reference_samples"], reference_weights)
        ess, epsilon = stable_ess(log_p_ref - data["log_q"])
        metrics["ess"] = ess
        metrics["is_efficiency"] = epsilon

    theta_true = data["theta_true"] if "theta_true" in data else None
    if theta_true is not None:
        metrics.update(central_coverage(samples, theta_true))

    if "ensemble_samples" in data:
        metrics.update(ensemble_kl_diagnostics(data["ensemble_samples"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{npz_path.stem}_diagnostics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    rng = np.random.default_rng(42)
    if samples.shape[0] > max_corner_samples:
        idx = rng.choice(samples.shape[0], size=max_corner_samples, replace=False)
        corner_samples = samples[idx]
    else:
        corner_samples = samples
    save_corner_plot(
        corner_samples,
        labels=labels,
        truths=theta_true,
        output=output_dir / f"{npz_path.stem}_corner.png",
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path, help="Posterior artifact to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/phase2_track2"))
    parser.add_argument("--max-corner-samples", type=int, default=10_000)
    args = parser.parse_args()

    metrics = evaluate(args.npz, args.output_dir, args.max_corner_samples)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
