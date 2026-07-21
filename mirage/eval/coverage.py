"""
Central-credible-interval coverage for MIRAGE (WI-5).

Calibration diagnostic that needs only posterior samples and the known true
theta — valid under any noise model (no reference posterior, no simulator),
which is exactly why it replaces IS-efficiency for the injected-noise ablation
(P2-D9).

For each parameter and credible level α, the central α interval is
[quantile((1-α)/2), quantile((1+α)/2)]; empirical coverage is the fraction of
planets whose true value falls inside. Well-calibrated → coverage ≈ α;
overconfident (e.g. unmodelled correlated noise) → coverage < α.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoverageResult:
    levels: tuple[float, ...]
    per_dim: np.ndarray   # (n_levels, D) empirical coverage per parameter
    overall: np.ndarray   # (n_levels,) mean over parameters
    n: int                # number of planets

    def calibration_error(self) -> np.ndarray:
        """Signed overall coverage − nominal level (negative = overconfident)."""
        return self.overall - np.asarray(self.levels)


def interval_contains(samples, theta_true, level: float) -> np.ndarray:
    """
    Per-parameter indicator that `theta_true` lies in the central `level`
    credible interval of `samples`.

    Args:
        samples: (N, D) posterior samples for one planet.
        theta_true: (D,) ground-truth parameters.
    Returns:
        (D,) boolean array.
    """

    samples = np.asarray(samples)
    lo = np.quantile(samples, (1.0 - level) / 2.0, axis=0)
    hi = np.quantile(samples, (1.0 + level) / 2.0, axis=0)
    t = np.asarray(theta_true)
    return (t >= lo) & (t <= hi)


def coverage(
    samples_list,
    theta_true_list,
    levels=(0.68, 0.95),
) -> CoverageResult:
    """
    Empirical central-interval coverage across planets.

    Args:
        samples_list: list of (N_i, D) posterior-sample arrays.
        theta_true_list: list of (D,) ground-truth parameters.
        levels: credible levels to report.
    """

    levels = tuple(float(lv) for lv in levels)
    if not samples_list:
        d0 = 0
        empty = np.zeros((len(levels), d0))
        return CoverageResult(levels, empty, np.zeros(len(levels)), 0)

    hits = {lv: [] for lv in levels}
    for samples, theta in zip(samples_list, theta_true_list):
        for lv in levels:
            hits[lv].append(interval_contains(samples, theta, lv))

    per_dim = np.stack([np.mean(hits[lv], axis=0) for lv in levels])  # (L, D)
    overall = per_dim.mean(axis=1)                                    # (L,)
    return CoverageResult(levels, per_dim, overall, len(samples_list))
