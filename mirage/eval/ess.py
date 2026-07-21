"""
Importance-sampling efficiency / effective sample size for MIRAGE (WI-4).

Promotes the inline Phase 1 computation to one reusable entry point, shared by
the Phase 2 synthetic benchmark and the Phase 3 first real-data measurement.

ESS is the Kish effective sample size of the importance weights w = exp(log_w):

    ESS = (Σ w)² / Σ w²,     epsilon = ESS / N

Deployment thresholds follow D7 (absolute ESS, independent of N):
  - ESS ≥ 500   → deployable      (adequate for 1D marginals)
  - ESS ≥ 2500  → high-quality    (citable in the paper)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# D7 thresholds on ESS = epsilon * N
ESS_DEPLOY = 500
ESS_HIGH_QUALITY = 2_500


@dataclass(frozen=True)
class ESSResult:
    """Per-evaluation ESS / efficiency with its D7 quality class."""

    ess: float
    epsilon: float
    n_samples: int
    quality: str  # "high-quality" | "deployable" | "insufficient"


@dataclass(frozen=True)
class ESSSummary:
    """Aggregate of several ESSResults (e.g. across benchmark planets)."""

    n: int
    mean_ess: float
    mean_epsilon: float
    median_epsilon: float
    frac_deployable: float  # fraction with ESS >= ESS_DEPLOY


def effective_sample_size(log_weights) -> float:
    """
    Kish ESS from (unnormalised) importance log-weights. Computed via the
    max-shift trick — the shift cancels in the (Σw)²/Σw² ratio, so only the
    numerics change, not the value.
    """

    lw = np.asarray(log_weights, dtype=np.float64).ravel()
    if lw.size == 0:
        return 0.0
    w = np.exp(lw - lw.max())
    s = w.sum()
    return float(s * s / np.square(w).sum())


def classify(ess: float) -> str:
    """D7 quality class for an ESS value."""

    if ess >= ESS_HIGH_QUALITY:
        return "high-quality"
    if ess >= ESS_DEPLOY:
        return "deployable"
    return "insufficient"


def is_efficiency(log_weights) -> ESSResult:
    """ESS, epsilon = ESS/N, and the D7 quality class from importance log-weights."""

    lw = np.asarray(log_weights, dtype=np.float64).ravel()
    n = int(lw.size)
    ess = effective_sample_size(lw)
    eps = ess / n if n else 0.0
    return ESSResult(ess=ess, epsilon=eps, n_samples=n, quality=classify(ess))


def aggregate(results) -> ESSSummary:
    """Summarise a list of ESSResults for a benchmark row."""

    results = list(results)
    if not results:
        return ESSSummary(0, 0.0, 0.0, 0.0, 0.0)
    ess = np.array([r.ess for r in results], dtype=np.float64)
    eps = np.array([r.epsilon for r in results], dtype=np.float64)
    return ESSSummary(
        n=len(results),
        mean_ess=float(ess.mean()),
        mean_epsilon=float(eps.mean()),
        median_epsilon=float(np.median(eps)),
        frac_deployable=float((ess >= ESS_DEPLOY).mean()),
    )
