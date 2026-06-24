"""
CorrelatedNoiseGenerator: wavelength-correlated noise for MIRAGE Phase 2 (WI-2).

Subclasses fm4ar's `NoiseGenerator` so it slots into the same interface, but
lives in the `mirage` package — fm4ar stays unmodified. `get_noise_generator`
here recognises the correlated generator and delegates everything else to
fm4ar's own factory.
"""

from __future__ import annotations

import numpy as np

from fm4ar.datasets.noise import (
    NoiseGenerator,
    get_noise_generator as _fm4ar_get_noise_generator,
)


class CorrelatedNoiseGenerator(NoiseGenerator):
    """
    Gaussian noise with a wavelength-correlated covariance, for the Phase 2
    noise-conditioning experiments on ABC (P2-D2). The hyperparameter
    randomisation here *is* the D4 domain-randomisation injection.

    Per draw the covariance is built from a kernel over wavelength:

        Sigma = white * I
              + se_amp^2 * exp(-0.5 * (d / se_length)^2)   # smooth detector drift
              + ou_amp^2 * exp(-|d| / ou_length)           # 1/f-like long range

    with d = lambda_i - lambda_j in micron. Variances are budgeted so that
    diag(Sigma) == sigma^2 for every bin: a fraction `rho` of sigma^2 is
    correlated (split evenly between the SE and OU terms), the rest is white.
    This keeps the per-bin error bars on the same scale as
    DefaultNoiseGenerator while adding the off-diagonal structure the
    residual-covariance embedding is meant to capture. The diagonal is flat
    by construction, so on ABC the per-wavelength sigma vector (WI-1) carries
    little information and the embedding's contribution (WI-3) is isolated.

    Primary API is explicit so a single drawn Sigma is shared between the
    injected realisation, its error bars, and the out-of-transit frames:
        Sigma = g.sample_covariance(wlen)
        noise = g.sample_noise_from_covariance(Sigma)
        oot   = g.sample_oot_frames(Sigma, n_frames)
    The abstract sample_error_bars / sample_noise pair is a stateful
    convenience for the existing AddNoise transform (caches Sigma between the
    two calls; call them in that order, once per sample).
    """

    def __init__(
        self,
        sigma_min: float = 0.05,
        sigma_max: float = 0.50,
        rho_min: float = 0.3,
        rho_max: float = 0.8,
        se_length_min: float = 0.10,
        se_length_max: float = 1.00,
        ou_length_min: float = 0.50,
        ou_length_max: float = 3.00,
        jitter: float = 1e-8,
        random_seed: int = 42,
    ) -> None:
        if sigma_min < 0 or sigma_max < 0:
            raise ValueError("sigma values must be non-negative!")
        if not 0.0 <= rho_min <= rho_max <= 1.0:
            raise ValueError("require 0 <= rho_min <= rho_max <= 1")

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.se_length_min = se_length_min
        self.se_length_max = se_length_max
        self.ou_length_min = ou_length_min
        self.ou_length_max = ou_length_max
        self.jitter = jitter
        self.rng = np.random.default_rng(random_seed)

        # Covariance cached between sample_error_bars and sample_noise
        self._cached_sigma: np.ndarray | None = None

    def sample_covariance(
        self, wlen: np.ndarray, return_params: bool = False
    ):
        """
        Draw a wavelength-correlated covariance matrix (n_bins, n_bins).
        With `return_params`, also return the drawn kernel hyperparameters
        (used by the recover-Sigma sanity check in WI-5).
        """

        wlen = np.asarray(wlen, dtype=np.float64).reshape(-1)
        n = wlen.size

        sigma = self.rng.uniform(self.sigma_min, self.sigma_max)
        rho = self.rng.uniform(self.rho_min, self.rho_max)
        se_length = self.rng.uniform(self.se_length_min, self.se_length_max)
        ou_length = self.rng.uniform(self.ou_length_min, self.ou_length_max)

        var = sigma ** 2
        white = (1.0 - rho) * var
        corr_var = 0.5 * rho * var  # each correlated term carries half

        d = np.abs(wlen[:, None] - wlen[None, :])
        sigma_mat = (
            corr_var * np.exp(-0.5 * (d / se_length) ** 2)
            + corr_var * np.exp(-d / ou_length)
        )
        # White floor + tiny jitter so diag == sigma^2 and Sigma stays PD
        sigma_mat[np.diag_indices(n)] += white + self.jitter * var

        if return_params:
            params = {
                "sigma": float(sigma),
                "rho": float(rho),
                "se_length": float(se_length),
                "ou_length": float(ou_length),
            }
            return sigma_mat, params
        return sigma_mat

    def _cholesky(self, sigma_mat: np.ndarray) -> np.ndarray:
        """Cholesky factor, escalating jitter on the diagonal if needed."""

        try:
            return np.linalg.cholesky(sigma_mat)
        except np.linalg.LinAlgError:
            n = sigma_mat.shape[0]
            scale = np.trace(sigma_mat) / n
            for k in range(-8, 1):
                jittered = sigma_mat + np.eye(n) * (10.0 ** k) * scale
                try:
                    return np.linalg.cholesky(jittered)
                except np.linalg.LinAlgError:
                    continue
            raise

    def sample_noise_from_covariance(self, sigma_mat: np.ndarray) -> np.ndarray:
        """Draw one correlated noise realisation ~ N(0, Sigma). Shape (n_bins,)."""

        chol = self._cholesky(sigma_mat)
        z = self.rng.standard_normal(sigma_mat.shape[0])
        return (chol @ z).astype(np.float32)

    def sample_oot_frames(
        self, sigma_mat: np.ndarray, n_frames: int
    ) -> np.ndarray:
        """
        Draw `n_frames` noise-only realisations ~ N(0, Sigma) sharing the
        visit covariance. Shape (n_frames, n_bins). These are the surrogate
        out-of-transit frames the covariance embedding (WI-3) consumes.
        """

        chol = self._cholesky(sigma_mat)
        z = self.rng.standard_normal((sigma_mat.shape[0], n_frames))
        return (chol @ z).T.astype(np.float32)

    def sample_error_bars(self, wlen: np.ndarray) -> np.ndarray:
        """
        AddNoise-compat: draw and cache a covariance, return its per-bin
        standard deviations. Must be followed by `sample_noise`.
        """

        sigma_mat = self.sample_covariance(wlen)
        self._cached_sigma = sigma_mat
        return np.sqrt(np.diag(sigma_mat)).astype(np.float32)

    def sample_noise(self, error_bars: np.ndarray) -> np.ndarray:
        """
        AddNoise-compat: draw a correlated realisation from the covariance
        cached by the preceding `sample_error_bars` call. (`error_bars` is
        ignored — correlation cannot be reconstructed from marginals alone.)
        """

        if self._cached_sigma is None:
            raise RuntimeError(
                "sample_noise requires a preceding sample_error_bars call "
                "(the covariance is cached between the two). Prefer the "
                "explicit sample_covariance / sample_noise_from_covariance API."
            )
        sigma_mat = self._cached_sigma
        self._cached_sigma = None
        return self.sample_noise_from_covariance(sigma_mat)


def get_noise_generator(config: dict) -> NoiseGenerator:
    """
    MIRAGE-aware noise-generator factory: builds `CorrelatedNoiseGenerator`,
    delegates every other type to fm4ar's `get_noise_generator`.
    """

    if config["type"] == "CorrelatedNoiseGenerator":
        return CorrelatedNoiseGenerator(**config["kwargs"])
    return _fm4ar_get_noise_generator(config)
