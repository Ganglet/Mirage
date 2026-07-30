"""
CovarianceEmbedding: learned embedding of JWST-like correlated noise for
MIRAGE Phase 2 (WI-3).

Estimates an empirical residual covariance Σ̂ from a stack of out-of-transit
(OOT) noise frames and compresses it into a fixed-length vector that the
flow conditions on alongside the spectrum. The network sees only the frames,
never Σ — identical interface on ABC (synthetic frames) and real JWST (true
OOT integrations), so the component transfers to Phase 3 unchanged (P2-D3).
This is DINGO's power-spectral-density conditioning ported to spectroscopy.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CovarianceEmbedding(nn.Module):
    """
    OOT frames (B, K, P) → empirical Σ̂ (B, P, P) → features → MLP → (B, embed_dim).

    Two swappable featurisations (`method`):
      - "flatten": upper-triangle of Σ̂ (incl. diagonal). Loss-free, simple,
        adequate for ABC's P=52. Cost grows as P², so it does not scale to
        large multi-instrument JWST grids.
      - "eigen": top-k eigenvalues of Σ̂ (with logs). Sign-invariant, fixed
        size for any P — the direct DINGO-PSD analogue and the JWST-scale
        route. Note: `eigvalsh` may be unsupported on MPS; use on CPU/CUDA.
    """

    def __init__(
        self,
        n_bins: int,
        embed_dim: int = 64,
        method: str = "flatten",
        n_eigen: int = 8,
        hidden_dims: tuple[int, ...] = (128, 64),
        whiten: bool = False,
    ) -> None:
        super().__init__()

        if method not in ("flatten", "eigen"):
            raise ValueError(f"unknown method: {method!r}")

        self.n_bins = n_bins
        self.method = method
        self.n_eigen = min(n_eigen, n_bins)
        self.embed_dim = embed_dim
        # whiten=True (P3-D1): standardise each bin across the K frames before
        # forming Σ̂, so it becomes the CORRELATION matrix (diag≈1). The
        # embedding then conditions on noise *structure*, not amplitude — scale
        # is carried separately by the per-λ σ vector (WI-1). This makes the
        # conditioning invariant to per-dataset noise scale, so a real visit
        # ~10× quieter (or louder) than the training σ regime stays in-distribution.
        self.whiten = whiten

        if method == "flatten":
            # Upper triangle incl. diagonal — the unique entries of symmetric Σ̂
            idx = torch.triu_indices(n_bins, n_bins)
            self.register_buffer("triu_i", idx[0])
            self.register_buffer("triu_j", idx[1])
            in_features = n_bins * (n_bins + 1) // 2
        else:
            in_features = 2 * self.n_eigen  # eigenvalues + log-eigenvalues

        layers: list[nn.Module] = []
        d = in_features
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers += [nn.Linear(d, embed_dim)]
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def empirical_cov(frames: torch.Tensor) -> torch.Tensor:
        """(B, K, P) noise frames → empirical covariance (B, P, P)."""

        k = frames.shape[1]
        centered = frames - frames.mean(dim=1, keepdim=True)
        return centered.transpose(1, 2) @ centered / max(k - 1, 1)

    def _featurise(self, sigma_hat: torch.Tensor) -> torch.Tensor:
        if self.method == "flatten":
            return sigma_hat[:, self.triu_i, self.triu_j]  # (B, n_triu)
        # eigen: eigvalsh returns ascending eigenvalues; take the largest k
        evals = torch.linalg.eigvalsh(sigma_hat)[:, -self.n_eigen:]
        evals = evals.clamp_min(1e-12)
        return torch.cat([evals, torch.log(evals)], dim=-1)

    def forward(self, oot_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            oot_frames: (B, K, P) out-of-transit noise-only frames.
        Returns:
            (B, embed_dim) covariance embedding.
        """

        if self.whiten:
            std = oot_frames.std(dim=1, keepdim=True).clamp_min(1e-8)
            oot_frames = (oot_frames - oot_frames.mean(dim=1, keepdim=True)) / std
        sigma_hat = self.empirical_cov(oot_frames)
        return self.mlp(self._featurise(sigma_hat))
