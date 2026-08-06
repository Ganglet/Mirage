"""
CycleGAN domain translation for MIRAGE Phase 3 Track 2.

Trains a sim-to-real spectral translator:
  G: simulated ABC spectra → real JWST-like spectra
  F: real JWST-like spectra → simulated ABC spectra

Cycle-consistency loss enforces F(G(x)) ≈ x and G(F(y)) ≈ y.
Identity loss stabilises colour/intensity during the first epochs.

This is the Phase 3 Track 2 ablation experiment (D4): it quantifies how much
IS-efficiency improvement comes from CycleGAN *translation* vs domain
*randomisation* alone (the Phase 2 approach). The two are directly comparable
because both condition the same FMPE backbone — only the pre-processing differs.

Design follows Zhu et al. 2017 (arXiv:1703.10593), adapted for 1-D spectra
(52-bin ABC grid) instead of images. Discriminators use 1-D PatchGAN.

Architecture:
  Generator:   ResNet encoder–decoder (1-D conv, 9 residual blocks)
  Discriminator: PatchGAN (1-D, 4 conv layers, spectral norm)

Losses:
  adversarial: LSGAN (Mao et al. 2017) — more stable than vanilla BCE on spectra
  cycle:       L1 (λ_cyc = 10.0, default)
  identity:    L1 (λ_id  = 5.0,  default)

Registered into the MIRAGE package via mirage/__init__.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Generator ──────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """1-D residual block with instance normalisation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad1d(1),
            nn.Conv1d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SpectralGenerator(nn.Module):
    """
    ResNet generator for 1-D spectra.

    Input/output shape: (batch, n_bins) — a single spectral channel.
    Internally treated as (batch, 1, n_bins) for 1-D convolutions.

    Architecture: c7s1-64, d128, d256, R×n_res, u128, u64, c7s1-1
    (following Zhu et al. 2017 image convention, projected to 1-D).
    """

    def __init__(self, n_bins: int = 52, n_res: int = 9, ngf: int = 64) -> None:
        super().__init__()

        self.n_bins = n_bins

        # Encoder
        self.encoder = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(1, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm1d(ngf),
            nn.ReLU(inplace=True),

            nn.Conv1d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(ngf * 4),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResBlock1D(ngf * 4) for _ in range(n_res)]
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(ngf * 4, ngf * 2, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm1d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(ngf * 2, ngf, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm1d(ngf),
            nn.ReLU(inplace=True),

            nn.ReflectionPad1d(3),
            nn.Conv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_bins) → (batch, n_bins)"""
        h = x.unsqueeze(1)                 # (B, 1, n_bins)
        h = self.encoder(h)
        h = self.res_blocks(h)
        h = self.decoder(h)
        out = h.squeeze(1)                 # (B, n_bins)
        # Ensure output length matches input (refl-pad may shift by 1)
        if out.shape[-1] != self.n_bins:
            out = out[..., :self.n_bins]
        return out


# ── Discriminator ──────────────────────────────────────────────────────────

class PatchDiscriminator1D(nn.Module):
    """
    1-D PatchGAN discriminator.

    Outputs a (batch, 1, n_patches) patch-decision map.
    Spectral normalisation stabilises adversarial training.
    """

    def __init__(self, ndf: int = 64) -> None:
        super().__init__()

        def block(in_ch, out_ch, stride=2, norm=True):
            layers: list[nn.Module] = [
                nn.utils.spectral_norm(
                    nn.Conv1d(in_ch, out_ch, kernel_size=4,
                              stride=stride, padding=1)
                )
            ]
            if norm:
                layers.append(nn.InstanceNorm1d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(1, ndf, norm=False),
            *block(ndf, ndf * 2),
            *block(ndf * 2, ndf * 4),
            *block(ndf * 4, ndf * 8, stride=1),
            nn.utils.spectral_norm(
                nn.Conv1d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_bins) → (batch, 1, n_patches)"""
        return self.model(x.unsqueeze(1))


# ── Loss functions ─────────────────────────────────────────────────────────

def lsgan_loss_d(real_pred: torch.Tensor,
                 fake_pred: torch.Tensor) -> torch.Tensor:
    """LSGAN discriminator loss: E[(D(real)-1)²] + E[D(fake)²]."""
    return 0.5 * (F.mse_loss(real_pred, torch.ones_like(real_pred)) +
                  F.mse_loss(fake_pred, torch.zeros_like(fake_pred)))


def lsgan_loss_g(fake_pred: torch.Tensor) -> torch.Tensor:
    """LSGAN generator loss: E[(D(G(x))-1)²]."""
    return F.mse_loss(fake_pred, torch.ones_like(fake_pred))


def cycle_loss(real: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """L1 cycle-consistency loss."""
    return F.l1_loss(reconstructed, real)


def identity_loss(real: torch.Tensor, identity_out: torch.Tensor) -> torch.Tensor:
    """L1 identity loss: G(real) ≈ real when real is already in target domain."""
    return F.l1_loss(identity_out, real)


# ── CycleGAN ──────────────────────────────────────────────────────────────

class CycleGAN(nn.Module):
    """
    CycleGAN for MIRAGE Phase 3 Track 2.

    Domain A = simulated ABC spectra (normalised, ~O(1))
    Domain B = real JWST-like spectra (same 52-bin grid, real noise)

    Generators:
        G_AB: A → B  (sim  → real-style)
        G_BA: B → A  (real → sim-style)
    Discriminators:
        D_A: distinguishes real A from G_BA(B)
        D_B: distinguishes real B from G_AB(A)
    """

    def __init__(
        self,
        n_bins: int = 52,
        n_res: int = 9,
        ngf: int = 64,
        ndf: int = 64,
        lambda_cyc: float = 10.0,
        lambda_id: float = 5.0,
    ) -> None:
        super().__init__()

        self.lambda_cyc = lambda_cyc
        self.lambda_id = lambda_id

        self.G_AB = SpectralGenerator(n_bins=n_bins, n_res=n_res, ngf=ngf)
        self.G_BA = SpectralGenerator(n_bins=n_bins, n_res=n_res, ngf=ngf)
        self.D_A = PatchDiscriminator1D(ndf=ndf)
        self.D_B = PatchDiscriminator1D(ndf=ndf)

        self._init_weights()

    def _init_weights(self) -> None:
        """Gaussian initialisation (σ=0.02), matching Zhu et al. 2017."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined generator loss.

        Returns:
            loss_G: scalar total generator loss
            metrics: dict of individual loss components for logging
        """
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)

        # Adversarial
        loss_G_AB = lsgan_loss_g(self.D_B(fake_B))
        loss_G_BA = lsgan_loss_g(self.D_A(fake_A))

        # Cycle
        rec_A = self.G_BA(fake_B)
        rec_B = self.G_AB(fake_A)
        loss_cyc = (cycle_loss(real_A, rec_A) + cycle_loss(real_B, rec_B))

        # Identity
        loss_id = (identity_loss(real_B, self.G_AB(real_B)) +
                   identity_loss(real_A, self.G_BA(real_A)))

        loss_G = (loss_G_AB + loss_G_BA
                  + self.lambda_cyc * loss_cyc
                  + self.lambda_id * loss_id)

        metrics = {
            "loss_G":     float(loss_G.detach()),
            "loss_G_AB":  float(loss_G_AB.detach()),
            "loss_G_BA":  float(loss_G_BA.detach()),
            "loss_cyc":   float(loss_cyc.detach()),
            "loss_id":    float(loss_id.detach()),
        }
        return loss_G, metrics

    def discriminator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined discriminator loss.

        Returns:
            loss_D: scalar total discriminator loss
            metrics: dict of individual loss components for logging
        """
        with torch.no_grad():
            fake_B = self.G_AB(real_A)
            fake_A = self.G_BA(real_B)

        loss_D_A = lsgan_loss_d(self.D_A(real_A), self.D_A(fake_A))
        loss_D_B = lsgan_loss_d(self.D_B(real_B), self.D_B(fake_B))
        loss_D = 0.5 * (loss_D_A + loss_D_B)

        metrics = {
            "loss_D":   float(loss_D.detach()),
            "loss_D_A": float(loss_D_A.detach()),
            "loss_D_B": float(loss_D_B.detach()),
        }
        return loss_D, metrics

    @torch.no_grad()
    def translate_sim_to_real(self, sim_spectra: torch.Tensor) -> torch.Tensor:
        """
        Translate a batch of simulated ABC spectra into real JWST-style.

        Args:
            sim_spectra: (batch, n_bins) normalised ABC spectra
        Returns:
            (batch, n_bins) JWST-style translated spectra
        """
        self.eval()
        return self.G_AB(sim_spectra)

    @torch.no_grad()
    def translate_real_to_sim(self, real_spectra: torch.Tensor) -> torch.Tensor:
        """
        Translate a batch of real JWST-style spectra into ABC-sim domain.

        Args:
            real_spectra: (batch, n_bins) JWST-style spectra
        Returns:
            (batch, n_bins) ABC-sim-style translated spectra
        """
        self.eval()
        return self.G_BA(real_spectra)
