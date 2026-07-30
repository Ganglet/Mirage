"""
SpectraEncoder: Transformer-based spectrum context encoder for MIRAGE Phase 1.

Independently reimplemented in PyTorch following the method of SpectraFM
(Koblischke & Bovy 2024, arXiv:2411.04750): per-bin tokenisation with a
wavelength positional encoding (their Eq. 2). No SpectraFM source code is used.
Replaces the DenseResidualNet context encoder in fm4ar's FMPE backbone.

Phase 1 (ABC): single instrument (id=0), 52 tokens, wlen/flux only.
Phase 2 extension: pass "instrument_id" in context for multi-instrument fusion;
                   set use_error_bars=True to condition on per-wavelength
                   uncertainty ("error_bars", WI-1); set use_covariance=True to
                   append a residual-covariance embedding of "oot_frames" (WI-3).

Registered into fm4ar's block registry by `mirage.register` (block type
"SpectraEncoder"); fm4ar itself stays unmodified.
"""

from collections.abc import Mapping

import torch
import torch.nn as nn

from fm4ar.nn.embedding_nets import SupportsDictInput

from mirage.nn.covariance_embedding import CovarianceEmbedding


class SpectraEncoder(SupportsDictInput, nn.Module):
    """
    Transformer encoder over spectral tokens.

    Each bin i becomes token = [flux_i, (σ_i,) inst_emb_i], projected to
    d_model, then augmented with sinusoidal wavelength positional encoding.
    Self-attention over all 52 tokens; mean-pool → linear → output_dim
    context vec. The σ_i feature is included only when use_error_bars=True.
    """

    requires_input_shape = False
    required_keys: list[str] = ["wlen", "flux"]

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_dim: int = 256,
        wlen_min: float = 0.55,   # μm — ABC range; extend to 0.5–12 for JWST
        wlen_max: float = 7.30,
        n_instruments: int = 4,   # NIRSpec, NIRISS, NIRCam, MIRI slots
        use_error_bars: bool = False,  # WI-1: condition on per-wavelength σ
        use_covariance: bool = False,  # WI-3: append residual-cov embedding
        n_bins: int | None = None,     # required when use_covariance
        cov_embed_dim: int = 64,
        cov_method: str = "flatten",
        cov_n_eigen: int = 8,
        cov_hidden_dims: tuple[int, ...] = (128, 64),
        cov_whiten: bool = False,      # P3-D1: condition on structure, not scale
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.wlen_min = wlen_min
        self.wlen_max = wlen_max
        self.use_error_bars = use_error_bars
        self.use_covariance = use_covariance
        self.cov_embed_dim = cov_embed_dim

        # Off by default → Phase 1 behaviour (and checkpoint) preserved; the
        # WI-5 ablation toggles the "+σ" arm with this single flag.
        self.required_keys = ["wlen", "flux"] + (
            ["error_bars"] if use_error_bars else []
        )

        # Instrument identity embedding; inst_embed_dim = d_model // 4
        inst_dim = max(d_model // 4, 1)
        self.instrument_embedding = nn.Embedding(n_instruments, inst_dim)

        # Token projection: (flux [+ σ] + inst_emb) → d_model
        n_scalar_feats = 2 if use_error_bars else 1  # flux (+ error_bars)
        self.token_proj = nn.Linear(n_scalar_feats + inst_dim, d_model)

        # Pre-LN transformer encoder (more stable than post-LN at this scale)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Mean-pool → output projection
        self.output_proj = nn.Linear(d_model, output_dim)

        # Residual-covariance branch (WI-3): embeds OOT noise frames and is
        # concatenated onto the spectrum context. `oot_frames` is (B, K, n_bins)
        # — a shape the build-time dummy-shape inference can't represent — so it
        # is deliberately kept out of required_keys and read in forward (absent
        # → zeros, which also gives the correct bumped output dim at build time).
        if use_covariance:
            if n_bins is None:
                raise ValueError("use_covariance=True requires n_bins.")
            self.cov_embedding = CovarianceEmbedding(
                n_bins=n_bins,
                embed_dim=cov_embed_dim,
                method=cov_method,
                n_eigen=cov_n_eigen,
                hidden_dims=cov_hidden_dims,
                whiten=cov_whiten,
            )

    def _wavelength_pe(self, wlen: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal positional encoding on normalised wavelength.
        Follows SpectraFM (Koblischke & Bovy 2024) Eq. 2, adapted to
        the ABC wavelength range [wlen_min, wlen_max] μm.

        wlen: (batch, seq) in μm
        returns: (batch, seq, d_model)
        """
        lam = (wlen - self.wlen_min) / (self.wlen_max - self.wlen_min)
        lam = lam.unsqueeze(-1)  # (batch, seq, 1)

        d = self.d_model
        k = torch.arange(d, device=wlen.device, dtype=wlen.dtype)
        scale = 1000.0 / (10000.0 ** (k / d))  # (d,)
        angles = lam * scale.unsqueeze(0).unsqueeze(0)  # (batch, seq, d)

        pe = torch.where(k % 2 == 0, torch.sin(angles), torch.cos(angles))
        return pe

    def forward(self, context: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            context["wlen"]: (batch, seq) wavelengths in μm
            context["flux"]: (batch, seq) normalised flux
            context["error_bars"]: (batch, seq) per-wavelength σ, used iff
                use_error_bars=True (zeros if the flag is set but key absent)
            context["instrument_id"]: (batch, seq) int, optional (default 0)

        Returns:
            (batch, output_dim) context vector
        """
        wlen = context["wlen"]
        flux = context["flux"]
        batch, seq = flux.shape

        if "instrument_id" in context:
            inst_ids = context["instrument_id"].long()
        else:
            inst_ids = torch.zeros(batch, seq, dtype=torch.long, device=flux.device)

        inst_emb = self.instrument_embedding(inst_ids)    # (batch, seq, inst_dim)

        # Token scalar features: flux, optionally the per-wavelength σ (WI-1)
        feats = [flux.unsqueeze(-1)]                      # (batch, seq, 1)
        if self.use_error_bars:
            sigma = context.get("error_bars")
            if sigma is None:
                sigma = torch.zeros_like(flux)
            feats.append(sigma.unsqueeze(-1))
        feats.append(inst_emb)
        token_feat = torch.cat(feats, dim=-1)

        tokens = self.token_proj(token_feat)              # (batch, seq, d_model)
        tokens = tokens + self._wavelength_pe(wlen)

        encoded = self.transformer(tokens)                # (batch, seq, d_model)
        pooled = encoded.mean(dim=1)                      # (batch, d_model)
        context_vec = self.output_proj(pooled)            # (batch, output_dim)

        if self.use_covariance:
            oot = context.get("oot_frames")
            if oot is not None:
                cov_emb = self.cov_embedding(oot)         # (batch, cov_embed_dim)
            else:
                # Absent at build-time shape inference (and as a safety net):
                # zeros of the right width keep dim_context = output_dim + cov.
                cov_emb = context_vec.new_zeros(
                    context_vec.shape[0], self.cov_embed_dim
                )
            context_vec = torch.cat([context_vec, cov_emb], dim=-1)

        return context_vec
