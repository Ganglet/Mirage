"""
Phase 3 P3-D1 validation — does whitening (condition on structure, not scale)
actually hold on real JWST data?

Two checks, both aimed at the reviewer cross-examination "the noise scale is
specific to this dataset":

  (A) Scale-invariance — feed the real OOT frames and a ×10-scaled copy through
      CovarianceEmbedding(whiten=True): the conditioning vector must be identical
      (and must NOT be, with whiten=False). This is the architectural answer.

  (B) Envelope check — the residual generalisation axis is STRUCTURE. Estimate
      each real instrument's correlation length (1/e distance of mean corr vs Δλ)
      and check it lands inside the Phase-2 training-kernel envelope. n=4 real
      instruments, not n=1.

Run from Project/:
    python scripts/validate_whitening.py
"""

import numpy as np
import pandas as pd
import torch

from mirage.datasets.noise import CorrelatedNoiseGenerator
from mirage.nn.covariance_embedding import CovarianceEmbedding
from build_real_covariance import abc_grid, rebin_frames, INSTRUMENTS, OOT


def corr_1e_distance(corr, wl):
    """Effective correlation length = Δλ where mean corr crosses 1/e."""
    P = len(wl)
    iu = np.triu_indices(P, k=1)
    dlam = np.abs(wl[:, None] - wl[None, :])[iu]
    c = corr[iu]
    edges = np.quantile(dlam, np.linspace(0, 1, 21))
    centers, means = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (dlam >= a) & (dlam <= b)
        if m.sum() > 3:
            centers.append(dlam[m].mean())
            means.append(c[m].mean())
    centers, means = np.array(centers), np.array(means)
    thr = 1.0 / np.e
    below = np.where(means < thr)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    x0, x1, y0, y1 = centers[i - 1], centers[i], means[i - 1], means[i]
    return float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0))


def rel_residuals(fn, edges):
    """Instrument CSV -> (n_int, n_covered) relative-flux residuals + covered idx."""
    df = pd.read_csv(OOT / fn)
    df = df[df["dq_flag"] == 0]
    M, _ = rebin_frames(df, edges)
    covered = np.isfinite(M).all(axis=0)
    idx = np.where(covered)[0]
    F = M[:, idx]
    return F / F.mean(axis=0) - 1.0, idx


def main():
    wl, edges = abc_grid()

    # ── training-kernel envelope: correlation lengths the embedding trained on ──
    gen = CorrelatedNoiseGenerator(
        sigma_min=0.05, sigma_max=0.30, rho_min=0.3, rho_max=0.8,
        se_length_min=0.10, se_length_max=1.00,
        ou_length_min=0.50, ou_length_max=3.00, random_seed=0,
    )
    train_L = []
    for _ in range(300):
        S = gen.sample_covariance(wl)
        d = np.sqrt(np.diag(S))
        L = corr_1e_distance(S / np.outer(d, d), wl)
        if np.isfinite(L):
            train_L.append(L)
    train_L = np.array(train_L)
    lo, hi = np.percentile(train_L, [2.5, 97.5])
    print("Training-kernel envelope (correlation 1/e length, µm):")
    print(f"  median {np.median(train_L):.2f}   95% range [{lo:.2f}, {hi:.2f}]\n")

    # ── (B) real instruments vs envelope ──
    print(f"(B) Envelope check — real correlation structure per instrument")
    print(f"  {'instrument':<14} {'frames':>7} {'bins':>5} {'nn-corr':>8} "
          f"{'|offdiag|':>10} {'corr-L µm':>11} {'in env?':>8}")
    print("  " + "-" * 66)
    for name, fn in INSTRUMENTS.items():
        rel, idx = rel_residuals(fn, edges)
        corr = np.corrcoef(rel, rowvar=False)
        off = np.abs(corr[~np.eye(len(idx), dtype=bool)]).mean()
        nn = np.median(np.diag(corr, k=1))
        L = corr_1e_distance(corr, wl[idx])
        # smallest resolvable Δλ on this instrument's covered grid
        dmin = np.diff(np.sort(wl[idx])).min()
        if np.isfinite(L):
            Ls, inside = f"{L:>9.2f}", ("YES" if lo <= L <= hi else "NO")
        else:
            Ls, inside = f"  <{dmin:>5.2f}", "sub-res"   # decays within one bin
        print(f"  {name:<14} {rel.shape[0]:>7} {len(idx):>5} {nn:>+8.2f} "
              f"{off:>10.3f} {Ls:>11} {inside:>8}")

    # ── (A) scale-invariance on real PRISM frames ──
    rel, idx = rel_residuals("NIRSPEC_CLEAR_PRISM_out_of_transit_full.csv", edges)
    frames = torch.from_numpy(rel[:200].astype(np.float32)).unsqueeze(0)  # (1,K,P)
    torch.manual_seed(0)
    print(f"\n(A) Scale-invariance — real PRISM frames vs ×10 copy (P={len(idx)} bins)")
    for whiten in (True, False):
        emb = CovarianceEmbedding(n_bins=len(idx), embed_dim=64, whiten=whiten)
        emb.eval()
        with torch.no_grad():
            o1 = emb(frames)
            o10 = emb(frames * 10.0)
        drift = (o1 - o10).abs().max().item()
        rel_drift = drift / o1.abs().max().item()
        tag = "INVARIANT" if rel_drift < 1e-4 else "scale-DEPENDENT"
        print(f"  whiten={str(whiten):<5}  max|Δ|={drift:.2e}  rel={rel_drift:.2e}  -> {tag}")


if __name__ == "__main__":
    main()
