"""
Phase 3 Track 2 — CycleGAN Ablation (standalone, no fm4ar required).

Evaluates CycleGAN translation quality vs domain randomisation using
metrics that don't require the fm4ar inference pipeline:

Metrics (all computable from data alone):
  1. Frechet Inception Distance (FID) proxy — MMD between real and translated
  2. Cycle-consistency error — ||G_BA(G_AB(x)) - x||_1
  3. Domain coverage — KL divergence between marginal flux distributions
  4. Per-wavelength shift — mean absolute bias per bin after translation
  5. Noise structure preservation — correlation matrix similarity

These metrics directly answer the D4 question:
  "Does CycleGAN translation bring simulated spectra closer to the real domain?"

Usage:
    python scripts/run_cyclegan_ablation_standalone.py [--n-samples 500]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

# ── Load CycleGAN without triggering mirage/__init__.py ──────────────────
_cg_path = Path(__file__).resolve().parents[1] / "mirage" / "nn" / "cyclegan.py"
_spec = importlib.util.spec_from_file_location("mirage.nn.cyclegan", _cg_path)
_cg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cg_mod)
CycleGAN = _cg_mod.CycleGAN

# ── Paths ─────────────────────────────────────────────────────────────────
ABC_TEST   = Path("data/abc/abc_test.hdf")
REAL_CSV   = Path("mirage_processed_corpus_v0.1/WASP39b/WASP39b_final_standardized.csv")
CKPT       = Path("configs/cyclegan/cyclegan_best.pt")
N_BINS     = 52


# ── Data loading ──────────────────────────────────────────────────────────

def load_abc(n: int) -> np.ndarray:
    with h5py.File(ABC_TEST) as f:
        flux = f["flux"][:n].astype(np.float32)
        if "flux_mean" in f and "flux_std" in f:
            flux = (flux - f["flux_mean"][:]) / f["flux_std"][:]
    return flux


def load_real(n_bins: int = N_BINS) -> np.ndarray:
    df = pd.read_csv(REAL_CSV)
    wav_col  = next((c for c in df.columns if "wavelength" in c.lower()), None)
    flux_col = next((c for c in df.columns if "flux" in c.lower()
                     and "error" not in c.lower()), None)
    df = df[[wav_col, flux_col]].dropna()
    df.columns = ["wavelength", "flux"]
    df = df[df["flux"] > 0].sort_values("wavelength")
    edges = np.linspace(df["wavelength"].min(), df["wavelength"].max(), n_bins + 1)
    df["bin"] = pd.cut(df["wavelength"], bins=edges, labels=False)
    binned = df.groupby("bin")["flux"].mean().values.astype(np.float32)
    if len(binned) < n_bins:
        binned = np.pad(binned, (0, n_bins - len(binned)),
                        constant_values=float(np.nanmedian(binned)))
    binned = binned[:n_bins]
    binned = np.nan_to_num(binned, nan=float(np.nanmedian(binned)))
    # Normalise to same scale as ABC
    binned = (binned - binned.mean()) / (binned.std() + 1e-8)
    return binned


def load_cyclegan() -> CycleGAN:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    model = CycleGAN(n_bins=cfg["n_bins"], n_res=cfg["n_res"],
                     ngf=cfg["ngf"], ndf=cfg["ndf"])
    model.G_AB.load_state_dict(ckpt["G_AB"])
    model.G_BA.load_state_dict(ckpt["G_BA"])
    model.eval()
    return model


# ── Metrics ───────────────────────────────────────────────────────────────

def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """Maximum Mean Discrepancy with RBF kernel — proxy for distributional distance."""
    def rbf(A, B):
        diff = A[:, None, :] - B[None, :, :]
        return np.exp(-np.sum(diff**2, axis=-1) / (2 * sigma**2))
    n, m = len(X), len(Y)
    kxx = rbf(X, X)
    kyy = rbf(Y, Y)
    kxy = rbf(X, Y)
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def cycle_consistency_error(model: CycleGAN, flux: np.ndarray) -> float:
    """Mean L1 error of round-trip G_BA(G_AB(x)) vs x."""
    x = torch.from_numpy(flux).float()
    with torch.no_grad():
        fake_B   = model.G_AB(x)
        rec_A    = model.G_BA(fake_B)
    return float((rec_A - x).abs().mean())


def per_bin_shift(orig: np.ndarray, translated: np.ndarray) -> np.ndarray:
    """Mean absolute shift per wavelength bin after translation."""
    return np.abs(translated.mean(axis=0) - orig.mean(axis=0))


def corr_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius similarity between correlation matrices of two datasets."""
    ca = np.corrcoef(A.T)
    cb = np.corrcoef(B.T)
    diff = ca - cb
    return float(1.0 - np.linalg.norm(diff, 'fro') / (np.linalg.norm(ca, 'fro') + 1e-8))


def kl_divergence_marginals(P: np.ndarray, Q: np.ndarray, bins: int = 50) -> float:
    """Mean KL divergence across marginal (per-bin) flux distributions."""
    kls = []
    for i in range(P.shape[1]):
        lo = min(P[:, i].min(), Q[:, i].min())
        hi = max(P[:, i].max(), Q[:, i].max())
        edges = np.linspace(lo, hi, bins + 1)
        p_hist, _ = np.histogram(P[:, i], bins=edges, density=True)
        q_hist, _ = np.histogram(Q[:, i], bins=edges, density=True)
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10
        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()
        kls.append(float(np.sum(p_hist * np.log(p_hist / q_hist))))
    return float(np.mean(kls))


# ── Domain randomisation simulation ──────────────────────────────────────

def apply_domain_randomisation(flux: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Simulate Phase 2 domain randomisation (correlated noise injection).
    This is the standalone proxy for the noisecond_cov arm.
    """
    rng = np.random.default_rng(seed)
    result = flux.copy()
    wlen = np.linspace(0.55, 7.30, flux.shape[1])

    for i in range(len(flux)):
        sigma = rng.uniform(0.05, 0.30)
        rho   = rng.uniform(0.30, 0.80)
        se_l  = rng.uniform(0.10, 1.00)
        ou_l  = rng.uniform(0.50, 3.00)

        d = np.abs(wlen[:, None] - wlen[None, :])
        var = sigma ** 2
        cov = ((1 - rho) * var * np.eye(len(wlen))
               + 0.5 * rho * var * np.exp(-0.5 * (d / se_l)**2)
               + 0.5 * rho * var * np.exp(-d / ou_l))
        cov += 1e-8 * var * np.eye(len(wlen))

        try:
            chol = np.linalg.cholesky(cov)
            noise = chol @ rng.standard_normal(len(wlen))
            result[i] += noise.astype(np.float32)
        except np.linalg.LinAlgError:
            result[i] += rng.standard_normal(len(wlen)).astype(np.float32) * sigma

    return result


# ── Main ──────────────────────────────────────────────────────────────────

def main(n_samples: int = 500) -> None:
    print(f"\n{'='*72}")
    print("  Phase 3 Track 2 — CycleGAN Ablation (standalone)")
    print(f"{'='*72}")

    # Verify files
    for p, name in [(ABC_TEST, "abc_test.hdf"), (REAL_CSV, "WASP39b CSV"), (CKPT, "cyclegan_best.pt")]:
        status = "✓" if p.exists() else "✗ MISSING"
        print(f"  {status}  {name}")
    if not all(p.exists() for p in [ABC_TEST, REAL_CSV, CKPT]):
        raise SystemExit("Missing required files.")

    print(f"\n  Loading data ({n_samples} ABC samples) ...")
    sim_raw   = load_abc(n_samples)
    real_ref  = load_real()

    # Normalise ABC to same scale as real
    sim_norm = (sim_raw - sim_raw.mean(axis=0)) / (sim_raw.std(axis=0) + 1e-8)
    real_rep = np.tile(real_ref, (len(sim_norm), 1))  # replicate for comparison

    print("  Loading CycleGAN model ...")
    model = load_cyclegan()

    print("  Running translations ...")
    with torch.no_grad():
        sim_t  = torch.from_numpy(sim_norm.astype(np.float32))
        trans  = model.G_AB(sim_t).numpy()   # CycleGAN translated

    domain_rand = apply_domain_randomisation(sim_norm, seed=42)

    print("\n  Computing metrics ...\n")

    # ── 1. Cycle-consistency error ─────────────────────────────────────────
    cyc_err = cycle_consistency_error(model, sim_norm.astype(np.float32))

    # ── 2. MMD to real domain ──────────────────────────────────────────────
    # Use PCA-reduced (10-dim) features to make MMD tractable
    from numpy.linalg import svd
    combined = np.vstack([sim_norm, real_rep, trans, domain_rand])
    _, _, Vt = svd(combined - combined.mean(axis=0), full_matrices=False)
    V10 = Vt[:10].T

    sim_pca   = sim_norm    @ V10
    real_pca  = real_rep    @ V10
    trans_pca = trans        @ V10
    dr_pca    = domain_rand  @ V10

    mmd_sim_real   = mmd_rbf(sim_pca,   real_pca,  sigma=1.0)
    mmd_trans_real = mmd_rbf(trans_pca, real_pca,  sigma=1.0)
    mmd_dr_real    = mmd_rbf(dr_pca,    real_pca,  sigma=1.0)

    # ── 3. KL divergence of marginals ─────────────────────────────────────
    kl_sim_real   = kl_divergence_marginals(sim_norm[:200],    real_rep[:200])
    kl_trans_real = kl_divergence_marginals(trans[:200],        real_rep[:200])
    kl_dr_real    = kl_divergence_marginals(domain_rand[:200],  real_rep[:200])

    # ── 4. Per-bin shift ──────────────────────────────────────────────────
    shift_sim   = per_bin_shift(real_rep, sim_norm).mean()
    shift_trans = per_bin_shift(real_rep, trans).mean()
    shift_dr    = per_bin_shift(real_rep, domain_rand).mean()

    # ── 5. Correlation structure similarity ───────────────────────────────
    corr_sim   = corr_similarity(sim_norm[:200],   real_rep[:200])
    corr_trans = corr_similarity(trans[:200],        real_rep[:200])
    corr_dr    = corr_similarity(domain_rand[:200],  real_rep[:200])

    # ── Print results table ───────────────────────────────────────────────
    print(f"  {'─'*70}")
    print(f"  Phase 3 Track 2 — CycleGAN vs Domain Randomisation")
    print(f"  (n={n_samples} ABC test spectra, real = WASP-39b PRISM collapsed spectrum)")
    print(f"  {'─'*70}")
    print(f"  {'condition':<22} {'MMD↓':>8} {'KL↓':>8} {'shift↓':>8} {'corr↑':>8}")
    print(f"  {'─'*70}")
    print(f"  {'sim (no adaptation)':<22} {mmd_sim_real:>8.4f} {kl_sim_real:>8.4f} {shift_sim:>8.4f} {corr_sim:>8.4f}")
    print(f"  {'domain_random':<22} {mmd_dr_real:>8.4f} {kl_dr_real:>8.4f} {shift_dr:>8.4f} {corr_dr:>8.4f}")
    print(f"  {'cyclegan_trans':<22} {mmd_trans_real:>8.4f} {kl_trans_real:>8.4f} {shift_trans:>8.4f} {corr_trans:>8.4f}")
    print(f"  {'─'*70}")
    print(f"  Cycle-consistency error: {cyc_err:.6f} (↓ better; <0.05 = tight)")

    # ── Comparative analysis ──────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  Comparative Analysis (D4 ablation)")
    print(f"  {'─'*70}")

    delta_mmd = mmd_dr_real - mmd_trans_real
    print(f"  MMD: domain_random − cyclegan = {delta_mmd:+.4f}")
    if delta_mmd > 0.001:
        print(f"    → CycleGAN is CLOSER to real domain (MMD lower by {delta_mmd:.4f})")
    elif delta_mmd < -0.001:
        print(f"    → Domain randomisation is CLOSER to real domain")
    else:
        print(f"    → Essentially EQUAL (|Δ| < 0.001) — both approaches equivalent")

    delta_kl = kl_dr_real - kl_trans_real
    print(f"  KL:  domain_random − cyclegan = {delta_kl:+.4f}")
    if delta_kl > 0.001:
        print(f"    → CycleGAN distribution CLOSER to real (KL lower by {delta_kl:.4f})")
    elif delta_kl < -0.001:
        print(f"    → Domain randomisation distribution closer to real")
    else:
        print(f"    → Essentially EQUAL — both approaches equivalent")

    print(f"\n  {'─'*70}")
    print(f"  D4 Conclusion:")
    if abs(delta_mmd) < 0.005 and abs(delta_kl) < 0.005:
        print(f"  ✓ domain_random ≈ cyclegan_trans — D4 VALIDATED")
        print(f"    Structured randomisation is sufficient; CycleGAN adds no value.")
        conclusion = "D4_VALIDATED"
    elif delta_mmd > 0.005 or delta_kl > 0.005:
        print(f"  ⚠ CycleGAN brings sim closer to real — consider using translation")
        conclusion = "CYCLEGAN_BENEFICIAL"
    else:
        print(f"  ✓ domain_random ≥ cyclegan_trans — domain randomisation preferred")
        conclusion = "RANDOMISATION_PREFERRED"
    print(f"  {'─'*70}\n")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "n_samples": n_samples,
        "conditions": {
            "sim_baseline": {
                "mmd_to_real": mmd_sim_real, "kl_to_real": kl_sim_real,
                "mean_shift": float(shift_sim), "corr_similarity": corr_sim
            },
            "domain_random": {
                "mmd_to_real": mmd_dr_real, "kl_to_real": kl_dr_real,
                "mean_shift": float(shift_dr), "corr_similarity": corr_dr
            },
            "cyclegan_trans": {
                "mmd_to_real": mmd_trans_real, "kl_to_real": kl_trans_real,
                "mean_shift": float(shift_trans), "corr_similarity": corr_trans,
                "cycle_consistency_error": cyc_err
            }
        },
        "deltas": {
            "mmd_dr_minus_cg":  float(delta_mmd),
            "kl_dr_minus_cg":   float(delta_kl),
        },
        "conclusion": conclusion
    }

    out_path = Path("configs/cyclegan/ablation_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=500)
    args = p.parse_args()
    main(n_samples=args.n_samples)
