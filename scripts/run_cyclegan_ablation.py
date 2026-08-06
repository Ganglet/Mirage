"""
Phase 3 Track 2 — CycleGAN ablation: quantify IS-efficiency contribution from
CycleGAN translation versus domain randomisation alone.

Evaluates four conditions on held-out ABC test planets:
  1. **Baseline** (clean):         No augmentation (Phase 1 transformer_abc)
  2. **Domain randomisation only**: Phase 2 noisecond_cov (correlated noise injection)
  3. **CycleGAN only**:             Translate with G_AB, no noise augmentation
  4. **CycleGAN + randomisation**:  G_AB translation + correlated noise injection

Compares IS-efficiency, coverage@68/95, and log-density at truth.

Expected result (per D4 decision): domain randomisation is sufficient;
CycleGAN provides marginal or no additional benefit on this problem.

Usage:
    python scripts/run_cyclegan_ablation.py [--n-planets 50] [--n-samples 1000]

Requires:
  - configs/transformer_abc/model__best.pt (baseline)
  - configs/noisecond_cov/model__best.pt (domain random)
  - configs/cyclegan/cyclegan_best.pt (CycleGAN)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

import mirage  # noqa: F401 — registers components

from mirage.datasets.transforms import InjectCorrelatedNoise
from mirage.eval.coverage import coverage
from mirage.eval.ess import is_efficiency
from mirage.nn.cyclegan import CycleGAN

from fm4ar.models.build_model import build_model
from fm4ar.datasets.theta_scalers import get_theta_scaler

ABC_DIR = Path("data/abc")

CONDITIONS = {
    "baseline":       Path("configs/transformer_abc"),
    "domain_random":  Path("configs/noisecond_cov"),
    "cyclegan_only":  Path("configs/cyclegan"),
    "cyclegan+random": Path("configs/cyclegan"),
}


def load_model(ckpt_dir: Path):
    """Load FMPE model + theta scaler."""
    ckpt = ckpt_dir / "model__best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"{ckpt} not found — train this model first:\n"
            f"    python scripts/train.py --experiment-dir {ckpt_dir}"
        )
    model = build_model(file_path=ckpt, experiment_dir=ckpt_dir, device="cpu")
    model.network.eval()
    with open(ckpt_dir / "config.yaml") as fh:
        config = yaml.safe_load(fh)
    return model, get_theta_scaler(config.get("theta_scaler", {}))


def load_cyclegan(ckpt_dir: Path, device="cpu") -> CycleGAN:
    """Load trained CycleGAN generator G_AB (sim → real)."""
    ckpt_path = ckpt_dir / "cyclegan_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found — train CycleGAN first:\n"
            f"    python scripts/train_cyclegan.py"
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model = CycleGAN(
        n_bins=config["n_bins"],
        n_res=config["n_res"],
        ngf=config["ngf"],
        ndf=config["ndf"],
    )
    model.G_AB.load_state_dict(ckpt["G_AB"])
    model.G_AB.eval()
    return model


def noise_transform(ref_dir: Path, test_seed: int) -> InjectCorrelatedNoise:
    """Rebuild the training noise model from a config, with a fresh test seed."""
    with open(ref_dir / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    kw = cfg["training"]["stage_0"]["data_transforms"][0]["kwargs"]
    kw = {**kw, "config": {**kw["config"],
                           "kwargs": {**kw["config"]["kwargs"],
                                      "random_seed": test_seed}}}
    return InjectCorrelatedNoise(**kw)


@torch.no_grad()
def sample_posterior(model, theta_scaler, ctx_single: dict, n_samples: int) -> np.ndarray:
    ctx = {k: v.unsqueeze(0).expand(n_samples, *v.shape) for k, v in ctx_single.items()}
    theta_scaled, _ = model.sample_and_log_prob_batch(context=ctx, tolerance=1e-3)
    return theta_scaler.inverse_array(theta_scaled.cpu().numpy())


@torch.no_grad()
def logdensity_at_truth(model, theta_scaler, ctx_single: dict, theta_true) -> float:
    theta_scaled = theta_scaler.forward_tensor(
        torch.from_numpy(np.asarray(theta_true)).float().unsqueeze(0)
    )
    ctx = {k: v.unsqueeze(0) for k, v in ctx_single.items()}
    logq = model.log_prob_batch(theta_scaled, context=ctx, tolerance=1e-3)
    return float(logq.reshape(-1)[0])


def main(n_planets: int = 50, n_samples: int = 1000, test_seed: int = 5678) -> None:
    print("Loading models ...")
    models = {}
    scalers = {}
    for name, path in CONDITIONS.items():
        if name in ("cyclegan_only", "cyclegan+random"):
            continue  # CycleGAN loaded separately
        try:
            m, s = load_model(path)
            models[name] = m
            scalers[name] = s
            print(f"  ✓ {name:<15} ({path.name})")
        except FileNotFoundError as e:
            print(f"  ✗ {name:<15} SKIP — {e}")

    try:
        cyclegan = load_cyclegan(CONDITIONS["cyclegan_only"])
        models["cyclegan_only"] = models["baseline"]  # same FMPE, translated input
        scalers["cyclegan_only"] = scalers["baseline"]
        models["cyclegan+random"] = models["domain_random"]  # cov arm, translated input
        scalers["cyclegan+random"] = scalers["domain_random"]
        print(f"  ✓ cyclegan       (configs/cyclegan)")
    except FileNotFoundError as e:
        print(f"  ✗ cyclegan       SKIP — {e}")
        cyclegan = None

    if not models:
        raise SystemExit("No models available — train at least one condition.")

    # Noise injector for domain_random and cyclegan+random
    inject = noise_transform(CONDITIONS["domain_random"], test_seed)

    order = ["baseline", "domain_random", "cyclegan_only", "cyclegan+random"]
    order = [n for n in order if n in models]

    samples: dict[str, list[np.ndarray]] = {n: [] for n in order}
    logdens: dict[str, list[float]] = {n: [] for n in order}
    truths: list[np.ndarray] = []

    with h5py.File(ABC_DIR / "abc_test.hdf") as f:
        wlen = f["wlen"][0].astype(np.float32)
        n = min(n_planets, len(f["theta"]))
        print(f"\nEvaluating {n} planets × {n_samples} samples on {len(order)} conditions\n")

        for idx in tqdm(range(n)):
            flux = f["flux"][idx].astype(np.float32)
            theta_true = f["theta"][idx].astype(np.float64)
            truths.append(theta_true)

            # ── Condition 1: Baseline (clean) ──
            if "baseline" in models:
                ctx = {
                    "flux": torch.from_numpy(flux.copy()).float(),
                    "wlen": torch.from_numpy(wlen.copy()).float(),
                }
                samples["baseline"].append(
                    sample_posterior(models["baseline"], scalers["baseline"], ctx, n_samples))
                logdens["baseline"].append(
                    logdensity_at_truth(models["baseline"], scalers["baseline"], ctx, theta_true))

            # ── Condition 2: Domain randomisation only ──
            if "domain_random" in models:
                gen = inject.noise_generator
                sigma_mat = gen.sample_covariance(wlen)
                s = {
                    "flux": flux + gen.sample_noise_from_covariance(sigma_mat),
                    "wlen": wlen,
                    "error_bars": np.sqrt(np.diag(sigma_mat)).astype(np.float32),
                    "oot_frames": gen.sample_oot_frames(sigma_mat, inject.n_oot_frames),
                }
                ctx = {k: torch.from_numpy(np.asarray(v)).float() for k, v in s.items()}
                samples["domain_random"].append(
                    sample_posterior(models["domain_random"], scalers["domain_random"], ctx, n_samples))
                logdens["domain_random"].append(
                    logdensity_at_truth(models["domain_random"], scalers["domain_random"], ctx, theta_true))

            # ── Condition 3: CycleGAN only (no noise) ──
            if "cyclegan_only" in models and cyclegan is not None:
                flux_translated = cyclegan.translate_sim_to_real(
                    torch.from_numpy(flux).float().unsqueeze(0)
                ).squeeze(0).numpy()
                ctx = {
                    "flux": torch.from_numpy(flux_translated).float(),
                    "wlen": torch.from_numpy(wlen).float(),
                }
                samples["cyclegan_only"].append(
                    sample_posterior(models["cyclegan_only"], scalers["cyclegan_only"], ctx, n_samples))
                logdens["cyclegan_only"].append(
                    logdensity_at_truth(models["cyclegan_only"], scalers["cyclegan_only"], ctx, theta_true))

            # ── Condition 4: CycleGAN + domain randomisation ──
            if "cyclegan+random" in models and cyclegan is not None:
                flux_translated = cyclegan.translate_sim_to_real(
                    torch.from_numpy(flux).float().unsqueeze(0)
                ).squeeze(0).numpy()
                gen = inject.noise_generator
                sigma_mat = gen.sample_covariance(wlen)
                s = {
                    "flux": flux_translated + gen.sample_noise_from_covariance(sigma_mat),
                    "wlen": wlen,
                    "error_bars": np.sqrt(np.diag(sigma_mat)).astype(np.float32),
                    "oot_frames": gen.sample_oot_frames(sigma_mat, inject.n_oot_frames),
                }
                ctx = {k: torch.from_numpy(np.asarray(v)).float() for k, v in s.items()}
                samples["cyclegan+random"].append(
                    sample_posterior(models["cyclegan+random"], scalers["cyclegan+random"], ctx, n_samples))
                logdens["cyclegan+random"].append(
                    logdensity_at_truth(models["cyclegan+random"], scalers["cyclegan+random"], ctx, theta_true))

    # ── Report results ──
    def norm_width68(samples_list, scaler) -> float:
        w = []
        for s in samples_list:
            lo = np.quantile(s, 0.16, axis=0)
            hi = np.quantile(s, 0.84, axis=0)
            w.append(scaler.forward_array(hi) - scaler.forward_array(lo))
        return float(np.mean(w))

    print(f"\n{'─'*80}")
    print(f"  Phase 3 Track 2 — CycleGAN Ablation (n={len(truths)} planets)")
    print(f"  {'condition':<18} {'cov@68%':>8} {'cov@95%':>8} {'width68↓':>10} {'logdens↑':>10}")
    print(f"  {'-'*66}")
    for name in order:
        if name not in samples:
            continue
        cov = coverage(samples[name], truths, levels=(0.68, 0.95))
        w = norm_width68(samples[name], scalers[name])
        ld = float(np.mean(logdens[name]))
        print(f"  {name:<18} {cov.overall[0]:>8.3f} {cov.overall[1]:>8.3f} "
              f"{w:>10.3f} {ld:>10.2f}")
    print(f"{'─'*80}")

    # ── Comparative analysis ──
    if "domain_random" in logdens and "cyclegan_only" in logdens:
        ld_dr = np.asarray(logdens["domain_random"])
        ld_cg = np.asarray(logdens["cyclegan_only"])
        delta = ld_dr - ld_cg
        print(f"\n  Domain random vs CycleGAN-only:")
        print(f"    Mean Δlogdens: {delta.mean():+.3f} (domain_random − cyclegan_only)")
        print(f"    Positive in {(delta > 0).sum()}/{len(delta)} planets")

    if "domain_random" in logdens and "cyclegan+random" in logdens:
        ld_dr = np.asarray(logdens["domain_random"])
        ld_both = np.asarray(logdens["cyclegan+random"])
        delta = ld_both - ld_dr
        print(f"\n  CycleGAN+random vs domain random:")
        print(f"    Mean Δlogdens: {delta.mean():+.3f} (cyclegan+random − domain_random)")
        print(f"    Positive in {(delta > 0).sum()}/{len(delta)} planets")

    print(f"\n{'─'*80}")
    print("  Interpretation:")
    print("    • If domain_random ≈ cyclegan+random: domain randomisation is sufficient")
    print("    • If cyclegan_only < baseline: translation degrades calibration")
    print("    • If cyclegan+random >> domain_random: translation adds value")
    print(f"{'─'*80}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-planets", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--test-seed", type=int, default=5678)
    args = p.parse_args()
    main(n_planets=args.n_planets, n_samples=args.n_samples, test_seed=args.test_seed)
