"""
Phase 2 noise-conditioning ablation (WI-5).

Evaluates the three arms — no-cond / +σ / +σ+cov — on held-out ABC test
planets under the SAME injected correlated noise, and reports central-interval
coverage at 68%/95% (P2-D9: coverage, not IS-efficiency, because the NS
reference does not match the injected-noise likelihood).

Each planet is noised ONCE and the identical noisy context is fed to all arms,
so the comparison is head-to-head. Expected story: no-cond is overconfident
(coverage << nominal); +σ+cov restores coverage toward 68/95%.

Run from Project/ AFTER training the three arms:
  python scripts/run_noisecond_ablation.py [--n-planets 50] [--n-samples 1000]
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

import mirage  # noqa: F401 — registers SpectraEncoder / InjectCorrelatedNoise
from mirage.datasets.transforms import InjectCorrelatedNoise
from mirage.eval.coverage import coverage

from fm4ar.models.build_model import build_model
from fm4ar.datasets.theta_scalers import get_theta_scaler

ABC_DIR = Path("data/abc")
ARMS: dict[str, Path] = {
    "no-cond":    Path("configs/noisecond_nocond"),
    "+sigma":     Path("configs/noisecond_sigma"),
    "+sigma+cov": Path("configs/noisecond_cov"),
}


def load_arm(ckpt_dir: Path):
    ckpt = ckpt_dir / "model__best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"{ckpt} not found — train this arm first:\n"
            f"    python scripts/train.py --experiment-dir {ckpt_dir}"
        )
    model = build_model(file_path=ckpt, experiment_dir=ckpt_dir, device="cpu")
    model.network.eval()
    with open(ckpt_dir / "config.yaml") as fh:
        config = yaml.safe_load(fh)
    return model, get_theta_scaler(config.get("theta_scaler", {}))


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
    """
    log q(θ*|x) via the reverse ODE. Returned in the model's SCALED θ space;
    the scaler (same MeanStdScaler across arms) contributes a constant Jacobian
    offset that cancels in cross-arm comparison. Higher = better (proper score).
    """
    theta_scaled = theta_scaler.forward_tensor(
        torch.from_numpy(np.asarray(theta_true)).float().unsqueeze(0)
    )                                                   # (1, D), model space
    ctx = {k: v.unsqueeze(0) for k, v in ctx_single.items()}  # batch = 1
    logq = model.log_prob_batch(theta_scaled, context=ctx, tolerance=1e-3)
    return float(logq.reshape(-1)[0])


def main(n_planets: int = 50, n_samples: int = 1000, test_seed: int = 1234,
         clean_ref: bool = False, save: str | None = None) -> None:
    print("Loading arms (CPU; ODE solver needs float64) ...")
    arms = {}
    for name, d in ARMS.items():
        try:
            arms[name] = load_arm(d)
        except FileNotFoundError:
            print(f"  [skip] {name}: not trained yet ({d}/model__best.pt missing)")
    # Optional clean Phase-1 reference (configs/transformer_abc, NO injection):
    # the informative-width target the injected arms should sit between.
    clean = None
    if clean_ref:
        clean_dir = Path("configs/transformer_abc")
        try:
            clean = load_arm(clean_dir)
        except FileNotFoundError:
            print(f"  [skip] clean-ref: {clean_dir}/model__best.pt missing")

    if not arms and clean is None:
        raise SystemExit("Nothing to evaluate — train an arm or pass --clean-ref.")
    inject = noise_transform(ARMS["+sigma+cov"], test_seed) if arms else None

    order = (["clean-ref"] if clean else []) + list(arms)
    scalers = {name: arms[name][1] for name in arms}
    if clean:
        scalers["clean-ref"] = clean[1]
    samples: dict[str, list[np.ndarray]] = {name: [] for name in order}
    logdens: dict[str, list[float]] = {name: [] for name in order}
    truths: list[np.ndarray] = []
    planet_sigma: list[float] = []   # injected σ per planet, for the stratified view

    with h5py.File(ABC_DIR / "abc_test.hdf") as f:
        wlen = f["wlen"][0].astype(np.float32)
        n = min(n_planets, len(f["theta"]))
        print(f"Evaluating {n} planets × {n_samples} samples on {len(order)} models "
              f"(~30–60s/planet/model on CPU)\n")

        for idx in tqdm(range(n)):
            flux = f["flux"][idx].astype(np.float32)
            theta_true = f["theta"][idx].astype(np.float64)
            truths.append(theta_true)

            # clean reference: un-injected flux, flux/wlen only
            if clean:
                clean_ctx = {
                    "flux": torch.from_numpy(flux.copy()).float(),
                    "wlen": torch.from_numpy(wlen.copy()).float(),
                }
                samples["clean-ref"].append(
                    sample_posterior(clean[0], clean[1], clean_ctx, n_samples))
                logdens["clean-ref"].append(
                    logdensity_at_truth(clean[0], clean[1], clean_ctx, theta_true))

            # inject correlated noise ONCE (capture per-planet σ); identical
            # noisy context to all arms
            if arms:
                gen = inject.noise_generator
                sigma_mat, params = gen.sample_covariance(wlen, return_params=True)
                planet_sigma.append(params["sigma"])
                s = {
                    "flux": flux + gen.sample_noise_from_covariance(sigma_mat),
                    "wlen": wlen,
                    "error_bars": np.sqrt(np.diag(sigma_mat)).astype(np.float32),
                    "oot_frames": gen.sample_oot_frames(sigma_mat, inject.n_oot_frames),
                }
                ctx_single = {
                    k: torch.from_numpy(np.asarray(v)).float() for k, v in s.items()
                }
                for name, (model, scaler) in arms.items():
                    samples[name].append(
                        sample_posterior(model, scaler, ctx_single, n_samples)
                    )
                    logdens[name].append(
                        logdensity_at_truth(model, scaler, ctx_single, theta_true)
                    )

    def norm_width68(samples_list, scaler) -> float:
        # Width in θ-scaler units (≈ prior-σ), robust and n-independent: the
        # scaler's mean cancels in the (hi − lo) difference, leaving (hi−lo)/std.
        w = []
        for s in samples_list:
            lo = np.quantile(s, 0.16, axis=0)
            hi = np.quantile(s, 0.84, axis=0)
            w.append(scaler.forward_array(hi) - scaler.forward_array(lo))
        return float(np.mean(w))

    print(f"\n{'─'*66}")
    print(f"  Noise-conditioning ablation (n={len(truths)} planets)")
    print(f"  {'model':<12} {'cov@68%':>8} {'cov@95%':>8} {'width68↓':>10} {'logdens↑':>10}")
    print(f"  {'-'*56}")
    for name in order:
        cov = coverage(samples[name], truths, levels=(0.68, 0.95))
        w = norm_width68(samples[name], scalers[name])
        ld = float(np.mean(logdens[name]))
        print(f"  {name:<12} {cov.overall[0]:>8.3f} {cov.overall[1]:>8.3f} "
              f"{w:>10.3f} {ld:>10.2f}")
    print(f"{'─'*66}")
    print("  clean-ref = informative target (no injection; prior-width ≈ 2.36).")
    print("  Injected arms should sit between clean-ref and 2.36; +cov sharper /")
    print("  higher-logdens than no-cond = the conditioning win.")

    # ── per-σ stratified logdens: does conditioning help most at high noise? ──
    arm_names = list(arms)
    if arm_names and len(planet_sigma) >= 6:
        sig = np.asarray(planet_sigma)
        binned = np.digitize(sig, np.quantile(sig, [1 / 3, 2 / 3]))  # 0/1/2
        ld = {nm: np.asarray(logdens[nm]) for nm in arm_names}
        has_gap = {"no-cond", "+sigma+cov"} <= set(arm_names)

        print(f"\n{'─'*66}")
        print("  Per-σ stratified logdens↑ (tertiles by injected σ)")
        header = f"  {'σ-bin':<8} " + " ".join(f"{nm:>11}" for nm in arm_names)
        if has_gap:
            header += f"   {'cov−nocond':>10}"
        print(header)
        print(f"  {'-'*56}")
        for b, lab in enumerate(("low-σ", "mid-σ", "high-σ")):
            m = binned == b
            if not m.any():
                continue
            row = f"  {lab:<8} " + " ".join(f"{ld[nm][m].mean():>11.2f}" for nm in arm_names)
            if has_gap:
                row += f"   {ld['+sigma+cov'][m].mean() - ld['no-cond'][m].mean():>+10.2f}"
            print(f"{row}   (σ∈[{sig[m].min():.3f},{sig[m].max():.3f}])")
        print(f"{'─'*66}")
        print("  cov−nocond Δlogdens growing with σ ⇒ conditioning helps most where")
        print("  the correlated noise is worst — the adaptivity story, made explicit.")

    if save:                                   # Phase 4.2: persist arrays for figures
        safe = {nm: nm.replace("+", "p").replace("-", "_") for nm in samples}
        np.savez(save, truths=truths, planet_sigma=np.asarray(planet_sigma),
                 arms=np.array(list(samples.keys()), dtype=object),
                 arms_safe=np.array([safe[nm] for nm in samples], dtype=object),
                 **{f"s__{safe[nm]}": np.asarray(samples[nm]) for nm in samples},
                 **{f"ld__{safe[nm]}": np.asarray(logdens[nm]) for nm in logdens})
        print(f"  [saved] ablation arrays → {save}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-planets", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--test-seed", type=int, default=1234)
    p.add_argument("--clean-ref", action="store_true",
                   help="also evaluate the clean Phase-1 model (no injection) as reference")
    p.add_argument("--save", type=str, default=None, help="npz path to persist arrays for figures")
    args = p.parse_args()
    main(n_planets=args.n_planets, n_samples=args.n_samples,
         test_seed=args.test_seed, clean_ref=args.clean_ref, save=args.save)
