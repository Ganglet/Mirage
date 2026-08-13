"""
WI-4 — first real-data importance-sampling ESS for MIRAGE on WASP-39b.

Two-env pipeline (MIRAGE = `mirage` env / torch; TauREx likelihood = `mirage-taurex`):
  1. `--sample`  (mirage env): draw θ_i ~ q(θ|x_real) from the trained model on the
     real WASP-39b input (WI-2 adapter), record log q, and the raw depth + σ for the
     likelihood. Writes real_ess_samples.npz.
  2. TauREx forward on those θ with WASP-39b geometry  (see taurex_forward.py --forward-npz)
     → real_ess_forward.npz (model spectra).
  3. `--compute`: w_i = exp(log p(x|θ_i) − log q_i); ESS = (Σw)²/Σw², ε = ESS/N.

    conda activate mirage      && python scripts/real_ess.py --sample --n 500
    conda run -n mirage-taurex python scripts/taurex_forward.py --forward-npz scratch/real_ess_samples.npz
    conda activate mirage      && python scripts/real_ess.py --compute
"""

import argparse
from pathlib import Path

import numpy as np

OUT = Path("data/real_ess")           # gitignored (under data/)
SPECTRUM = "data/jwst_wasp39b_spectrum.csv"


ARMS = {"base": "configs/transformer_abc",
        "sigma": "configs/noisecond_sigma",
        "cov": "configs/noisecond_cov",
        "ext": "configs/transformer_abc_ext",        # Fix-1 extended-coverage retrain
        "shape": "configs/transformer_abc_shape",    # Fix-2 shape (baseline removed)
        "rad": "configs/noisecond_rad_cov",          # P3-D11 radius cov arm (θ=[rp,T,5mols])
        "rad_nocond": "configs/noisecond_rad_nocond",  # Phase-4 ablation: radius, no noise cond
        "rad_sigma": "configs/noisecond_rad_sigma",    # Phase-4 ablation: radius, σ-only
        "rad_hires": "configs/noisecond_rad_nocond_hires",  # P5: radius, 150-bin hi-res grid
        "k218": "configs/noisecond_rad_nocond_k218",   # P5 multi-target: K2-18b radius model
        "wasp96": "configs/noisecond_rad_nocond_wasp96"}  # P5 multi-target: WASP-96b (self-reduced NIRISS)
# per-arm real spectrum (default = WASP-39b); multi-target planets use their own published depths
SPECTRA = {"k218": "data/jwst_k2_18b_spectrum.csv",
           "wasp96": "data/jwst_wasp96b_spectrum.csv"}
# which training set's normalisation each arm expects (ext/shape/rad* differ from abc)
STATS = {"ext": "data/abc_ext/abc_ext_train.hdf",
         "shape": "data/abc_ext/abc_ext_shape_train.hdf",
         "rad": "data/abc_rad/abc_rad_train.hdf",
         "rad_nocond": "data/abc_rad/abc_rad_train.hdf",
         "rad_sigma": "data/abc_rad/abc_rad_train.hdf",
         "rad_hires": "data/abc_rad_hires/abc_rad_train.hdf",  # 150-bin grid + hi-res stats
         "k218": "data/abc_rad_k218/abc_rad_train.hdf",  # K2-18b 150-bin grid (0.85-5.17µm) + stats
         "wasp96": "data/abc_rad_wasp96/abc_rad_train.hdf"}  # WASP-96b 90-bin NIRISS grid (0.85-2.81µm)
STATS_DEFAULT = "data/abc/abc_test.hdf"
_COV_ARMS = ("cov", "rad")                            # arms with covariance embedding → need OOT frames


def _real_oot_frames(k=100):
    """Real PRISM OOT frames on the 52-bin grid, model (stored) bin order.
    Raw relative residuals — the cov arm (cov_whiten=True) whitens internally."""
    import pandas as pd
    from build_real_covariance import abc_grid, rebin_frames, OOT
    _, edges = abc_grid()                                  # ascending grid edges
    df = pd.read_csv(OOT / "NIRSPEC_CLEAR_PRISM_out_of_transit_full.csv")
    df = df[df["dq_flag"] == 0]
    M, ints = rebin_frames(df, edges)                      # (n_int, 52) ascending
    covered = np.isfinite(M).all(axis=0)
    rel = np.zeros_like(M)
    rel[:, covered] = M[:, covered] / M[:, covered].mean(axis=0) - 1.0
    sel = np.linspace(0, len(ints) - 1, k).astype(int)
    return rel[sel][:, ::-1].astype(np.float32)            # ascending → stored order


def sample(n, arm="base"):
    import torch, yaml, h5py
    import mirage  # noqa
    from mirage.datasets.real_jwst import abc_grid_and_norm, rebin_spectrum, build_observation
    from fm4ar.models.build_model import build_model
    from fm4ar.datasets.theta_scalers import get_theta_scaler

    ckpt_dir = Path(ARMS[arm])                      # base / +σ / +σ+cov arm
    model = build_model(file_path=ckpt_dir / "model__best.pt", experiment_dir=ckpt_dir, device="cpu")
    model.network.eval()
    with open(ckpt_dir / "config.yaml") as fh:
        scaler = get_theta_scaler(yaml.safe_load(fh).get("theta_scaler", {}))

    # real model-input (normalised with the arm's OWN training stats) + RAW depth/σ
    stats_hdf = STATS.get(arm, STATS_DEFAULT)
    spec_csv = SPECTRA.get(arm, SPECTRUM)           # multi-target: per-planet published depths
    ctx_np, covered = build_observation(spec_csv, stats_hdf, remove_baseline=(arm == "shape"))
    if arm in _COV_ARMS:                             # WI-3 / P3-D11 need real OOT frames
        ctx_np["oot_frames"] = _real_oot_frames()   # (K, 52), model bin order
    wl_s, mean_s, std_s = abc_grid_and_norm(stats_hdf)   # arm's OWN grid (52 or hi-res 150)
    order = np.argsort(wl_s)
    import pandas as pd
    depth_a, err_a, cov_a = rebin_spectrum(pd.read_csv(spec_csv), wl_s[order])
    inv = np.argsort(order)
    x_obs = depth_a[inv].astype(np.float64)        # raw (Rp/Rs)^2, stored order
    sig_obs = err_a[inv].astype(np.float64)

    ctx = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).expand(n, *np.shape(v))
           for k, v in ctx_np.items()}
    with torch.no_grad():
        theta_scaled, log_q = model.sample_and_log_prob_batch(context=ctx, tolerance=1e-3)
    theta_phys = scaler.inverse_array(theta_scaled.cpu().numpy())

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "real_ess_samples.npz",
             theta=theta_phys, log_q=log_q.cpu().numpy().reshape(-1),
             x_obs=x_obs, sig_obs=sig_obs, covered=covered, wlen=wl_s)
    print(f"[sample] {n} draws from '{arm}' model on real WASP-39b input")
    if arm.startswith("rad") or arm in ("k218", "wasp96"):   # radius-θ arms: [rp_rj, T, 5×logX]
        print(f"  θ means: radius={theta_phys[:,0].mean():.3f}RJ  T={theta_phys[:,1].mean():.0f}K"
              f"  logX={np.round(theta_phys[:,2:].mean(0),2)}   (NS anchor: R=1.23, T=606)")
    else:
        print(f"  θ means: T={theta_phys[:,0].mean():.0f}K  logX={np.round(theta_phys[:,1:].mean(0),2)}")
    print(f"  covered bins: {covered.sum()}/{len(covered)}   saved → {OUT/'real_ess_samples.npz'}")


def compute():
    s = np.load(OUT / "real_ess_samples.npz")
    fwd = np.load(OUT / "real_ess_forward.npz")
    x, sig, cov = s["x_obs"], s["sig_obs"], s["covered"]
    log_q = s["log_q"]
    model_spec = fwd["model_spec"]                 # (N, 52)
    m = cov & np.isfinite(sig) & (sig > 0)
    # Gaussian log-likelihood on covered bins
    # error budget MUST include the WI-1 forward-model systematic (~5% of depth),
    # else the tiny JWST σ makes the likelihood a needle → ESS collapses / NS stalls
    sig_eff = np.sqrt(sig ** 2 + (0.05 * x) ** 2)
    valid = np.isfinite(model_spec[:, m]).all(axis=1)     # unphysical samples → 0 weight
    r = (x[None, m] - model_spec[:, m]) / sig_eff[None, m]
    log_lik = -0.5 * np.nansum(r ** 2, axis=1)
    log_lik = np.where(valid, log_lik, -np.inf)
    logw = log_lik - log_q
    logw -= logw.max()
    w = np.exp(logw)
    ess = w.sum() ** 2 / np.sum(w ** 2)
    n = len(w)
    print(f"[compute] N={n} covered-bins={m.sum()}")
    print(f"  ESS = {ess:.1f}   ε = {ess/n*100:.3f}%   (D7: ESS>500 primary, >2500 high-quality)")
    print(f"  best-fit χ²/dof = {(-2*log_lik.max())/m.sum():.2f}")
    print(f"  {'clears ESS>500' if ess > 500 else 'below ESS>500 — quantifies the sim-to-real gap (expected on real data)'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--arm", choices=list(ARMS), default="base")
    a = ap.parse_args()
    if a.sample: sample(a.n, a.arm)
    elif a.compute: compute()
