"""
Register MIRAGE components into pristine fm4ar via runtime patches.

fm4ar extends itself through hardcoded factory functions (string -> class
`match` statements) with no plugin hook, so MIRAGE teaches them about its
components here instead of editing fm4ar's source. Importing `mirage` runs
`register()` once. See P2-D8.
"""

from __future__ import annotations

import numpy as np

# ABC theta normalisation stats (from abc_train.hdf, 73,113 planets;
# params: [T, log_H2O, log_CO2, log_CH4, log_CO, log_NH3])
_ABC_MEAN = np.array([1201.1842, -5.9989, -6.5019, -5.9979, -4.4954, -6.4910])
_ABC_STD = np.array([681.0441, 1.7337, 1.4457, 1.7390, 0.8639, 1.4373])

# Path B 7-param cloud extension: θ = [T, 5×log-abundance, log10 P_cloud].
# Uniform-prior mean/std over the extended generation ranges (T[300,2500],
# logX[-9,-1], log P_cloud[2,6]); std = range/√12.
_ABC_CLOUD_MEAN = np.array([1400.0, -5.0, -5.0, -5.0, -5.0, -5.0, 4.0])
_ABC_CLOUD_STD = np.array([635.1, 2.309, 2.309, 2.309, 2.309, 2.309, 1.155])

# P3-D8 gradient forward model: θ = [T_surface, T_top, 5×log-abundance] (no cloud).
# Isothermal T is replaced by a 2-point vertical profile so the network can
# represent the P-T structure the isothermal model cannot (the diagnosed
# misspecification). Uniform-prior mean/std over the generation ranges
# (T_surface,T_top ∈ [300,2500], logX ∈ [-9,-1]); std = range/√12.
_ABC_GRAD_MEAN = np.array([1400.0, 1400.0, -5.0, -5.0, -5.0, -5.0, -5.0])
_ABC_GRAD_STD = np.array([635.1, 635.1, 2.309, 2.309, 2.309, 2.309, 2.309])

# P3-D11 radius model: θ = [rp_rj, T, 5×log-abundance]. Freeing the planet radius is
# what flipped the real WASP-39b retrieval physical (NS χ²=0.76 vs cold-flat 2.55) —
# the baseline degeneracy MIRAGE couldn't absorb with a fixed radius. Uniform-prior
# mean/std over generation ranges (rp_rj[0.8,1.6], T[300,2500], logX[-9,-1]); std=range/√12.
_ABC_RAD_MEAN = np.array([1.2, 1400.0, -5.0, -5.0, -5.0, -5.0, -5.0])
_ABC_RAD_STD = np.array([0.2309, 635.1, 2.309, 2.309, 2.309, 2.309, 2.309])

# P5 K2-18b radius model: cold sub-Neptune priors rp[0.10,0.40], T[100,600] (uniform,
# std=range/√12); same θ=[rp,T,5×logX] layout as abc_rad but a very different regime.
_ABC_RAD_K218_MEAN = np.array([0.25, 350.0, -5.0, -5.0, -5.0, -5.0, -5.0])
_ABC_RAD_K218_STD = np.array([0.0866, 144.3, 2.309, 2.309, 2.309, 2.309, 2.309])

_REGISTERED = False


def register() -> None:
    """Patch fm4ar's factories to recognise MIRAGE components (idempotent)."""

    global _REGISTERED
    if _REGISTERED:
        return

    import fm4ar.datasets.data_transforms as _dt
    import fm4ar.datasets.theta_scalers as _ts
    import fm4ar.nn.embedding_nets as _emb
    import fm4ar.training.stages as _stages

    # 1) Block registry. Callers in embedding_nets.py reference this as a
    #    module global, so patching the module attribute is enough.
    _orig_block = _emb.block_type_string_to_class

    def block_type_string_to_class(block_type: str) -> type:
        if block_type == "SpectraEncoder":
            from mirage.nn.spectra_encoder import SpectraEncoder
            return SpectraEncoder
        return _orig_block(block_type)

    _emb.block_type_string_to_class = block_type_string_to_class

    # 2) Theta normalisation stats. Same-module caller (theta_scalers.py:137).
    _orig_stats = _ts.get_mean_and_std

    def get_mean_and_std(dataset: str, **kwargs):
        if dataset == "abc":
            return _ABC_MEAN, _ABC_STD
        if dataset == "abc_cloud":
            return _ABC_CLOUD_MEAN, _ABC_CLOUD_STD
        if dataset == "abc_grad":
            return _ABC_GRAD_MEAN, _ABC_GRAD_STD
        if dataset == "abc_rad":
            return _ABC_RAD_MEAN, _ABC_RAD_STD
        if dataset == "abc_rad_k218":
            return _ABC_RAD_K218_MEAN, _ABC_RAD_K218_STD
        return _orig_stats(dataset, **kwargs)

    _ts.get_mean_and_std = get_mean_and_std

    # 3) Data transforms. stages.py did `from ... import get_data_transforms`,
    #    so rebind the name in the importing module too, not just on _dt.
    from mirage.datasets.transforms import InjectCorrelatedNoise

    _orig_transforms = _dt.get_data_transforms

    def get_data_transforms(configs):
        out = []
        for cfg in configs:
            if cfg.type == "InjectCorrelatedNoise":
                out.append(InjectCorrelatedNoise(**cfg.kwargs))
            else:
                out.extend(_orig_transforms([cfg]))  # AddNoise / Subsample
        return out

    _dt.get_data_transforms = get_data_transforms
    _stages.get_data_transforms = get_data_transforms

    _REGISTERED = True
