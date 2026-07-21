"""
InjectCorrelatedNoise: MIRAGE Phase 2 (WI-2) data transform.

Subclasses fm4ar's `DataTransform` and is registered into fm4ar's
`get_data_transforms` by `mirage.register` (transform type
"InjectCorrelatedNoise"); fm4ar itself stays unmodified.
"""

from collections.abc import Mapping

import numpy as np

from fm4ar.datasets.data_transforms import DataTransform

from mirage.datasets.noise import CorrelatedNoiseGenerator, get_noise_generator


class InjectCorrelatedNoise(DataTransform):
    """
    Phase 2 (WI-2) correlated-noise injection. Per sample, draw one
    wavelength covariance Σ and use it for three consistent outputs:

      - `flux`        += a correlated realisation ~ N(0, Σ)
      - `error_bars`  = √diag(Σ)                       (feeds WI-1)
      - `oot_frames`  = K noise-only draws ~ N(0, Σ)   (feeds WI-3)

    Sharing a single Σ is the point: the error bars and out-of-transit
    frames describe the same noise process that was added to the spectrum,
    mirroring a real JWST visit. The downstream covariance embedding (WI-3)
    estimates Σ̂ from `oot_frames` and never sees Σ directly (P2-D3).
    """

    def __init__(self, config: dict, n_oot_frames: int = 100) -> None:
        """
        Args:
            config: Noise-generator config (`type` + `kwargs`), passed to
                mirage's `get_noise_generator`. Must resolve to a
                `CorrelatedNoiseGenerator`.
            n_oot_frames: Number of out-of-transit frames to emit per sample.
        """

        super().__init__()

        self.noise_generator = get_noise_generator(config=config)
        if not isinstance(self.noise_generator, CorrelatedNoiseGenerator):
            raise TypeError(
                "InjectCorrelatedNoise requires a CorrelatedNoiseGenerator; "
                f"got {type(self.noise_generator).__name__}."
            )
        self.n_oot_frames = int(n_oot_frames)

    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Inject correlated noise and attach the matching σ and OOT frames.
        """

        output = dict(x)

        # One covariance for this sample; reused for all three outputs
        sigma_mat = self.noise_generator.sample_covariance(wlen=x["wlen"])

        noise = self.noise_generator.sample_noise_from_covariance(sigma_mat)
        output["flux"] = x["flux"] + noise  # new array — do not mutate input
        output["error_bars"] = np.sqrt(np.diag(sigma_mat)).astype(np.float32)

        # OOT frames only for arms that use them (WI-3 covariance embedding).
        # Set n_oot_frames=0 for the no-cond / +σ arms so this heavy (K, n_bins)
        # tensor is not produced/moved to the GPU — it destabilises MPS.
        if self.n_oot_frames > 0:
            output["oot_frames"] = self.noise_generator.sample_oot_frames(
                sigma_mat, n_frames=self.n_oot_frames
            )

        return output
