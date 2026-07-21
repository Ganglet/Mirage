"""
MIRAGE training entry point.

Registers MIRAGE's components into pristine fm4ar (via `import mirage`), then
runs fm4ar's `train_local.py` as __main__ with the same CLI args. Use this
instead of calling fm4ar's script directly, e.g.:

    python scripts/train.py --experiment-dir configs/transformer_abc
"""

import runpy
from pathlib import Path

import mirage  # noqa: F401 — registers SpectraEncoder / InjectCorrelatedNoise / abc stats

_TRAIN_LOCAL = (
    Path(__file__).resolve().parents[1]
    / "fm4ar" / "scripts" / "training" / "train_local.py"
)

if __name__ == "__main__":
    runpy.run_path(str(_TRAIN_LOCAL), run_name="__main__")
