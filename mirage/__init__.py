"""
MIRAGE — sim-to-real domain-adaptive JWST atmospheric retrieval (Track 1).

Importing this package registers MIRAGE's components (SpectraEncoder,
CorrelatedNoiseGenerator / InjectCorrelatedNoise, ABC theta stats) into a
pristine fm4ar. Put `import mirage` before any model/dataset construction.
"""

from mirage.register import register

register()
