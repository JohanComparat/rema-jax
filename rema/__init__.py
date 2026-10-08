"""rema: red-sequence matched-filter galaxy cluster finding in JAX.

A redMaPPer-inspired redesign (Rykoff et al. 2014, 2016) for the DESI Legacy Imaging Surveys
DR11: blind cluster finding, richness scans at given positions, and
spectroscopic post-processing (cluster redshift and velocity dispersion), with a red-sequence
model calibrated on DR11.

Subpackages are imported on demand, so ``import rema`` stays light.
"""

__version__ = "0.4.0"

__all__ = ["__version__"]
