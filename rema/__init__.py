"""rema finds galaxy clusters with a red-sequence matched filter, in JAX.

It takes the galaxies of the DESI Legacy Imaging Surveys DR11 and produces cluster catalogues by
blind cluster finding, richness scans at given positions, and spectroscopic post-processing
(cluster redshift and velocity dispersion), with a red-sequence model calibrated on DR11. It is a
redMaPPer-inspired redesign (Rykoff et al. 2014, 2016).

Subpackages are imported on demand, so ``import rema`` stays light.
"""

__version__ = "0.6.0"

__all__ = ["__version__"]
