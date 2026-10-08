"""Galaxy photo-z as Gaussian redshift distributions.

The DR11 photo-z sweeps give, per galaxy, the median and the standard deviation of a random-forest
redshift distribution (Zhou et al. 2021, 2023), read at ingest as ZPHOT and ZPHOT_STD. Compared
with spectroscopic redshifts the standard deviation is too large, by a factor that depends on the
magnitude, so the photo-z of galaxy i is taken as the Gaussian

    p_i(z) = N(z; ZPHOT_i, s_i),   s_i = max(err_scale(m_i) ZPHOT_STD_i, err_floor (1 + ZPHOT_i)),

with ``err_scale`` measured on the spectroscopic galaxies (:func:`rema.validate.photoz.fit_err_scale`)
and given in :class:`rema.config.PhotozConfig`.
"""

from __future__ import annotations

import numpy as np

from ..config import PhotozConfig


def err_scale(refmag, pcfg: PhotozConfig) -> np.ndarray:
    """The factor applied to ZPHOT_STD at reference magnitude ``refmag``.

    >>> pc = PhotozConfig(err_scale=(0.8, 0.6), err_scale_mag=(19.0, 22.0))
    >>> err_scale([18.0, 20.5, 23.0], pc).round(3).tolist()
    [0.8, 0.7, 0.6]
    """
    m = np.asarray(refmag, np.float64)
    vals = np.asarray(pcfg.err_scale, np.float64)
    nodes = np.asarray(pcfg.err_scale_mag, np.float64)
    if nodes.size == 0:
        if vals.size != 1:
            raise ValueError("photoz.err_scale needs one value, or one per photoz.err_scale_mag node")
        return np.full(m.shape, vals[0])
    if nodes.size != vals.size or np.any(np.diff(nodes) <= 0):
        raise ValueError("photoz.err_scale_mag must increase and match photoz.err_scale")
    return np.interp(m, nodes, vals)


def photoz_valid(zphot, zphot_std) -> np.ndarray:
    """Galaxies with a usable photo-z (ingest writes -1 for a missing one)."""
    zp = np.asarray(zphot, np.float64)
    sd = np.asarray(zphot_std, np.float64)
    return np.isfinite(zp) & np.isfinite(sd) & (zp >= 0) & (sd > 0)


def photoz_sigma(zphot, zphot_std, refmag, pcfg: PhotozConfig) -> np.ndarray:
    """s_i, the calibrated photo-z width (float32); -1 for galaxies without a usable photo-z.

    >>> pc = PhotozConfig(err_scale=(0.5,), err_floor=0.01)
    >>> [round(float(s), 4) for s in photoz_sigma([0.3, 0.3, -1.0], [0.1, 0.01, 0.1], [20.0] * 3, pc)]
    [0.05, 0.013, -1.0]
    """
    zp = np.asarray(zphot, np.float64)
    sd = np.asarray(zphot_std, np.float64)
    ok = photoz_valid(zp, sd)
    s = np.maximum(err_scale(refmag, pcfg) * sd, pcfg.err_floor * (1.0 + np.where(ok, zp, 0.0)))
    return np.where(ok, s, -1.0).astype(np.float32)
