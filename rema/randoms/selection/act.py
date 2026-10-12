"""ACT DR6 clusters (ACT Collaboration 2025): selection by the S/N of the reference filter.

nemo detects clusters with a matched filter of fixed angular scale (2.4 arcmin, the "fixed"
columns of the catalogue); a cluster of central Compton parameter y0 is detected at a position
x with probability

    P(x) = Phi(y0 / sigma_y0(x) - q_cut),

with sigma_y0 the noise of that filter (nemo's RMS map) and q_cut the S/N threshold (5 for the
DR6 cosmology-grade sample). The measured fixed_y_c (in 1e-4) stands for y0; the area mask
(and the flag mask, when given) bounds the footprint. The RMS map is reduced once to a
HEALPix map (``rema cluster-randoms selfunc act-dr6``, :mod:`.carmaps`).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from ...sky.maps import SparseMap
from .base import ExternalSelection, eddington_average, register


@register("act_dr6")
class ACTDR6Selection(ExternalSelection):
    """``selfn``: FITS of a SparseMap (HDU RMS, column RMS = sigma of fixed_y_c, in 1e-4) built
    from the area-masked RMS map; positions outside it are outside the survey."""

    columns = ("EXT_FIXED_Y_C", "EXT_FIXED_SNR", "EXT_FLAGS")

    def __init__(self, selfn: str | None = None, q_cut: float = 5.0, require_flags0: bool = True,
                 eddington_alpha: float | None = None, rms_map: SparseMap | None = None):
        self.selfn = selfn
        self.q_cut = float(q_cut)
        self.require_flags0 = bool(require_flags0)
        self.alpha = eddington_alpha
        self.rms = rms_map if rms_map is not None else SparseMap.from_hdu(selfn, "RMS")

    def sigma(self, ra, dec) -> np.ndarray:
        return self.rms.lookup_np(np.asarray(ra, np.float64), np.asarray(dec, np.float64), "RMS")

    def in_footprint(self, ra, dec) -> np.ndarray:
        return self.sigma(ra, dec) > 0

    def own_cut(self, cat: dict) -> np.ndarray:
        ok = np.asarray(cat["EXT_FIXED_SNR"], np.float64) >= self.q_cut
        if self.require_flags0 and "EXT_FLAGS" in cat:
            ok &= np.asarray(cat["EXT_FLAGS"]) == 0
        return ok

    def p_detect(self, src: dict, ra, dec) -> np.ndarray:
        sig = self.sigma(ra, dec)
        y = np.asarray(src["EXT_FIXED_Y_C"], np.float64)
        good = sig > 0
        out = np.zeros(sig.shape)
        if self.alpha is None:
            out[good] = ndtr(y[good] / sig[good] - self.q_cut)
            return out
        # Average over the true y0 given the measurement (its error is the noise at the source's
        # own position, fixed_y_c / fixed_SNR).
        snr = np.asarray(src["EXT_FIXED_SNR"], np.float64)
        sy = y / np.maximum(snr, 1e-6)
        s = sig[good]
        out[good] = eddington_average(lambda yt: ndtr(yt / s[:, None] - self.q_cut), y[good], sy[good],
                                      self.alpha)
        return out

    def provenance(self) -> dict:
        return {**super().provenance(), "SELFN": str(self.selfn), "QCUT": self.q_cut}
