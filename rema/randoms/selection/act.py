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


def build_rms_map(rms_path, area_path, flag_path=None, nside: int = 2048, max_flag: int = 0,
                  chunk_rows: int = 256) -> SparseMap:
    """HEALPix (NESTED) map of the nemo noise of the reference filter (column RMS, in the units
    of fixed_y_c), from the stitched CAR images of the DR6 selection function: the mean noise of
    the image pixels inside the area mask (and with flags <= ``max_flag``) that fall in each
    HEALPix pixel, read ``chunk_rows`` image rows at a time."""
    from astropy.io import fits
    from astropy.wcs import WCS

    from ...sky.healpix import ang2pix_nest_np

    npix = 12 * nside**2
    num = np.zeros(npix)
    cnt = np.zeros(npix)
    hs = [fits.open(p, memmap=True) for p in (rms_path, area_path, flag_path) if p is not None]
    try:
        rms, area = hs[0][0].data, hs[1][0].data
        flag = hs[2][0].data if flag_path is not None else None
        w = WCS(hs[0][0].header).celestial
        ny, nx = rms.shape
        for y0 in range(0, ny, chunk_rows):
            v = np.asarray(rms[y0:y0 + chunk_rows], np.float64)
            ok = np.isfinite(v) & (v > 0) & (np.asarray(area[y0:y0 + chunk_rows]) > 0)
            if flag is not None:
                ok &= np.asarray(flag[y0:y0 + chunk_rows]) <= max_flag
            if not ok.any():
                continue
            yy, xx = np.nonzero(ok)
            ra, dec = w.all_pix2world(xx, yy + y0, 0)
            p = ang2pix_nest_np(nside, np.mod(ra, 360.0), dec)
            num += np.bincount(p, weights=v[ok], minlength=npix)
            cnt += np.bincount(p, minlength=npix)
    finally:
        for h in hs:
            h.close()
    pix = np.flatnonzero(cnt > 0)
    return SparseMap(nside, pix, {"RMS": (num[pix] / cnt[pix]).astype(np.float32),
                                  "NIMG": cnt[pix].astype(np.int32)})


@register("act_dr6")
class ACTDR6Selection(ExternalSelection):
    """``selfn``: FITS of a SparseMap (HDU RMS, column RMS = sigma of fixed_y_c, in its unit of
    1e-4; ``cluster-randoms act-selfn``) built from the area- and flag-masked RMS map; positions
    outside it are outside the survey."""

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
