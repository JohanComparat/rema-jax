"""Planck PSZ2 (Planck Collaboration 2016): the survey (or cosmology) mask only.

The PSZ2 completeness is released averaged over the sky (a function of Y500, theta500 and the
S/N threshold, file HFI_PCCS_SZ-selfunc-*_R2.08.fits): within the mask it does not depend on
the position, so it cancels in the weights of each cluster's copies and only the mask (HEALPix,
RING, Galactic coordinates, extension 1) restricts the randoms. Variations of the Planck noise
across the sky are not modelled; a complete sub-sample (S/N well above the cut) is the safer
choice.
"""

from __future__ import annotations

import numpy as np

from .base import ExternalSelection, register


@register("psz2")
class PSZ2Selection(ExternalSelection):
    columns = ("EXT_SNR",)

    def __init__(self, mask: str | None = None, snr_min: float = 4.5, mask_map=None, nside: int | None = None):
        import healpy as hp

        self.mask_path = mask
        self.map = np.asarray(mask_map if mask_map is not None else hp.read_map(mask, field=0, nest=False,
                                                                            dtype=np.float64))
        self.nside = nside or hp.npix2nside(self.map.size)
        self.snr_min = float(snr_min)

    def in_footprint(self, ra, dec) -> np.ndarray:
        import healpy as hp
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        g = SkyCoord(np.asarray(ra, np.float64) * u.deg, np.asarray(dec, np.float64) * u.deg).galactic
        pix = hp.ang2pix(self.nside, g.l.deg, g.b.deg, lonlat=True, nest=False)
        return self.map[pix] > 0.5

    def own_cut(self, cat: dict) -> np.ndarray:
        return np.asarray(cat["EXT_SNR"], np.float64) >= self.snr_min

    def provenance(self) -> dict:
        return {**super().provenance(), "SELFN": str(self.mask_path), "SNRMIN": self.snr_min}
