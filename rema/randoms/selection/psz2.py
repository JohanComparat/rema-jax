"""Planck PSZ2 (Planck Collaboration 2016): the survey (or cosmology) mask, and the noise.

The released PSZ2 completeness (``HFI_PCCS_SZ-selfunc-*_R2.08.fits``) is averaged over the sky; within
the mask it does not depend on the position. Planck's noise does: the scan observed the ecliptic
poles longest. With ``model`` (``cluster-randoms psz2-model``) the noise follows the hit counts H
of an HFI map (smoothed to the scale of the matched filters), sigma ~ H^gamma, gamma = -1/2 (white
noise) by default; the regression ln(Y5R500 / SNR) = a + gamma ln H + delta ln Y5R500 on the
catalogue is recorded (gamma_fit) but diluted towards 0 by the noise of Y5R500 / SNR. A cluster of S/N SNR_i moved
from x_i to x_j then has SNR_i (H(x_i) / H(x_j))^gamma and is detected with probability

    P = Phi(SNR_i (H_i / H_j)^gamma - snr_min).

Without ``model``, the mask only (every position of the mask equally likely).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .base import ExternalSelection, register


def fit_noise_model(snr, y5r500, hits) -> dict:
    """gamma and delta of ln(Y5R500 / SNR) = a + gamma ln H + delta ln Y5R500 (least squares on
    the clusters with Y5R500 > 0, SNR > 0 and H > 0)."""
    snr, y, h = (np.asarray(a, np.float64) for a in (snr, y5r500, hits))
    ok = (snr > 0) & (y > 0) & (h > 0)
    A = np.vstack([np.ones(ok.sum()), np.log(h[ok]), np.log(y[ok])]).T
    b = np.log(y[ok] / snr[ok])
    coef, res, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ coef
    cov = np.linalg.inv(A.T @ A) * np.var(resid, ddof=3)
    return {"a": float(coef[0]), "gamma": float(coef[1]), "delta": float(coef[2]),
            "gamma_err": float(np.sqrt(cov[1, 1])), "scatter": float(np.std(resid)), "n_fit": int(ok.sum())}


@register("psz2")
class PSZ2Selection(ExternalSelection):
    columns = ("EXT_SNR",)

    def __init__(self, mask: str | None = None, snr_min: float = 4.5, mask_map=None, nside: int | None = None,
                 model: str | dict | None = None, hits: str | None = None, hits_column: str | None = None,
                 hits_map=None):
        import healpy as hp

        self.mask_path = mask
        self.map = np.asarray(mask_map if mask_map is not None else hp.read_map(mask, field=0, nest=False,
                                                                            dtype=np.float64))
        self.nside = nside or hp.npix2nside(self.map.size)
        self.snr_min = float(snr_min)
        self.model_path = model if isinstance(model, str) else None
        self.m = None
        if model is not None:
            import yaml

            from .erass1 import HealpixColumn

            self.m = yaml.safe_load(open(model)) if isinstance(model, str) else dict(model)
            self.hits = hits_map if hits_map is not None else HealpixColumn(hits, hits_column)
            self.columns = ("EXT_SNR", "EXT_HITS")

    def in_footprint(self, ra, dec) -> np.ndarray:
        import astropy.units as u
        import healpy as hp
        from astropy.coordinates import SkyCoord

        g = SkyCoord(np.asarray(ra, np.float64) * u.deg, np.asarray(dec, np.float64) * u.deg).galactic
        pix = hp.ang2pix(self.nside, g.l.deg, g.b.deg, lonlat=True, nest=False)
        return self.map[pix] > 0.5

    def own_cut(self, cat: dict) -> np.ndarray:
        return np.asarray(cat["EXT_SNR"], np.float64) >= self.snr_min

    def p_detect(self, src: dict, ra, dec) -> np.ndarray:
        if self.m is None:
            return np.ones(np.shape(ra), np.float64)
        hj = self.hits(ra, dec)
        hi = np.asarray(src["EXT_HITS"], np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.asarray(src["EXT_SNR"], np.float64) * (hi / hj) ** self.m["gamma"]
            p = ndtr(snr - self.snr_min)
        return np.where(np.isfinite(p) & (hj > 0), p, 0.0)

    def provenance(self) -> dict:
        out = {**super().provenance(), "SELFN": str(self.mask_path), "SNRMIN": self.snr_min}
        if self.m is not None:
            out.update(NOISEMOD=str(self.model_path), GAMMA=self.m["gamma"])
        return out
