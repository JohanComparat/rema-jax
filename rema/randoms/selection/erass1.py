"""eRASS1 clusters (Bulbul et al. 2024): an approximate detection model.

The eRASS1 selection function (Clerc et al. 2024) is a Gaussian-process classifier trained on
simulations; it is not public. This model is built from public inputs and the catalogue itself,
and every product made with it is flagged APPROX:

- exposure: T(x) = a T_ref(x), with T_ref an exposure map (e.g. the eRASS:4 one) and ``a`` the
  median of EXP / T_ref at the clusters (EXP is the eRASS1 exposure in the catalogue);
- absorption: the count rate in 0.2-2.3 keV of a cluster scales as A(nH) = exp(-nH / n_abs);
  n_abs is fitted on CR500 / F500 against the HI4PI column density nH at the clusters when
  that ratio falls with nH, else it is ``n_abs_default`` (1.7e21 cm^-2, a few-keV cluster
  spectrum through the eROSITA response; in the eRASS1 primary catalogue CR500 / F500 does not
  depend on nH, so the default is used);
- detection: the extent likelihood follows log EXT_LIKE = c0 + c1 log C + c2 log theta500 + e,
  e ~ N(0, s), with C = CR500 T the expected counts; the coefficients come from a truncated
  (Tobit) fit of the clusters with EXT_LIKE >= 3, the threshold of the catalogue.

A cluster with count rate CR_i (at its own nH_i) moved to x_j has C_j = CR_i A(nH_j) / A(nH_i)
T(x_j) and is detected with probability

    P = Phi((c0 + c1 log C_j + c2 log theta500_i - log L_cut) / s),

with L_cut the sample's EXT_LIKE cut (6 for the cosmology sample). The footprint is the
eROSITA-DE half of the sky (179.944 < l < 359.944 deg) where T > 0.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtr

from .base import ExternalSelection, register

L_RANGE = (179.944, 359.944)


def _galactic(ra, dec):
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    g = SkyCoord(np.asarray(ra, np.float64) * u.deg, np.asarray(dec, np.float64) * u.deg).galactic
    return g.l.deg, g.b.deg


class HealpixColumn:
    """Values of a HEALPix map (any ordering and frame) at sky positions."""

    def __init__(self, path=None, column=None, values=None, nside=None, nest=False, frame="C"):
        import healpy as hp
        from astropy.io import fits

        if values is None:
            with fits.open(path, memmap=True) as h:
                hdr = h[1].header
                col = column or h[1].columns.names[0]
                values = np.asarray(h[1].data[col], np.float64).ravel()
                nest = str(hdr.get("ORDERING", "RING")).upper().startswith("NEST")
                frame = "G" if str(hdr.get("COORDSYS", "C")).upper().startswith("G") else "C"
        self.values = np.asarray(values, np.float64)
        self.nside = nside or hp.npix2nside(self.values.size)
        self.nest, self.frame = nest, frame

    def __call__(self, ra, dec) -> np.ndarray:
        import healpy as hp

        if self.frame == "G":
            lon, lat = _galactic(ra, dec)
        else:
            lon, lat = np.asarray(ra, np.float64), np.asarray(dec, np.float64)
        return self.values[hp.ang2pix(self.nside, lon, lat, lonlat=True, nest=self.nest)]


def theta500_arcmin(r500_kpc, z, cosmo) -> np.ndarray:
    """Angular R500 [arcmin] of R500 [kpc] at redshift z (astropy cosmology)."""
    da = cosmo.angular_diameter_distance(np.asarray(z, np.float64)).to("kpc").value
    return np.degrees(np.asarray(r500_kpc, np.float64) / da) * 60.0


N_ABS_DEFAULT = 1.7e21


def fit_detection_model(cr, f500, extlike, exp, texp_ref, nh, theta, ext_like_min: float = 3.0,
                        n_abs_default: float = N_ABS_DEFAULT) -> dict:
    """Coefficients of the approximate model from the catalogue (arrays per cluster): count
    rate CR500 [cts/s], flux F500, EXT_LIKE, catalogue exposure EXP [s], reference exposure at
    the cluster, nH [cm^-2] and theta500 [arcmin]."""
    from scipy.optimize import minimize

    cr, f500, extlike, exp, texp_ref, nh, theta = (np.asarray(a, np.float64) for a in
                                                   (cr, f500, extlike, exp, texp_ref, nh, theta))
    ok = (cr > 0) & (f500 > 0) & (extlike >= ext_like_min) & (exp > 0) & (texp_ref > 0) & (nh > 0) & (theta > 0)
    cr, f500, extlike, exp, texp_ref, nh, theta = (a[ok] for a in (cr, f500, extlike, exp, texp_ref, nh, theta))
    a = float(np.median(exp / texp_ref))
    # Absorption: ln(CR / F) = k - nH / n_abs
    A = np.vstack([np.ones_like(nh), -nh / 1e21]).T
    coef, *_ = np.linalg.lstsq(A, np.log(cr / f500), rcond=None)
    fitted = bool(coef[1] > 0.05)                  # at least 5% fainter per 1e21 cm^-2
    n_abs = 1e21 / coef[1] if fitted else float(n_abs_default)
    # Tobit on log EXT_LIKE, truncated at log(ext_like_min)
    x = np.vstack([np.ones_like(cr), np.log(cr * exp), np.log(theta)]).T
    y = np.log(extlike)
    t = np.log(ext_like_min)

    def nll(p):
        b, ls = p[:3], p[3]
        s = np.exp(ls)
        mu = x @ b
        return -np.sum(-0.5 * ((y - mu) / s) ** 2 - ls - log_ndtr((mu - t) / s))

    b0, *_ = np.linalg.lstsq(x, y, rcond=None)
    s0 = np.log(np.std(y - x @ b0))
    res = minimize(nll, np.r_[b0, s0], method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6,
                                                                       "fatol": 1e-8})
    b, s = res.x[:3], float(np.exp(res.x[3]))
    return {"exp_scale": a, "n_abs": float(n_abs), "n_abs_fitted": fitted, "dlncr_dnh21": float(coef[1]), "c0": float(b[0]), "c1": float(b[1]), "c2": float(b[2]),
            "sigma": s, "n_fit": int(ok.sum()), "ext_like_min": float(ext_like_min), "converged": bool(res.success)}


@register("erass1_approx")
class ERASS1ApproxSelection(ExternalSelection):
    approximate = True
    columns = ("EXT_CR500", "EXT_EXT_LIKE", "EXT_THETA500", "EXT_NH")

    def __init__(self, model: dict | str | None = None, expmap: str | None = None, nhmap: str | None = None,
                 ext_like_cut: float = 6.0, exp_column: str | None = None, nh_column: str = "NHI",
                 texp=None, nh=None):
        import yaml

        self.model_path = model if isinstance(model, str) else None
        self.m = yaml.safe_load(open(model)) if isinstance(model, str) else dict(model)
        self.texp = texp if texp is not None else HealpixColumn(expmap, exp_column)
        self.nh = nh if nh is not None else HealpixColumn(nhmap, nh_column)
        self.ext_like_cut = float(ext_like_cut)
        self.expmap, self.nhmap = expmap, nhmap

    def exposure(self, ra, dec) -> np.ndarray:
        return self.m["exp_scale"] * self.texp(ra, dec)

    def in_footprint(self, ra, dec) -> np.ndarray:
        lon, _ = _galactic(ra, dec)
        return (lon > L_RANGE[0]) & (lon < L_RANGE[1]) & (self.exposure(ra, dec) > 0)

    def own_cut(self, cat: dict) -> np.ndarray:
        return np.asarray(cat["EXT_EXT_LIKE"], np.float64) >= self.ext_like_cut

    def p_detect(self, src: dict, ra, dec) -> np.ndarray:
        m = self.m
        nh_j = self.nh(ra, dec)
        cr = np.asarray(src["EXT_CR500"], np.float64) * np.exp(-(nh_j - np.asarray(src["EXT_NH"], np.float64))
                                                                / m["n_abs"])
        c = cr * self.exposure(ra, dec)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = m["c0"] + m["c1"] * np.log(c) + m["c2"] * np.log(np.asarray(src["EXT_THETA500"], np.float64))
            p = ndtr((mu - np.log(self.ext_like_cut)) / m["sigma"])
        return np.where(np.isfinite(p) & (c > 0), p, 0.0)

    def provenance(self) -> dict:
        return {**super().provenance(), "SELFN": str(self.model_path), "EXPMAP": str(self.expmap),
                "NHMAP": str(self.nhmap), "EXTLCUT": self.ext_like_cut}
