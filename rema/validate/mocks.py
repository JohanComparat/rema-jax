"""Synthetic galaxies and clusters for testing richness and redshift recovery.

A mock cluster of richness lambda at redshift z has, on average, lambda members brighter than
L_min = 0.2 L* inside r_lambda = r0 (lambda/100)^beta, distributed with the projected NFW profile
(with core) of the filter, Schechter magnitudes (alpha = -1) and red-sequence colours with the
model's intrinsic scatter. Members fainter than 0.2 L* down to ``dmag_faint`` below m* are added
in Schechter proportion, and photometric noise follows given 5-sigma depths. Fluxes are in the
galaxy-table convention (dereddened nanomaggies, bands blue to red).

Every galaxy carries its true redshift (ZTRUE). With a photo-z model (:class:`GaussianPhotoz`)
the tables also get ZPHOT and ZPHOT_STD, as the DR11 photo-z sweeps; clusters can have a fraction
of blue members (``blue_fraction``), with the colours of the blue field galaxies.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ..model import profiles as prof
from ..model.redsequence import RSModel, band_offsets_matrix


def _draw_radii(rng, n, r_lambda, rs=0.15, rcore=0.1):
    """Inverse-CDF draw of projected radii from 2 pi r Sigma_NFW(r) on [0, r_lambda]."""
    rr = np.linspace(0.0, r_lambda, 2001)
    cdf = np.asarray(prof.nfw_enclosed(rr, rs, rcore), np.float64)
    cdf /= cdf[-1]
    return np.interp(rng.uniform(size=n), cdf, rr)


def _draw_mags(rng, n, mstar, mfaint, alpha=-1.0, mbright_off=6.0):
    m = np.linspace(mstar - mbright_off, mfaint, 4001)
    pdf = np.asarray(prof.schechter(m, mstar, alpha), np.float64)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return np.interp(rng.uniform(size=n), cdf, m)


def noisy_fluxes(rng, mags: np.ndarray, depth5: np.ndarray, zeropoint: float = 22.5):
    """Fluxes and inverse variances for true magnitudes [N, nb] and 5-sigma depths [nb]."""
    f = 10.0 ** (-0.4 * (mags - zeropoint))
    sig = np.broadcast_to(10.0 ** (-0.4 * (np.asarray(depth5) - zeropoint)) / 5.0, f.shape)
    return (f + rng.normal(size=f.shape) * sig).astype(np.float32), (1.0 / sig**2).astype(np.float32)


def red_sequence_mags(rng, rs: RSModel, z, refmag: np.ndarray, scatter: bool = True):
    """True magnitudes [N, nb] of red-sequence galaxies at z (scalar or [N]) with magnitudes refmag."""
    import jax.numpy as jnp

    at = rs.at(jnp.broadcast_to(jnp.asarray(z, jnp.float32), (refmag.size,)))
    col = np.asarray(at.colours(jnp.asarray(refmag)), np.float64)
    if scatter:
        L = np.linalg.cholesky(np.asarray(at.cint, np.float64))
        col = col + np.einsum("nij,nj->ni", L, rng.normal(size=col.shape))
    T, others = band_offsets_matrix(rs.nband, rs.iref)
    mags = np.zeros((refmag.size, rs.nband))
    mags[:, rs.iref] = refmag
    mags[:, others] = refmag[:, None] + col @ T.T
    return mags


class GaussianPhotoz:
    """Photo-z of mock galaxies: ZPHOT = z + sigma N(0, 1), and outliers.

    sigma = sigma0 (1 + z) 10^(slope (m - m0)), clipped to [sigma_min, sigma_max] (1 + z): the
    scatter grows with the reference magnitude m as in DR11. A fraction ``outlier_frac`` of the
    galaxies get a redshift uniform in ``outlier_range``. ZPHOT_STD = ``std_factor`` sigma (the
    DR11 values overestimate the scatter).
    """

    def __init__(self, sigma0: float = 0.02, slope: float = 0.25, m0: float = 20.5,
                 sigma_min: float = 0.005, sigma_max: float = 0.3, std_factor: float = 1.0,
                 outlier_frac: float = 0.0, outlier_range=(0.0, 1.5)):
        self.sigma0, self.slope, self.m0 = sigma0, slope, m0
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.std_factor, self.outlier_frac, self.outlier_range = std_factor, outlier_frac, outlier_range

    def sigma(self, z, refmag):
        frac = np.clip(self.sigma0 * 10.0 ** (self.slope * (np.asarray(refmag) - self.m0)),
                       self.sigma_min, self.sigma_max)
        return frac * (1.0 + np.asarray(z))

    def __call__(self, rng, z, refmag) -> dict:
        z = np.asarray(z, np.float64)
        s = self.sigma(z, refmag)
        zp = z + s * rng.normal(size=z.size)
        if self.outlier_frac > 0:
            out = rng.uniform(size=z.size) < self.outlier_frac
            zp[out] = rng.uniform(*self.outlier_range, out.sum())
        return {"ZPHOT": np.maximum(zp, 0.0).astype(np.float32),
                "ZPHOT_STD": (self.std_factor * s).astype(np.float32)}


def _bluer(rs: RSModel, mags, off):
    """Shift every non-reference band towards the reference band by ``off`` (mag) [N]."""
    T, others = band_offsets_matrix(rs.nband, rs.iref)
    w = np.abs(T).sum(axis=1)
    mags[:, others] -= off[:, None] * w[None, :] / w.max()
    return mags


def mock_cluster(rng, rs: RSModel, mstar_fn, mpc_per_deg_fn, ra0: float, dec0: float, z: float,
                 lam: float, depth5, r0: float = 1.0, beta: float = 0.2, lval: float = 0.2,
                 dmag_faint: float = 2.5, poisson: bool = True, central_dmag: float | None = None,
                 extent: float = 1.0, blue_fraction: float = 0.0, photoz: GaussianPhotoz | None = None):
    """Members of a mock cluster: dict with RA, DEC, FLUX, FLUX_IVAR, REFMAG, REFMAG_ERR, TRUE_MAG,
    SNR, IS_CENTRAL, ZTRUE and IS_BLUE (and ZPHOT, ZPHOT_STD with ``photoz``).

    ``central_dmag``: if given, a central galaxy of magnitude m*(z) + central_dmag is added at
    (ra0, dec0) (the first row), on the red sequence. ``extent``: members follow the NFW profile
    out to ``extent`` r_lambda (lambda of them, on average, inside r_lambda); with the default 1
    the cluster stops at r_lambda, so a larger aperture finds no more members.
    ``blue_fraction``: that fraction of the members (not the central) has blue colours, offset
    by 0.2-1 mag as the blue field galaxies of :func:`mock_field`.
    """
    mstar = float(mstar_fn(z))
    maxmag = mstar - 2.5 * np.log10(lval)
    mfaint = mstar + dmag_faint
    frac = float(prof.lumnorm(mstar, mfaint) / prof.lumnorm(mstar, maxmag))
    rl = r0 * (lam / 100.0) ** beta
    if extent > 1.0:
        frac *= float(prof.nfw_enclosed(extent * rl) / prof.nfw_enclosed(rl))
    n = rng.poisson(lam * frac) if poisson else int(round(lam * frac))
    r = _draw_radii(rng, n, max(extent, 1.0) * rl)
    ang = rng.uniform(0, 2 * np.pi, n)
    D = float(mpc_per_deg_fn(z))
    dra = r / D * np.cos(ang) / np.cos(np.radians(dec0))
    ddec = r / D * np.sin(ang)
    refmag = _draw_mags(rng, n, mstar, mfaint)
    central = np.zeros(n, bool)
    if central_dmag is not None:
        dra, ddec = np.concatenate([[0.0], dra]), np.concatenate([[0.0], ddec])
        refmag = np.concatenate([[mstar + central_dmag], refmag])
        central = np.concatenate([[True], central])
    mags = red_sequence_mags(rng, rs, z, refmag)
    blue = np.zeros(refmag.size, bool)
    if blue_fraction > 0:
        blue = (rng.uniform(size=refmag.size) < blue_fraction) & ~central
        mags = _bluer(rs, mags, np.where(blue, rng.uniform(0.2, 1.0, refmag.size), 0.0))
    flux, ivar = noisy_fluxes(rng, mags, depth5)
    fref = np.maximum(flux[:, rs.iref], 1e-6)
    snr = fref * np.sqrt(ivar[:, rs.iref])
    out = {"RA": ra0 + dra, "DEC": dec0 + ddec, "FLUX": flux, "FLUX_IVAR": ivar,
           "REFMAG": (22.5 - 2.5 * np.log10(fref)).astype(np.float32),
           "REFMAG_ERR": (2.5 / np.log(10) / np.maximum(snr, 1e-3)).astype(np.float32),
           "TRUE_MAG": mags, "SNR": snr, "IS_CENTRAL": central,
           "ZTRUE": np.full(refmag.size, z, np.float32), "IS_BLUE": blue}
    if photoz is not None:
        out.update(photoz(rng, out["ZTRUE"], out["REFMAG"]))
    return out


def mock_field(rng, rs: RSModel, box, density: float, depth5, mag_range=(12.0, 23.5),
               slope: float = 0.4, red_fraction: float = 0.15, zrange=(0.05, 1.2),
               photoz: GaussianPhotoz | None = None):
    """Uniform background galaxies: number counts N(<m) ~ 10^(slope m); a fraction on the red
    sequence at random redshifts, the rest with colours offset to the blue by 0.2-1 mag.
    Their redshifts are in ZTRUE (and photo-z in ZPHOT, ZPHOT_STD with ``photoz``)."""
    area = box.area_deg2()
    n = rng.poisson(density * area)
    ra = rng.uniform(box.ra_min, box.ra_max, n)
    s0, s1 = np.sin(np.radians(box.dec_min)), np.sin(np.radians(box.dec_max))
    dec = np.degrees(np.arcsin(rng.uniform(s0, s1, n)))
    u = rng.uniform(size=n)
    a, b = 10 ** (slope * mag_range[0]), 10 ** (slope * mag_range[1])
    refmag = np.log10(a + u * (b - a)) / slope
    zz = rng.uniform(*zrange, n)
    red = rng.uniform(size=n) < red_fraction
    mags = red_sequence_mags(rng, rs, zz, refmag)
    off = np.zeros(n)
    off[~red] = rng.uniform(0.2, 1.0, (~red).sum())
    mags = _bluer(rs, mags, off)
    flux, ivar = noisy_fluxes(rng, mags, depth5)
    fref = np.maximum(flux[:, rs.iref], 1e-6)
    snr = fref * np.sqrt(ivar[:, rs.iref])
    keep = snr >= 5
    out = {"RA": ra[keep], "DEC": dec[keep], "FLUX": flux[keep], "FLUX_IVAR": ivar[keep],
           "REFMAG": (22.5 - 2.5 * np.log10(fref[keep])).astype(np.float32),
           "REFMAG_ERR": (2.5 / np.log(10) / snr[keep]).astype(np.float32),
           "ZTRUE": zz[keep].astype(np.float32)}
    if photoz is not None:
        out.update(photoz(rng, out["ZTRUE"], out["REFMAG"]))
    return out


def concat(*tables):
    keys = set.intersection(*(set(t) for t in tables))
    return {k: np.concatenate([t[k] for t in tables]) for k in keys}


def solve_lambda_numpy(u, b, w, r, r0=1.0, beta=0.2, rsig=0.05, K=1.0):
    """Reference lambda (brentq) for tests: sum p w theta_r = lambda K."""
    from scipy.special import erf

    def g(lam):
        rl = r0 * (lam / 100.0) ** beta
        N = 1.0 / float(prof.nfw_enclosed(rl))
        p = lam * N * u / (lam * N * u + b)
        thr = 0.5 * (1 + erf((rl - r) / (np.sqrt(2) * rsig)))
        return np.sum(p * w * thr) - lam * K

    return brentq(g, 0.5, 2000.0, xtol=1e-8)


def mock_template_cluster(rng, rs: RSModel, tpl, ra0: float, dec0: float, z: float, A: float, depth5,
                          mag_max: float, photoz: GaussianPhotoz, blue_fraction: float = 0.5,
                          poisson: bool = True):
    """A cluster drawn from the PSCD template (:class:`rema.pscd.model.Template`) with amplitude A:
    A times the template's galaxies within its truncation radius and brighter than ``mag_max``
    (true magnitudes), with photo-z and a fraction of blue members. Same columns as
    :func:`mock_cluster`."""
    zz = float(z)
    rmax = float(tpl.rmax_deg(zz))
    th = np.linspace(0.0, rmax, 4001)
    cdf_r = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(th) * (th[1:] * tpl.psi(th[1:], zz)
                                                                  + th[:-1] * tpl.psi(th[:-1], zz)))])
    nr = 2.0 * np.pi * cdf_r[-1]                        # int Psi d^2theta within the truncation
    ms = float(tpl.mstar(zz))
    m = np.linspace(ms - 6.0, mag_max, 4001)
    pm = tpl.phi(m, zz)
    cdf_m = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(m) * (pm[1:] + pm[:-1]))])
    mean = A * nr * cdf_m[-1]
    n = rng.poisson(mean) if poisson else int(round(mean))
    r = np.interp(rng.uniform(size=n), cdf_r / cdf_r[-1], th)
    ang = rng.uniform(0, 2 * np.pi, n)
    dra = r * np.cos(ang) / np.cos(np.radians(dec0))
    ddec = r * np.sin(ang)
    refmag = np.interp(rng.uniform(size=n), cdf_m / cdf_m[-1], m)
    mags = red_sequence_mags(rng, rs, zz, refmag)
    blue = rng.uniform(size=n) < blue_fraction
    mags = _bluer(rs, mags, np.where(blue, rng.uniform(0.2, 1.0, n), 0.0))
    flux, ivar = noisy_fluxes(rng, mags, depth5)
    fref = np.maximum(flux[:, rs.iref], 1e-6)
    snr = fref * np.sqrt(ivar[:, rs.iref])
    out = {"RA": ra0 + dra, "DEC": dec0 + ddec, "FLUX": flux, "FLUX_IVAR": ivar,
           "REFMAG": (22.5 - 2.5 * np.log10(fref)).astype(np.float32),
           "REFMAG_ERR": (2.5 / np.log(10) / np.maximum(snr, 1e-3)).astype(np.float32),
           "TRUE_MAG": mags, "SNR": snr, "IS_CENTRAL": np.zeros(n, bool),
           "ZTRUE": np.full(n, zz, np.float32), "IS_BLUE": blue}
    out.update(photoz(rng, out["ZTRUE"], out["REFMAG"]))
    keep = snr >= 5
    return {k: v[keep] for k, v in out.items()}
