"""Synthetic galaxies and clusters for testing richness and redshift recovery.

A mock cluster of richness lambda at redshift z has, on average, lambda members brighter than
L_min = 0.2 L* inside r_lambda = r0 (lambda/100)^beta, distributed with the projected NFW profile
(with core) of the filter, Schechter magnitudes (alpha = -1) and red-sequence colours with the
model's intrinsic scatter. Members fainter than 0.2 L* down to ``dmag_faint`` below m* are added
in Schechter proportion, and photometric noise follows given 5-sigma depths. Fluxes are in the
galaxy-table convention (dereddened nanomaggies, bands blue to red).
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


def mock_cluster(rng, rs: RSModel, mstar_fn, mpc_per_deg_fn, ra0: float, dec0: float, z: float,
                 lam: float, depth5, r0: float = 1.0, beta: float = 0.2, lval: float = 0.2,
                 dmag_faint: float = 2.5, poisson: bool = True, central_dmag: float | None = None):
    """Members of a mock cluster: dict with RA, DEC, FLUX, FLUX_IVAR, REFMAG, REFMAG_ERR, TRUE_MAG,
    SNR and IS_CENTRAL.

    ``central_dmag``: if given, a central galaxy of magnitude m*(z) + central_dmag is added at
    (ra0, dec0) (the first row), on the red sequence.
    """
    mstar = float(mstar_fn(z))
    maxmag = mstar - 2.5 * np.log10(lval)
    mfaint = mstar + dmag_faint
    frac = float(prof.lumnorm(mstar, mfaint) / prof.lumnorm(mstar, maxmag))
    n = rng.poisson(lam * frac) if poisson else int(round(lam * frac))
    rl = r0 * (lam / 100.0) ** beta
    r = _draw_radii(rng, n, rl)
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
    flux, ivar = noisy_fluxes(rng, mags, depth5)
    fref = np.maximum(flux[:, rs.iref], 1e-6)
    snr = fref * np.sqrt(ivar[:, rs.iref])
    return {"RA": ra0 + dra, "DEC": dec0 + ddec, "FLUX": flux, "FLUX_IVAR": ivar,
            "REFMAG": (22.5 - 2.5 * np.log10(fref)).astype(np.float32),
            "REFMAG_ERR": (2.5 / np.log(10) / np.maximum(snr, 1e-3)).astype(np.float32),
            "TRUE_MAG": mags, "SNR": snr, "IS_CENTRAL": central}


def mock_field(rng, rs: RSModel, box, density: float, depth5, mag_range=(12.0, 23.5),
               slope: float = 0.4, red_fraction: float = 0.15, zrange=(0.05, 1.2)):
    """Uniform background galaxies: number counts N(<m) ~ 10^(slope m); a fraction on the red
    sequence at random redshifts, the rest with colours offset to the blue by 0.2-1 mag."""
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
    T, others = band_offsets_matrix(rs.nband, rs.iref)
    # bluer: shift every non-reference band towards the reference band
    mags[:, others] -= off[:, None] * np.abs(T).sum(axis=1)[None, :] / np.abs(T).sum(axis=1).max()
    flux, ivar = noisy_fluxes(rng, mags, depth5)
    fref = np.maximum(flux[:, rs.iref], 1e-6)
    snr = fref * np.sqrt(ivar[:, rs.iref])
    keep = snr >= 5
    return {"RA": ra[keep], "DEC": dec[keep], "FLUX": flux[keep], "FLUX_IVAR": ivar[keep],
            "REFMAG": (22.5 - 2.5 * np.log10(fref[keep])).astype(np.float32),
            "REFMAG_ERR": (2.5 / np.log(10) / snr[keep]).astype(np.float32)}


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
