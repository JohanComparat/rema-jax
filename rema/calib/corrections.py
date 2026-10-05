"""Redshift corrections fitted against spectroscopic redshifts.

zred correction (redMaPPer ``corr``, ``corr_r``): for red galaxies with spectroscopic redshifts,
dz = z_spec - zred_uncorr is binned in zred_uncorr; per node the corrected offset is the
5-sigma-clipped median of dz and the error scaling is the clipped robust spread of
dz / zred_uncorr_e. The slope with magnitude is set to zero (DR10: ``calib_corr_nocorrslope``).

z_lambda correction (``ZlambdaCorrection``): for clusters with a spectroscopic redshift,
dz = z_spec - z_lambda binned in z_lambda gives the offset; the extra scatter is the part of the
robust variance of dz not explained by z_lambda_e.

z -> zred_uncorr mapping (``ZlambdaCorrection.zred_uncorr``, redMaPPer zlambdacal.py:338-343):
the median zred of the central galaxies against the raw z_lambda of their clusters, a natural
cubic spline fitted with an L1 cost. It centres the zred term of LNCGLIKE.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import scipy.optimize
from scipy.interpolate import CubicSpline

from ..calibration import ZlambdaCorrection
from ..model.redsequence import ZredCorrection


def _clipped(x, nsig: float = 5.0, niter: int = 5):
    x = np.asarray(x, np.float64)
    keep = np.isfinite(x)
    for _ in range(niter):
        if keep.sum() < 3:
            break
        med = np.median(x[keep])
        mad = 1.4826 * np.median(np.abs(x[keep] - med))
        new = np.isfinite(x) & (np.abs(x - med) <= nsig * max(mad, 1e-6))
        if np.array_equal(new, keep):
            break
        keep = new
    return keep


def _binned(x, y, nodes, halfwidth, min_n=20, stat="median"):
    out = np.full(nodes.size, np.nan)
    for k, z0 in enumerate(nodes):
        sel = np.abs(x - z0) < halfwidth
        if sel.sum() < min_n:
            continue
        ys = y[sel]
        keep = _clipped(ys)
        if keep.sum() < min_n:
            continue
        if stat == "median":
            out[k] = np.median(ys[keep])
        else:
            out[k] = 1.4826 * np.median(np.abs(ys[keep] - np.median(ys[keep])))
    return out


def _fill(nodes, vals, default):
    ok = np.isfinite(vals)
    if not np.any(ok):
        return np.full(nodes.size, default)
    return np.interp(nodes, nodes[ok], vals[ok])


def fit_zred_correction(zred_u, zred_u_e, zspec, *, zrange=(0.05, 0.9), dz_node: float = 0.05,
                        min_n: int = 20) -> ZredCorrection:
    zred_u, zred_u_e, zspec = (np.asarray(a, np.float64) for a in (zred_u, zred_u_e, zspec))
    ok = (zred_u > 0) & (zred_u_e > 0) & (zspec > 0)
    nodes = np.arange(zrange[0], zrange[1] + dz_node / 2, dz_node)
    dz = zspec - zred_u
    off = _fill(nodes, _binned(zred_u[ok], dz[ok], nodes, dz_node, min_n), 0.0)
    pull = (dz - np.interp(zred_u, nodes, off)) / zred_u_e
    r = _fill(nodes, _binned(zred_u[ok], pull[ok], nodes, dz_node, min_n, stat="spread"), 1.0)
    r = np.clip(r, 0.5, 3.0)
    return ZredCorrection(jnp.asarray(off), jnp.zeros(nodes.size), jnp.asarray(r),
                          jnp.asarray(nodes), jnp.asarray(nodes))


def fit_zlambda_correction(z_lambda, z_lambda_e, zspec, lam, *, zrange=(0.05, 0.9),
                           dz_node: float = 0.08, min_n: int = 15, pivot: float = 30.0) -> ZlambdaCorrection:
    zl, ze, zs, lam = (np.asarray(a, np.float64) for a in (z_lambda, z_lambda_e, zspec, lam))
    ok = (zl > 0) & (zs > 0) & (ze > 0)
    nodes = np.arange(zrange[0], zrange[1] + dz_node / 2, dz_node)
    dz = zs - zl
    off = _fill(nodes, _binned(zl[ok], dz[ok], nodes, dz_node, min_n), 0.0)
    resid = dz - np.interp(zl, nodes, off)
    spread = _binned(zl[ok], resid[ok], nodes, dz_node, min_n, stat="spread")
    med_e = np.full(nodes.size, np.nan)
    for k, z0 in enumerate(nodes):
        sel = ok & (np.abs(zl - z0) < dz_node)
        if sel.sum() >= min_n:
            med_e[k] = np.median(ze[sel])
    extra = np.sqrt(np.maximum(spread**2 - med_e**2, 0.0))
    scatter = _fill(nodes, extra, 0.0)
    return ZlambdaCorrection(jnp.asarray(nodes), jnp.asarray(off), jnp.zeros(nodes.size),
                             jnp.asarray(scatter), pivot=pivot)


def fit_zred_uncorr(z_lambda, zred, nodes) -> np.ndarray | None:
    """Node values of the z -> zred_uncorr mapping, or None with fewer clusters than nodes.

    redMaPPer's MedZFitter (fitters.py:14-91), as called in zlambdacal.py:338-343: the natural
    cubic spline s through ``nodes`` minimising sum |zred - s(z_lambda)| (a running median), from
    s(z) = z, with L-BFGS-B on forward differences. ``z_lambda`` is the clusters' raw z_lambda
    and ``zred`` their central galaxy's zred; failed values (<= 0) are left out.
    """
    x, y = (np.asarray(a, np.float64) for a in (z_lambda, zred))
    nodes = np.asarray(nodes, np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < nodes.size:
        return None
    # The spline is linear in its node values: s(x) = basis(x) @ values.
    basis = CubicSpline(nodes, np.eye(nodes.size), bc_type="natural")(x[ok])
    y = y[ok]
    res = scipy.optimize.minimize(lambda v: np.sum(np.abs(y - basis @ v)), nodes.copy(),
                                  method="L-BFGS-B",
                                  options=dict(maxfun=2000, maxiter=2000, maxcor=20, eps=1e-5,
                                               gtol=1e-8))
    return res.x
