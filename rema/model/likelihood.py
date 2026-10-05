"""Photometric likelihood of a galaxy under the red-sequence model.

Two chi^2 definitions, both with ncol = nband - 1 degrees of freedom:

``chisq_lupt`` (default)
    asinh magnitudes ("luptitudes", Lupton et al. 1999) of the non-reference bands are compared
    with those of the model, each band softened at its own 1-sigma flux error b = sigma_f:

        mu(f) = zp - 2.5/ln10 [asinh(f / 2b) + ln b].

    mu equals the magnitude at high S/N, where the intrinsic colour scatter dominates and is
    Gaussian in magnitudes, and is linear in flux at low S/N, where the photometric noise
    dominates and fluxes can be negative. With model fluxes F_b = 10^{-0.4 (m_b - zp)},
    m_b = m_ref + (T c)_b, J_b = F_b / sqrt(F_b^2 + 4 b^2) = d mu_b / d m_b, the residual
    r_b = mu_b(f_b) - mu_b(F_b) has covariance

        S = diag((2.5/ln10)^2 (sigma_b^2 + (eps F_b)^2) / (F_b^2 + 4 b^2))   (noise + floor)
          + J (T C_int T^T) J                                                 (intrinsic)
          + g g^T sigma_mref^2,  g = J o (1 + T s)                            (reference noise)

    where T maps colours to band offsets and s are the slopes.

``chisq_mag`` (redMaPPer)
    Colours from magnitudes, C = C_int + R diag(sigma_m^2) R^T + s s^T sigma_ref^2; needs
    positive fluxes in every band (others get a large chi^2).

Both return (chi2, logdet): logdet is that of the covariance in magnitude units, comparable
across redshifts for a given galaxy.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax

HI = lax.Precision.HIGHEST


def _mm(a, b):
    """Small matrix product as broadcast-multiply-and-sum (no dot op).

    The matrices here are at most 4x4. Written elementwise they fuse with the surrounding code,
    stay exact in float32 (GPU dots default to TF32, ~1e-3 relative error, which moved chi^2 and
    memberships), and avoid the per-shape GEMM autotuning that made GPU compiles very slow.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if b.ndim == 1:
        return jnp.sum(a * b, axis=-1)
    return jnp.sum(a[..., :, :, None] * b[..., None, :, :], axis=-2)

from .redsequence import RSAt, band_offsets_matrix

MAG_PER_LN = 2.5 / np.log(10.0)
BAD_CHISQ = 1e6


def small_cholesky_solve(S, r):
    """chi^2 = r^T S^{-1} r and ln det S for small SPD matrices ``S[..., n, n]`` (unrolled)."""
    n = S.shape[-1]
    L = [[None] * n for _ in range(n)]
    for j in range(n):
        acc = S[..., j, j]
        for k in range(j):
            acc = acc - L[j][k] ** 2
        L[j][j] = jnp.sqrt(jnp.maximum(acc, 1e-30))
        for i in range(j + 1, n):
            acc = S[..., i, j]
            for k in range(j):
                acc = acc - L[i][k] * L[j][k]
            L[i][j] = acc / L[j][j]
    y = []
    for i in range(n):
        acc = r[..., i]
        for k in range(i):
            acc = acc - L[i][k] * y[k]
        y.append(acc / L[i][i])
    chi2 = sum(yi * yi for yi in y)
    logdet = 2.0 * sum(jnp.log(L[i][i]) for i in range(n))
    return chi2, logdet


def chisq_lupt(flux, ivar, rs: RSAt, iref: int, eps: float = 0.015, zeropoint: float = 22.5):
    """asinh-magnitude chi^2 (see module docstring). ``flux``, ``ivar``: [..., nband]."""
    flux = jnp.asarray(flux)
    ivar = jnp.maximum(jnp.asarray(ivar), 1e-20)
    nband = flux.shape[-1]
    T, others = band_offsets_matrix(nband, iref)
    T = jnp.asarray(T, dtype=flux.dtype)
    sig = 1.0 / jnp.sqrt(ivar)
    f_ref = jnp.maximum(flux[..., iref], 1e-10)
    m_ref = zeropoint - 2.5 * jnp.log10(f_ref)
    col = rs.colours(m_ref)                                   # [..., ncol]
    m_mod = m_ref[..., None] + _mm(col, T.T)                  # [..., nb-1]
    F = 10.0 ** (-0.4 * (m_mod - zeropoint))
    b = sig[..., others]
    fo = flux[..., others]
    # mu(f) - mu(F) with the same softening: the ln b terms cancel.
    r = -MAG_PER_LN * (jnp.arcsinh(fo / (2.0 * b)) - jnp.arcsinh(F / (2.0 * b)))
    den = F * F + 4.0 * b * b
    J = F / jnp.sqrt(den)
    var_b = MAG_PER_LN**2 * (b * b + (eps * F) ** 2) / den
    var_ref = MAG_PER_LN**2 * (sig[..., iref] ** 2 / (f_ref * f_ref) + eps**2)
    g = J * (1.0 + _mm(rs.slope, T.T))
    tct = _mm(_mm(T, rs.cint), T.T)                           # [..., nb-1, nb-1]
    S = J[..., :, None] * tct * J[..., None, :]
    S = S + g[..., :, None] * g[..., None, :] * var_ref[..., None, None]
    S = S + jnp.eye(nband - 1, dtype=flux.dtype) * var_b[..., None, :]
    return small_cholesky_solve(S, -r)


def chisq_mag(flux, ivar, rs: RSAt, iref: int, eps: float = 0.015, zeropoint: float = 22.5):
    """redMaPPer colour-space chi^2 (needs positive fluxes; others get BAD_CHISQ)."""
    flux = jnp.asarray(flux)
    ivar = jnp.maximum(jnp.asarray(ivar), 1e-20)
    nband = flux.shape[-1]
    good = jnp.all(flux > 0, axis=-1)
    fpos = jnp.where(flux > 0, flux, 1.0)
    mags = zeropoint - 2.5 * jnp.log10(fpos)
    merr2 = (2.5 / np.log(10.0)) ** 2 / (fpos * fpos * ivar) + (2.5 / np.log(10.0) * eps) ** 2
    cobs = mags[..., :-1] - mags[..., 1:]
    cmod = rs.colours(mags[..., iref])
    d = cobs - cmod
    ncol = nband - 1
    R = np.zeros((ncol, nband))
    for j in range(ncol):
        R[j, j], R[j, j + 1] = 1.0, -1.0
    R = jnp.asarray(R, dtype=flux.dtype)
    cobs_cov = _mm(R * merr2[..., None, :], R.T)
    S = rs.cint + cobs_cov + rs.slope[..., :, None] * rs.slope[..., None, :] * merr2[..., iref, None, None]
    chi2, logdet = small_cholesky_solve(S, d)
    return jnp.where(good, chi2, BAD_CHISQ), jnp.where(good, logdet, 0.0)


def chisq(flux, ivar, rs: RSAt, iref: int, mode: str = "lupt", eps: float = 0.015):
    if mode == "lupt":
        return chisq_lupt(flux, ivar, rs, iref, eps)
    if mode == "mag":
        return chisq_mag(flux, ivar, rs, iref, eps)
    raise ValueError(f"unknown chisq mode {mode!r}")
