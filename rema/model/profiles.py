"""Radial, luminosity and colour filters of the matched filter.

All functions are pure ``jax.numpy`` and differentiable.

Radial filter: projected NFW profile with scale radius r_s = 0.15 h^-1 Mpc, constant inside a
core r_c = 0.1 h^-1 Mpc (redMaPPer). With x = r/r_s and u = 1 - x^2,

    F(x) = (1 - h(x)) / (x^2 - 1),   h(x) = artanh(sqrt(u))/sqrt(u)  (x < 1)
                                       h(x) = arctan(sqrt(-u))/sqrt(-u) (x > 1),

and the enclosed integral of 2 pi r Sigma is closed form: 2 pi r_s^2 g(x), g = ln(x/2) + h.
Near x = 1 both are evaluated from their power series in u, F = sum u^k/(2k+3) and
h = sum u^k/(2k+1), so they are smooth and finite there.

Luminosity filter: Schechter function in magnitudes,
phi(m) = 10^{0.4 (alpha+1)(m* - m)} exp(-10^{0.4 (m* - m)}).

Colour filter: chi^2 probability density with k degrees of freedom (number of colours).

>>> import numpy as np
>>> float(nfw_sigma(np.float32(0.15)))   # x = 1, F = 1/3
0.333...
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf, gammaln

LN10 = np.log(10.0)
TWO_POINT_FIVE_OVER_LN10 = 2.5 / LN10   # = 1.0857..., d(mag)/d(ln flux)

# --------------------------------------------------------------------------- NFW

_SERIES_U = 0.1     # |1 - x^2| below which the power series is used
_NTERMS = 10


def _series(u, first):
    """sum_k u^k / (2k + first)."""
    out = jnp.zeros_like(u)
    for k in range(_NTERMS - 1, -1, -1):
        out = out * u + 1.0 / (2 * k + first)
    return out


def nfw_h(x):
    """h(x) = arccosh(1/x)/sqrt(1-x^2) (x<1), arccos(1/x)/sqrt(x^2-1) (x>1); h(1) = 1."""
    x = jnp.asarray(x)
    u = 1.0 - x * x
    near = jnp.abs(u) < _SERIES_U
    # Safe arguments so that the unused branch is finite (keeps gradients NaN-free).
    u_lt = jnp.where((u > 0) & ~near, u, 0.5)
    u_gt = jnp.where((u < 0) & ~near, -u, 0.5)
    s, t = jnp.sqrt(u_lt), jnp.sqrt(u_gt)
    h_lt = jnp.arctanh(s) / s
    h_gt = jnp.arctan(t) / t
    h_near = _series(jnp.where(near, u, 0.0), 1)
    return jnp.where(near, h_near, jnp.where(u > 0, h_lt, h_gt))


def nfw_F(x):
    """Projected NFW shape F(x) = (1 - h)/(x^2 - 1); F(1) = 1/3."""
    x = jnp.asarray(x)
    u = 1.0 - x * x
    near = jnp.abs(u) < _SERIES_U
    den = jnp.where(near, 1.0, -u)
    far = (1.0 - nfw_h(x)) / den
    return jnp.where(near, _series(jnp.where(near, u, 0.0), 3), far)


def nfw_g(x):
    """g(x) = ln(x/2) + h(x); the enclosed projected mass is proportional to g."""
    return jnp.log(x / 2.0) + nfw_h(x)


def nfw_sigma(r, rs: float = 0.15, rcore: float = 0.1):
    """Radial filter Sigma(r) (unnormalised), flat inside ``rcore``; r in h^-1 Mpc."""
    r = jnp.asarray(r)
    return nfw_F(jnp.maximum(r, rcore) / rs)


def nfw_enclosed(r, rs: float = 0.15, rcore: float = 0.1):
    """int_0^r 2 pi r' Sigma(r') dr' for :func:`nfw_sigma` (closed form)."""
    r = jnp.asarray(r)
    xc = rcore / rs
    fc = nfw_F(jnp.asarray(xc))
    core = jnp.pi * jnp.minimum(r, rcore) ** 2 * fc
    outer = 2.0 * jnp.pi * rs**2 * (nfw_g(jnp.maximum(r, rcore) / rs) - nfw_g(jnp.asarray(xc)))
    return core + outer


def nfw_norm(r_lambda, rs: float = 0.15, rcore: float = 0.1):
    """Normalisation N(r_lambda) = 1 / int_0^{r_lambda} 2 pi r Sigma dr of the radial filter."""
    return 1.0 / nfw_enclosed(r_lambda, rs, rcore)


def theta_r(r, r_lambda, rsig: float = 0.05):
    """Soft radial cut 1/2 [1 + erf((r_lambda - r)/(sqrt(2) rsig))]."""
    return 0.5 * (1.0 + erf((r_lambda - r) / (np.sqrt(2.0) * rsig)))


# --------------------------------------------------------------------------- luminosity

def schechter(m, mstar, alpha: float = -1.0):
    """Schechter function in magnitudes (unnormalised)."""
    x = 10.0 ** (0.4 * (mstar - m))
    return x ** (alpha + 1.0) * jnp.exp(-x)


@lru_cache(maxsize=8)
def _gl(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def lumnorm(mstar, maxmag, alpha: float = -1.0, n: int = 64, bright: float = 8.0):
    """int_{-inf}^{maxmag} phi(m) dm by Gauss-Legendre on [m* - bright, maxmag].

    For alpha = -1 this equals 2.5/ln(10) * E1(10^{0.4 (m* - maxmag)}).
    """
    x, w = _gl(n)
    mstar = jnp.asarray(mstar)
    lo = mstar - bright
    hi = jnp.asarray(maxmag)
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    m = mid[..., None] + half[..., None] * x
    return half * jnp.sum(w * schechter(m, mstar[..., None], alpha), axis=-1)


def theta_i(m, maxmag, sigma_m):
    """Soft luminosity cut 1/2 [1 + erf((maxmag - m)/(sqrt(2) sigma_m))]."""
    return 0.5 * (1.0 + erf((maxmag - m) / (np.sqrt(2.0) * jnp.maximum(sigma_m, 1e-4))))


def maxmag_from_mstar(mstar, lval: float = 0.2):
    """Faint limit of the richness, m*(z) - 2.5 log10(L_min/L*)."""
    return mstar - 2.5 * np.log10(lval)


class MStar:
    """m*(z) from a table (Z, MSTAR), natural-cubic-spline interpolated onto a fine grid.

    ``name`` is a packaged table (``des_z03``, ``legacy_z_ezgal``) or a path to a FITS table.

    >>> ms = MStar("des_z03")
    >>> round(float(ms(0.5)), 2)
    19.65
    """

    def __init__(self, name: str = "des_z03", dz: float = 1e-3):
        from astropy.io import fits
        from scipy.interpolate import CubicSpline

        path = _table_path(name, "mstar")
        with fits.open(path) as h:
            z = np.asarray(h[1].data["Z"], dtype=np.float64)
            m = np.asarray(h[1].data["MSTAR"], dtype=np.float64)
        order = np.argsort(z)
        z, m = z[order], m[order]
        self.name = name
        self.zmin, self.zmax = float(z[0]), float(z[-1])
        self.z = np.arange(self.zmin, self.zmax + dz / 2, dz)
        self.m = CubicSpline(z, m, bc_type="natural")(self.z)

    def __call__(self, z):
        return jnp.interp(z, self.z, self.m)


def _table_path(name: str, kind: str) -> str:
    """Packaged table ``<kind>_<name>.fits`` or an explicit path."""
    if name.endswith((".fits", ".fit")):
        return name
    return str(resources.files("rema.data").joinpath(f"{kind}_{name}.fits"))


# --------------------------------------------------------------------------- colour

def chisq_pdf(chisq, k: int):
    """chi^2 probability density with k degrees of freedom."""
    chisq = jnp.maximum(jnp.asarray(chisq), 1e-30)
    half = 0.5 * k
    return jnp.exp((half - 1.0) * jnp.log(chisq) - 0.5 * chisq - half * np.log(2.0)
                   - gammaln(half))
