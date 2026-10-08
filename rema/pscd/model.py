"""PSCD cluster template M_c(r, m) = Psi(r) Phi(m), the noise N(m, z) and the filter's
normalisation integrals.

- Psi: projected NFW (r_s = r200 / c, flat inside ``rcore``), truncated at ``rtrunc`` r200 and
  normalised to 1 inside r200; per deg^2 at the cluster redshift (r = theta D(z)).
- Phi: Schechter function around m*(z), per mag, with ``n200`` galaxies brighter than
  m* + ``dmag_n200`` (Maturi et al. 2019, Sect. 3.4).
- N: Sigma_pz(z, m), the stacked photo-z distributions per deg^2, per mag and per unit z.
- The redshift statistics q of Bellagamba et al. (2018, Eq. 8) for Gaussian photo-z: a member at
  z_c with width s has <p_i(z_c)> = 1 / (2 sqrt(pi) s) and <p_i(z_c)^2> = 1 / (2 sqrt(3) pi s^2);
  their averages over the galaxies of photo-z z and magnitude m are tabulated (:class:`WidthTable`,
  the magnitude dependence of Maturi et al. 2019, Eqs. 8-11).

For a pixel of local magnitude limit m_lim the magnitude integrals of alpha, beta and gamma are

    I1 = int^m_lim Phi dm,   I2 = int^m_lim Phi^2 <g^2> / N dm,   I3 = int^m_lim Phi^3 <g^3> / N^2 dm,

tabulated per redshift slice as cumulative integrals in m_lim (:class:`MagIntegrals`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..config import PscdConfig
from ..model.background import ZredBkg

SQRT_PI = float(np.sqrt(np.pi))


# NumPy versions of rema.model.profiles (the JAX functions compile for every new array shape).
def nfw_h(x):
    """arccosh(1/x)/sqrt(1-x^2) (x < 1), arccos(1/x)/sqrt(x^2-1) (x > 1), 1 at x = 1."""
    x = np.asarray(x, np.float64)
    u = 1.0 - x * x
    out = np.ones_like(u)
    near = np.abs(u) < 1e-3
    lt, gt = (u > 0) & ~near, (u < 0) & ~near
    s, t = np.sqrt(u[lt]), np.sqrt(-u[gt])
    out[lt] = np.arctanh(s) / s
    out[gt] = np.arctan(t) / t
    out[near] = 1.0 + u[near] / 3.0 + u[near] ** 2 / 5.0
    return out


def nfw_F(x):
    """Projected NFW shape (1 - h)/(x^2 - 1), 1/3 at x = 1."""
    x = np.asarray(x, np.float64)
    u = 1.0 - x * x
    near = np.abs(u) < 1e-3
    out = np.empty_like(u)
    out[~near] = (1.0 - nfw_h(x[~near])) / (-u[~near])
    out[near] = 1.0 / 3.0 + u[near] / 5.0 + u[near] ** 2 / 7.0
    return out


def nfw_sigma(r, rs: float, rcore: float):
    """Projected NFW with a flat core inside ``rcore`` (unnormalised); r in h^-1 Mpc."""
    return nfw_F(np.maximum(np.asarray(r, np.float64), rcore) / rs)


def nfw_enclosed(r, rs: float, rcore: float):
    """int_0^r 2 pi r' nfw_sigma(r') dr'."""
    g = lambda x: np.log(x / 2.0) + nfw_h(x)
    r = np.asarray(r, np.float64)
    xc = np.asarray(rcore / rs)
    core = np.pi * np.minimum(r, rcore) ** 2 * nfw_F(xc)
    return core + 2.0 * np.pi * rs**2 * (g(np.maximum(r, rcore) / rs) - g(xc))


def schechter(m, mstar, alpha: float):
    """Schechter function in magnitudes (unnormalised)."""
    x = 10.0 ** (0.4 * (np.asarray(mstar, np.float64) - np.asarray(m, np.float64)))
    return x ** (alpha + 1.0) * np.exp(-x)


@lru_cache(maxsize=4)
def _gl(n: int):
    return np.polynomial.legendre.leggauss(n)


def lumnorm(mstar: float, mlim: float, alpha: float, n: int = 64, bright: float = 8.0) -> float:
    """int_{-inf}^{mlim} schechter dm (Gauss-Legendre on [m* - bright, mlim])."""
    x, w = _gl(n)
    lo = mstar - bright
    half, mid = 0.5 * (mlim - lo), 0.5 * (mlim + lo)
    return float(half * np.sum(w * schechter(mid + half * x, mstar, alpha)))


@dataclass
class Template:
    """The cluster template of a configuration, with the distances and m*(z) of the run."""

    pc: PscdConfig
    mstar_z: np.ndarray
    mstar_m: np.ndarray
    dist_z: np.ndarray
    dist_mpc_per_deg: np.ndarray

    @classmethod
    def create(cls, pc: PscdConfig, mstar, cosmo) -> "Template":
        zt = np.asarray(cosmo.z, np.float64)
        return cls(pc, np.asarray(mstar.z, np.float64), np.asarray(mstar.m, np.float64), zt,
                   np.asarray(cosmo.mpc_per_deg(zt), np.float64))

    # ---------------------------------------------------------------- scalars of z
    def mstar(self, z):
        return np.interp(np.asarray(z, np.float64), self.mstar_z, self.mstar_m)

    def mpc_per_deg(self, z):
        return np.interp(np.asarray(z, np.float64), self.dist_z, self.dist_mpc_per_deg)

    def r200_deg(self, z):
        return self.pc.r200 / self.mpc_per_deg(z)

    def rmax_deg(self, z):
        """Truncation radius of the profile [deg]."""
        return self.pc.rtrunc * self.r200_deg(z)

    # ---------------------------------------------------------------- profile and LF
    def _sigma(self, r):
        return nfw_sigma(r, self.pc.r200 / self.pc.concentration, self.pc.rcore)

    def _norm200(self):
        return float(nfw_enclosed(self.pc.r200, self.pc.r200 / self.pc.concentration, self.pc.rcore))

    def psi(self, theta_deg, z):
        """Psi(theta) per deg^2 (theta in deg), 0 beyond the truncation."""
        D = self.mpc_per_deg(z)
        r = np.asarray(theta_deg, np.float64) * D
        out = self._sigma(r) * D * D / self._norm200()
        return np.where(r <= self.pc.rtrunc * self.pc.r200, out, 0.0)

    def phi(self, m, z):
        """Phi(m) per mag: n200 galaxies brighter than m*(z) + dmag_n200."""
        ms = float(self.mstar(z))
        norm = lumnorm(ms, ms + self.pc.dmag_n200, self.pc.alpha)
        return self.pc.n200 * schechter(m, ms, self.pc.alpha) / norm

    def kernel(self, z, pixel: float, nsub: int = 4, rmax: float | None = None) -> np.ndarray:
        """Psi on the pixel grid around a centre (odd square array), averaged over nsub^2
        sub-pixels (an even nsub never samples the centre itself); times the pixel area its sum
        is the profile's integral. ``rmax``: truncation radius [deg] (default the profile's)."""
        rmax = self.rmax_deg(z) if rmax is None else rmax
        h = int(np.ceil(rmax / pixel)) + 1
        off = (np.arange(nsub) + 0.5) / nsub - 0.5
        g = np.arange(-h, h + 1, dtype=np.float64)
        x = (g[:, None] + off[None, :]).ravel() * pixel
        th = np.hypot(x[:, None], x[None, :])
        val = self.psi(th, z)
        val = np.where(th <= rmax, val, 0.0)
        n = 2 * h + 1
        return val.reshape(n, nsub, n, nsub).mean(axis=(1, 3))


@dataclass
class WidthTable:
    """<1/s> and <1/s^2> of the galaxies' photo-z widths in (photo-z, magnitude) bins."""

    z: np.ndarray             # [nz] bin centres
    m: np.ndarray             # [nm]
    inv1: np.ndarray          # [nz, nm]
    inv2: np.ndarray

    @classmethod
    def build(cls, zphot, zphot_e, refmag, dz: float = 0.05, dm: float = 0.25,
              zrange=(0.0, 1.6), mrange=(12.0, 25.0), min_count: int = 5) -> "WidthTable":
        zphot, s, m = (np.asarray(a, np.float64) for a in (zphot, zphot_e, refmag))
        ok = (s > 0) & np.isfinite(zphot) & np.isfinite(m)
        ze = np.arange(zrange[0], zrange[1] + dz / 2, dz)
        me = np.arange(mrange[0], mrange[1] + dm / 2, dm)
        n, _, _ = np.histogram2d(zphot[ok], m[ok], [ze, me])
        s1, _, _ = np.histogram2d(zphot[ok], m[ok], [ze, me], weights=1.0 / s[ok])
        s2, _, _ = np.histogram2d(zphot[ok], m[ok], [ze, me], weights=1.0 / s[ok] ** 2)
        good = n >= min_count
        if not good.any():
            raise ValueError("no galaxy with a usable photo-z")
        inv1 = np.where(good, s1 / np.maximum(n, 1), np.nan)
        inv2 = np.where(good, s2 / np.maximum(n, 1), np.nan)
        return cls(0.5 * (ze[1:] + ze[:-1]), 0.5 * (me[1:] + me[:-1]), _fill(inv1, good), _fill(inv2, good))

    def at(self, z, m):
        """(<1/s>, <1/s^2>) at photo-z z (scalar) and magnitudes m."""
        iz = np.clip(np.searchsorted(self.z, z) - 1, 0, self.z.size - 2)
        t = np.clip((z - self.z[iz]) / (self.z[iz + 1] - self.z[iz]), 0.0, 1.0)
        a = (1 - t) * self.inv1[iz] + t * self.inv1[iz + 1]
        b = (1 - t) * self.inv2[iz] + t * self.inv2[iz + 1]
        return np.interp(m, self.m, a), np.interp(m, self.m, b)


def _fill(arr, good):
    """Empty bins take the value of the nearest filled bin (in index distance)."""
    from scipy.ndimage import distance_transform_edt

    idx = distance_transform_edt(~good, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def noise_np(bkg: ZredBkg, z: float, m) -> np.ndarray:
    """N(m, z) of the stacked photo-z background, numpy (bilinear; inf outside the table)."""
    zt = np.asarray(bkg.zred, np.float64)
    mt = np.asarray(bkg.mag, np.float64)
    sg = np.asarray(bkg.sigma_g, np.float64)
    iz = int(np.clip(np.searchsorted(zt, z) - 1, 0, zt.size - 2))
    t = float(np.clip((z - zt[iz]) / (zt[iz + 1] - zt[iz]), 0.0, 1.0))
    col = (1 - t) * sg[iz] + t * sg[iz + 1]
    m = np.asarray(m, np.float64)
    out = np.interp(m, mt, col)
    dm = 0.5 * (mt[1] - mt[0])
    return np.where((m >= mt[0] - dm) & (m <= mt[-1] + dm) & np.isfinite(out) & (out > 0), out, np.inf)


@dataclass
class MagIntegrals:
    """Cumulative magnitude integrals I1, I2, I3 [nm] of one redshift slice, on a magnitude grid."""

    m: np.ndarray
    i1: np.ndarray
    i2: np.ndarray
    i3: np.ndarray

    @classmethod
    def build(cls, tpl: Template, bkg: ZredBkg, widths: WidthTable, z: float, mag_max: float,
              dm: float = 0.02) -> "MagIntegrals":
        lo = float(tpl.mstar(z)) - 6.0
        m = np.arange(lo, max(mag_max, lo + 1.0) + dm / 2, dm)
        phi = tpl.phi(m, z)
        N = noise_np(bkg, z, m)
        inv1, inv2 = widths.at(z, m)
        g2 = inv1 / (2.0 * SQRT_PI)
        g3 = inv2 / (2.0 * np.sqrt(3.0) * np.pi)
        f1 = phi
        f2 = np.where(np.isfinite(N), phi**2 * g2 / N, 0.0)
        f3 = np.where(np.isfinite(N), phi**3 * g3 / N**2, 0.0)

        def cum(f):
            return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(m))])

        return cls(m, cum(f1), cum(f2), cum(f3))

    def at(self, mlim):
        """(I1, I2, I3) at magnitude limits ``mlim`` (any shape)."""
        mlim = np.asarray(mlim, np.float64)
        return (np.interp(mlim, self.m, self.i1), np.interp(mlim, self.m, self.i2),
                np.interp(mlim, self.m, self.i3))
