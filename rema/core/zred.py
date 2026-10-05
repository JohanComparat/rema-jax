"""zred: photometric redshift of a galaxy assuming it is on the red sequence.

As in redMaPPer (``ZredColor``), on a regular grid z_k:

    ln D(z_k) = -chi^2(z_k)/2 [- ln det S/2] + ln phi(m_ref; m*(z_k), alpha) + ln V(z_k),

with phi the Schechter function and V(z) = E(z)/E(z_ref) redMaPPer's volume factor. Grid points
where the galaxy is outside the magnitude range of the red sequence (brighter than m* - 4 or
fainter than m* + 2.5) are excluded. Then:

- zred_uncorr, zred_uncorr_e: mean and standard deviation of exp(ln D) over points with
  D > 1e-5 D_max (width floored at 0.005);
- a parabola through the 5 grid points around the peak replaces the mean when its vertex lies
  within 2 sigma of it;
- chisq is chi^2 at the grid point nearest to zred;
- the calibrated corrections (:class:`rema.model.redsequence.ZredCorrection`) give zred, zred_e.

Galaxies are processed in fixed-size chunks so that the kernel compiles once.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..model.likelihood import chisq as chisq_fn
from ..model.redsequence import RSAt, RSModel, ZredCorrection

LN10 = np.log(10.0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ZredGrid:
    """Red-sequence model tabulated on the zred grid."""

    z: jnp.ndarray          # [nz]
    mean: jnp.ndarray       # [nz, ncol]
    slope: jnp.ndarray
    cint: jnp.ndarray       # [nz, ncol, ncol]
    pivot: jnp.ndarray      # [nz]
    mstar: jnp.ndarray      # [nz]
    lnvol: jnp.ndarray      # [nz]

    @classmethod
    def create(cls, rs: RSModel, mstar, cosmo, zrange, dz: float = 0.005, zref: float | None = None,
               margin: float = 0.1) -> "ZredGrid":
        lo = max(0.01, zrange[0] - margin / 2)
        hi = zrange[1] + margin
        z = np.arange(lo, hi + dz / 2, dz)
        at = rs.at(jnp.asarray(z))
        zref = zrange[1] if zref is None else zref
        lnvol = jnp.log(cosmo.volume_factor(jnp.asarray(z), zref))
        return cls(jnp.asarray(z), at.mean, at.slope, at.cint, at.pivot, mstar(jnp.asarray(z)), lnvol)


@dataclass(frozen=True)
class ZredResult:
    zred: np.ndarray
    zred_e: np.ndarray
    zred_uncorr: np.ndarray
    zred_uncorr_e: np.ndarray
    chisq: np.ndarray

    def as_columns(self) -> dict:
        return {"ZRED": self.zred, "ZRED_E": self.zred_e, "ZRED_UNCORR": self.zred_uncorr,
                "ZRED_UNCORR_E": self.zred_uncorr_e, "ZRED_CHISQ": self.chisq}


@partial(jax.jit, static_argnames=("iref", "mode", "use_lndet"))
def _zred_kernel(flux, ivar, grid: ZredGrid, iref: int, mode: str, eps: float, alpha: float,
                 use_lndet: bool, dmag_bright: float, dmag_faint: float, zred_e_min: float):
    at = RSAt(grid.mean[None], grid.slope[None], grid.cint[None], grid.pivot[None])
    chi2, logdet = chisq_fn(flux[:, None, :], ivar[:, None, :], at, iref, mode, eps)   # [N, nz]
    refmag = 22.5 - 2.5 * jnp.log10(jnp.maximum(flux[:, iref], 1e-10))[:, None]
    dm = grid.mstar[None, :] - refmag
    lnphi = (alpha + 1.0) * 0.4 * LN10 * dm - 10.0 ** (0.4 * dm)
    lnd = -0.5 * chi2 + lnphi + grid.lnvol[None, :]
    if use_lndet:
        lnd = lnd - 0.5 * logdet
    valid = (dm > -dmag_faint) & (dm < dmag_bright) & (chi2 < 1e5) & jnp.isfinite(lnd)
    lnd = jnp.where(valid, lnd, -jnp.inf)
    nvalid = jnp.sum(valid, axis=1)
    mx = jnp.max(lnd, axis=1, keepdims=True)
    dist = jnp.exp(lnd - mx)
    dist = jnp.where(dist > 1e-5, dist, 0.0)
    z = grid.z[None, :]
    w = jnp.sum(dist, axis=1)
    zmean = jnp.sum(dist * z, axis=1) / w
    var = jnp.sum(dist * z * z, axis=1) / w - zmean**2
    zerr = jnp.where(var > 0, jnp.sqrt(jnp.maximum(var, 0.0)), 1.0)
    zerr = jnp.maximum(zerr, zred_e_min)
    # Parabola through 5 points around the peak (symmetric stencil: closed-form fit).
    nz = grid.z.shape[0]
    imax = jnp.argmax(lnd, axis=1)
    ic = jnp.clip(imax, 2, nz - 3)
    offs = jnp.arange(-2, 3)
    idx = ic[:, None] + offs[None, :]
    y = jnp.take_along_axis(lnd, idx, axis=1)
    dz = grid.z[1] - grid.z[0]
    x = offs * dz
    b = jnp.sum(x * y, axis=1) / jnp.sum(x * x)
    x2c = x * x - jnp.mean(x * x)
    a = jnp.sum(x2c * y, axis=1) / jnp.sum(x2c * x2c)
    ok = jnp.all(jnp.isfinite(y), axis=1) & (a < 0)
    vertex = grid.z[ic] - b / (2.0 * jnp.where(ok, a, -1.0))
    use = ok & (jnp.abs(vertex - zmean) < 2.0 * zerr)
    zred = jnp.where(use, vertex, zmean)
    inear = jnp.clip(jnp.round((zred - grid.z[0]) / dz).astype(jnp.int32), 0, nz - 1)
    chisq_at = jnp.take_along_axis(chi2, inear[:, None], axis=1)[:, 0]
    good = nvalid >= 2
    return (jnp.where(good, zred, -1.0), jnp.where(good, zerr, -1.0),
            jnp.where(good, chisq_at, -1.0))


def compute_zred(flux: np.ndarray, ivar: np.ndarray, grid: ZredGrid, iref: int, *,
                 mode: str = "lupt", eps: float = 0.015, alpha: float = -1.0,
                 use_lndet: bool = False, correction: ZredCorrection | None = None,
                 rs: RSModel | None = None, dmag_bright: float = 4.0, dmag_faint: float = 2.5,
                 zred_e_min: float = 0.005, chunk: int = 16384) -> ZredResult:
    """zred for every galaxy (arrays ``flux``, ``ivar`` of shape [N, nband])."""
    flux = np.asarray(flux, np.float32)
    ivar = np.asarray(ivar, np.float32)
    n = flux.shape[0]
    out_z, out_e, out_c = (np.empty(n, np.float32) for _ in range(3))
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        f = np.zeros((chunk, flux.shape[1]), np.float32)
        iv = np.ones((chunk, flux.shape[1]), np.float32)
        f[:hi - lo], iv[:hi - lo] = flux[lo:hi], ivar[lo:hi]
        f[hi - lo:, :] = 1.0
        z, e, c = _zred_kernel(jnp.asarray(f), jnp.asarray(iv), grid, iref, mode, eps, alpha,
                               use_lndet, dmag_bright, dmag_faint, zred_e_min)
        out_z[lo:hi], out_e[lo:hi], out_c[lo:hi] = (np.asarray(v)[:hi - lo] for v in (z, e, c))
    zred, zred_e = out_z.copy(), out_e.copy()
    if correction is not None and rs is not None:
        good = out_z > 0
        refmag = 22.5 - 2.5 * np.log10(np.maximum(flux[:, iref], 1e-10))
        zc, ec = correction.apply(jnp.asarray(out_z[good]), jnp.asarray(out_e[good]),
                                  jnp.asarray(refmag[good]), lambda zz: rs.at(zz).pivot)
        zred[good], zred_e[good] = np.asarray(zc), np.asarray(ec)
    return ZredResult(zred, zred_e, out_z, out_e, out_c)
