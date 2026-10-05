"""Mask and depth completeness of a cluster aperture (replaces redMaPPer's Monte-Carlo maskgals).

For a cluster at (ra, dec, z), points on a polar grid (Gauss-Legendre radii of
:class:`rema.core.richness.RadialQuad` times ``n_phi`` azimuths) are looked up in the footprint
map built from the DR11 randoms (:mod:`rema.sky.maps`):

- FRACGOOD(x): fraction of the area that is observed and unmasked;
- S(x): fraction of the cluster's galaxies that pass the catalogue selection at the local depth,

      S = int phi(m) theta_i(m) P_sel(m) dm / int phi(m) theta_i(m) dm,
      P_sel(m) = Phi((F(m) - F_cut) / sigma_f(x)),  F_cut = max(snr_min sigma_f, F(mag_max)),

  with phi the Schechter function at z, theta_i the soft cut at m*(z) + 1.75 and sigma_f the
  local 1-sigma flux error of the reference band.

The azimuthal means give the radial completeness ``frad`` (FRACGOOD x S) and ``fgeo``
(FRACGOOD), from which :mod:`rema.core.richness` computes K(r_lambda) inside the solve and
MASKFRAC = 1 - K_geo(r_lambda).
"""

from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr

from ..model import profiles as prof
from ..sky.maps import DeviceMap
from .context import FilterModel
from .richness import RadialQuad


@lru_cache(maxsize=8)
def _gl(n):
    return np.polynomial.legendre.leggauss(n)


def selection_completeness(sigf, mstar, maxmag, alpha: float, snr_min: float, mag_max: float,
                           n_mag: int = 24, zeropoint: float = 22.5):
    """S(sigma_f) for arrays ``sigf`` (broadcast with mstar/maxmag)."""
    x, w = _gl(n_mag)
    lo = mstar - 6.0
    hi = maxmag + 1.0
    m = 0.5 * (hi - lo)[..., None] * x + 0.5 * (hi + lo)[..., None]          # [..., n_mag]
    wm = 0.5 * (hi - lo)[..., None] * w
    F = 10.0 ** (-0.4 * (m - zeropoint))
    sig = jnp.maximum(sigf, 1e-12)[..., None]
    fcut = jnp.maximum(snr_min * sig, 10.0 ** (-0.4 * (mag_max - zeropoint)))
    psel = ndtr((F - fcut) / sig)
    sigma_m = 2.5 / np.log(10.0) * sig / F
    phi = prof.schechter(m, mstar[..., None], alpha) * prof.theta_i(m, maxmag[..., None], sigma_m)
    num = jnp.sum(wm * phi * psel, axis=-1)
    den = jnp.sum(wm * phi, axis=-1)
    return jnp.where(sigf > 0, num / jnp.maximum(den, 1e-30), 0.0)


@partial(jax.jit, static_argnames=("ref_band", "n_phi", "n_mag"))
def radial_completeness(ra, dec, z, fmap: DeviceMap, quad: RadialQuad, model: FilterModel,
                        ref_band: str = "Z", snr_min: float = 5.0, mag_max: float = 30.0,
                        n_phi: int = 16, n_mag: int = 24):
    """(frad, fgeo), each [B, n_r], for clusters at ``ra``, ``dec`` [deg] and ``z`` [B]."""
    phi = (jnp.arange(n_phi) + 0.5) * (2.0 * jnp.pi / n_phi)                 # from north to east
    D = model.mpc_per_deg(z)                                                    # [B]
    th = jnp.deg2rad(quad.r[None, :, None] / D[:, None, None])                  # [B, n_r, 1] rad
    # Exact spherical offsets (valid up to the poles, unlike RA steps of dtheta / cos Dec).
    d = jnp.deg2rad(dec)[:, None, None]
    sin_d2 = jnp.sin(d) * jnp.cos(th) + jnp.cos(d) * jnp.sin(th) * jnp.cos(phi)
    pdec = jnp.rad2deg(jnp.arcsin(jnp.clip(sin_d2, -1.0, 1.0)))
    pra = ra[:, None, None] + jnp.rad2deg(jnp.arctan2(jnp.sin(phi) * jnp.sin(th) * jnp.cos(d),
                                                      jnp.cos(th) - jnp.sin(d) * sin_d2))
    good = fmap.lookup(pra, pdec, "FRACGOOD")
    sigf = fmap.lookup(pra, pdec, f"SIGF_{ref_band.upper()}")
    mstar = model.mstar(z)[:, None, None]
    maxmag = model.maxmag(z)[:, None, None]
    S = selection_completeness(sigf, jnp.broadcast_to(mstar, sigf.shape),
                               jnp.broadcast_to(maxmag, sigf.shape), model.alpha, snr_min,
                               mag_max, n_mag, model.zeropoint)
    return jnp.mean(good * S, axis=-1), jnp.mean(good, axis=-1)


def no_footprint(nclusters: int, quad: RadialQuad):
    """Completeness arrays for runs without a footprint (everything observed)."""
    ones = jnp.ones((nclusters, quad.r.shape[0]))
    return ones, ones
