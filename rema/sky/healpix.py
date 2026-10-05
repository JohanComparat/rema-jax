"""HEALPix NESTED pixel indexing in JAX (for on-device map lookups).

A transcription of ``ang2pix_nest_z_phi`` from the HEALPix C library (Gorski et al. 2005). Pixel
numbers are int32, which covers nside <= 8192. Galaxy pixels are computed on the host with
healpy; this version serves quadrature points inside jitted code. In float32 a small fraction of
points within ~1e-7 rad of a pixel edge can land in the neighbouring pixel, which is harmless
for map lookups.

>>> import numpy as np, healpy as hp
>>> ra, dec = np.array([10.3, 45.7, 200.1]), np.array([1.2, 60.4, -75.3])
>>> bool(np.all(np.asarray(ang2pix_nest(64, ra, dec)) == hp.ang2pix(64, ra, dec, nest=True, lonlat=True)))
True
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

TWOTHIRD = 2.0 / 3.0


def _spread_bits(v):
    """Interleave the low 16 bits of ``v`` with zeros (Morton code helper)."""
    v = v.astype(jnp.uint32) & 0xFFFF
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    v = (v | (v << 1)) & 0x55555555
    return v


def ang2pix_nest(nside: int, ra, dec):
    """NESTED pixel index of (ra, dec) in degrees, as int32.

    ``nside`` must be a Python int (power of two, <= 8192).
    """
    if nside > 8192 or nside & (nside - 1):
        raise ValueError("nside must be a power of two <= 8192")
    ra = jnp.asarray(ra)
    dec = jnp.asarray(dec)
    z = jnp.sin(jnp.deg2rad(dec))
    phi = jnp.deg2rad(ra)
    za = jnp.abs(z)
    tt = jnp.mod(phi, 2.0 * np.pi) / (0.5 * np.pi)          # in [0, 4)
    ns = nside

    # Equatorial region.
    temp1 = ns * (0.5 + tt)
    temp2 = ns * (z * 0.75)
    jp = jnp.floor(temp1 - temp2).astype(jnp.int32)          # ascending edge line
    jm = jnp.floor(temp1 + temp2).astype(jnp.int32)          # descending edge line
    ifp = jp // ns
    ifm = jm // ns
    face_eq = jnp.where(ifp == ifm, ifp | 4, jnp.where(ifp < ifm, ifp, ifm + 8))
    ix_eq = jm & (ns - 1)
    iy_eq = ns - (jp & (ns - 1)) - 1

    # Polar caps.
    ntt = jnp.minimum(jnp.floor(tt).astype(jnp.int32), 3)
    tp = tt - ntt
    tmp = ns * jnp.sqrt(3.0 * jnp.maximum(1.0 - za, 0.0))
    jp_p = jnp.minimum(jnp.floor(tp * tmp).astype(jnp.int32), ns - 1)
    jm_p = jnp.minimum(jnp.floor((1.0 - tp) * tmp).astype(jnp.int32), ns - 1)
    north = z >= 0
    face_p = jnp.where(north, ntt, ntt + 8)
    ix_p = jnp.where(north, ns - jm_p - 1, jp_p)
    iy_p = jnp.where(north, ns - jp_p - 1, jm_p)

    equatorial = za <= TWOTHIRD
    face = jnp.where(equatorial, face_eq, face_p).astype(jnp.int32)
    ix = jnp.where(equatorial, ix_eq, ix_p)
    iy = jnp.where(equatorial, iy_eq, iy_p)
    sub = _spread_bits(ix) | (_spread_bits(iy) << 1)
    return face * (ns * ns) + sub.astype(jnp.int32)


def ang2pix_nest_np(nside: int, ra, dec) -> np.ndarray:
    """Host version (healpy), int64."""
    import healpy as hp

    return hp.ang2pix(nside, np.asarray(ra), np.asarray(dec), nest=True, lonlat=True).astype(np.int64)


def pix_area_deg2(nside: int) -> float:
    return 4.0 * np.pi * (180.0 / np.pi) ** 2 / (12 * nside * nside)
