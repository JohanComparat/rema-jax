"""Richness at fixed positions and fixed redshifts on a nested HEALPix grid (scan-mode randoms).

The randoms of :mod:`rema.randoms` place copies of the clusters on the footprint without looking
at the galaxies there. This module measures what the finder records at a position that holds no
cluster: at every grid point and every redshift z_k of a grid, lambda(z_k) in the percolation
aperture (r_lambda = r0 (lambda / 100)^beta, solved with lambda), with LAMBDA_E, R_LAMBDA,
SCALEVAL, MASKFRAC and LNLAMLIKE. Unlike the scan mode (:mod:`rema.modes.scan`) the position and
the redshift are not moved: no maximum-likelihood redshift, no z_lambda iteration, no centring.
Those would carry every random point to the nearest overdensity.

Grid
----
The points are centres of HEALPix pixels at ``nside_max`` (NESTED, default 4096, 0.74 arcmin2).
A pixel of a coarser ``nside`` is represented by its sub-pixel of lowest nested index (the first
of its ``(nside_max / nside)**2`` sub-pixels), so the points of ``nside`` are those whose nested
index at ``nside_max`` is a multiple of ``(nside_max / nside)**2``. Level ``nside`` adds the
points of ``nside`` that ``nside / 2`` does not have (3 per pixel of ``nside / 2``): the levels
64, 128, ..., ``nside`` together are one point per pixel of ``nside``, and a finer run reuses
every point measured before. A point lies within one ``nside_max`` pixel (0.86 arcmin at 4096)
of the centre of the pixel it represents.

>>> level_offsets(4, nside_max=8, base=2)                    # nested indices at NSIDE 8
array([ 4,  8, 12])
>>> level_offsets(2, nside_max=8, base=2)
array([0])
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np

from ..core.richness import RadialQuad, Stage, richness_one
from .common import Region
from .scan import _completeness_grid, _z_chunks

log = logging.getLogger(__name__)

NSIDE_MAX = 4096
NSIDE_BASE = 64


def _check_nside(nside: int, nside_max: int, base: int) -> int:
    for n, name in ((nside, "nside"), (nside_max, "nside_max"), (base, "base")):
        if n < 1 or n & (n - 1):
            raise ValueError(f"{name} = {n} is not a power of two")
    if not base <= nside <= nside_max:
        raise ValueError(f"need base ({base}) <= nside ({nside}) <= nside_max ({nside_max})")
    return (nside_max // nside) ** 2


def level_offsets(nside: int, nside_max: int = NSIDE_MAX, base: int = NSIDE_BASE) -> np.ndarray:
    """Return the nested indices at ``nside_max``, within one pixel of ``nside / 2``, of the points that
    level ``nside`` adds (for ``nside == base``: the one point of a pixel of ``base``, index 0)."""
    step = _check_nside(nside, nside_max, base)
    if nside == base:
        return np.zeros(1, np.int64)
    return np.arange(1, 4, dtype=np.int64) * step


def level_pixels(nside: int, nside_max: int = NSIDE_MAX, base: int = NSIDE_BASE,
                 parents: np.ndarray | None = None) -> np.ndarray:
    """Return the nested indices at ``nside_max`` of the points of level ``nside``, sorted.

    ``parents``: nested indices at ``base`` of the pixels to cover (default the whole sky).

    >>> a = np.concatenate([level_pixels(n, 8, 2) for n in (2, 4, 8)])
    >>> bool(np.array_equal(np.sort(a), np.arange(12 * 8 * 8)))   # every pixel of NSIDE 8 once
    True
    """
    _check_nside(nside, nside_max, base)
    parents = np.arange(12 * base * base, dtype=np.int64) if parents is None else np.asarray(parents, np.int64)
    per_base = (nside_max // base) ** 2
    half = (nside_max // (nside // 2)) ** 2 if nside > base else per_base
    # pixels of nside / 2 inside each base pixel, then the new offsets inside each of them
    within = np.arange(0, per_base, half, dtype=np.int64)
    off = (within[:, None] + level_offsets(nside, nside_max, base)[None, :]).ravel()
    return np.sort((parents[:, None] * per_base + off[None, :]).ravel())


def grid_points(nside: int, domain=None, nside_max: int = NSIDE_MAX, base: int = NSIDE_BASE,
                chunk: int = 4096) -> dict:
    """Return the points of level ``nside`` inside ``domain`` (a :class:`rema.randoms.domain.RunDomain`,
    or None for the whole sky): columns PIX (nested at ``nside_max``), RA, DEC, LEVEL (= nside).

    The base pixels are processed ``chunk`` at a time, and a base pixel whose sub-pixels are all
    outside the domain costs one test per point of the level only.
    """
    import healpy as hp

    _check_nside(nside, nside_max, base)
    nbase = 12 * base * base
    pix, ra, dec = [], [], []
    for b0 in range(0, nbase, chunk):
        p = level_pixels(nside, nside_max, base, np.arange(b0, min(nbase, b0 + chunk)))
        r, d = hp.pix2ang(nside_max, p, nest=True, lonlat=True)
        if domain is not None:
            keep = domain.contains(r, d)
            p, r, d = p[keep], r[keep], d[keep]
        pix.append(p)
        ra.append(r)
        dec.append(d)
    pix = np.concatenate(pix)
    return {"PIX": pix, "RA": np.concatenate(ra), "DEC": np.concatenate(dec),
            "LEVEL": np.full(pix.size, nside, np.int16)}


# --------------------------------------------------------------------------- measurement
@jax.jit
def _grid_kernel(nb, zsteps, frad, fgeo, quad, model, stage):
    """Return (lambda, lambda_e, r_lambda, scaleval, maskfrac, lnlamlike), each [B, S]."""
    def one_cluster(nb1, fr, fg):
        def one_step(z, fr1, fg1):
            r = richness_one(nb1, z, fr1, fg1, quad, model, stage)
            return r.lam, r.lam_e, r.r_lambda, r.scaleval, r.maskfrac, r.lnlamlike
        return jax.vmap(one_step)(zsteps, fr, fg)

    return jax.vmap(one_cluster)(nb, frad, fgeo)


GRID_COLUMNS = ("LAMBDA", "LAMBDA_E", "R_LAMBDA", "SCALEVAL", "MASKFRAC", "LNLAMLIKE")


def grid_zsteps(cfg, zrange=None, zstep: float = 0.02) -> np.ndarray:
    """Return the redshift grid: ``zrange`` (default the model's) in steps of ``zstep``.

    >>> from rema.config import RemaConfig
    >>> grid_zsteps(RemaConfig(), (0.1, 0.2), 0.05).tolist()
    [0.1, 0.15, 0.2]
    """
    z0, z1 = zrange if zrange is not None else cfg.model.zrange
    return np.round(np.arange(z0, z1 + zstep / 2, zstep), 6)


def run_grid(region: Region, ra, dec, ids=None, zsteps=None, batch: int = 128):
    """Measure lambda(z_k) at fixed positions and return (catalogue, {}): MEM_MATCH_ID, RA, DEC and the
    columns of GRID_COLUMNS as ``<NAME>_STEPS`` [N, S] (float32; -1 where lambda < 1, LNLAMLIKE -inf there, MASKFRAC always set)."""
    cfg = region.cfg
    ra = np.atleast_1d(np.asarray(ra, np.float64))
    dec = np.atleast_1d(np.asarray(dec, np.float64))
    ids = np.arange(ra.size) if ids is None else np.atleast_1d(np.asarray(ids))
    zsteps = grid_zsteps(cfg) if zsteps is None else np.asarray(zsteps, np.float64)
    zstep = float(np.median(np.diff(zsteps))) if zsteps.size > 1 else 0.1
    st = Stage.make(cfg.richness.percolation.r0, cfg.richness.percolation.beta, cfg.richness.maxrad_factor)
    quad = RadialQuad.make(rmax=float(st.maxrad) + 5 * cfg.model.rsig)
    chunks = _z_chunks(zsteps, zstep)
    out = {k: [] for k in GRID_COLUMNS}
    for b0 in range(0, ra.size, batch):
        sl = slice(b0, min(ra.size, b0 + batch))
        bra, bdec = ra[sl], dec[sl]
        steps = {k: [] for k in GRID_COLUMNS}
        for zc in chunks:
            rad = region.radius_deg(float(st.maxrad), zc.min())
            pad = region.query(bra, bdec, rad, mag_max=float(region.maxmag(zc.max())) + 1.0)
            nb = region.neighbors(pad)
            fr, fg = _completeness_grid(region, bra, bdec, zc, quad)
            res = _grid_kernel(nb, jnp.asarray(zc, jnp.float32), fr, fg, quad, region.model, st)
            for k, v in zip(GRID_COLUMNS, res):
                steps[k].append(np.asarray(v, np.float32))
        for k in GRID_COLUMNS:
            out[k].append(np.concatenate(steps[k], axis=1))
    cat = {"MEM_MATCH_ID": ids, "RA": ra, "DEC": dec}
    for k in GRID_COLUMNS:
        cat[f"{k}_STEPS"] = (np.concatenate(out[k]) if out[k]
                             else np.zeros((0, zsteps.size), np.float32))
    return cat, {}
