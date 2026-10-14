"""Aperture completeness of a cluster placed at any position (redMaPPer's RunRandomsZmask).

A cluster of richness lambda at redshift z placed at (ra, dec) is not re-measured: its aperture
r_lambda = r0 (lambda / 100)^beta (the percolation stage) is laid on the footprint map, and the
radial completeness of :func:`rema.core.maskcorr.footprint_completeness` (the call the cluster
finders make) gives

    K = <FRACGOOD x S>_NFW(r < r_lambda),   K_geo = <FRACGOOD>_NFW(r < r_lambda),

with the profile weighting of the membership solve (:func:`rema.core.richness.aperture_k`). At
the richness solution sum pmem = lambda K, so the catalogue's SCALEVAL = lambda / sum pmem is
1 / K and MASKFRAC = 1 - K_geo: the values the finder would have recorded for that cluster at
that position. Outside the footprint K = K_geo = 0 (SCALEVAL = inf, MASKFRAC = 1).

The blind catalogue computed the completeness of each cluster at its seed galaxy and at the
redshift of its likelihood pass (``rema.modes.blind.percolate``), which is not stored; z_lambda
before the correction (Z_LAMBDA_RAW) differs from it by a few 1e-3 and changes SCALEVAL by less
than that.
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..core.context import FilterModel
from ..core.maskcorr import footprint_completeness
from ..core.richness import RadialQuad, aperture_k, percolation_quad
from ..sky.maps import DeviceMap, Footprint

log = logging.getLogger(__name__)


def aperture_model(cfg, cosmo=None) -> FilterModel:
    """The parts of the filter model the aperture needs (m*(z), distances, NFW profile), without
    a red-sequence model or background."""
    if cfg.model.filter != "redsequence":
        cfg = cfg.replace(model={"filter": "redsequence"})
    return FilterModel.create(None, None, cfg, cosmo=cosmo)


def footprint_map(fp: Footprint, ref_band: str) -> DeviceMap:
    """Device copy of the two footprint columns the aperture reads."""
    return fp.fine.device(("FRACGOOD", f"SIGF_{ref_band.upper()}"))


@jax.jit
def _k_batch(lam, r0, beta, frad, fgeo, quad: RadialQuad, model: FilterModel):
    rl = r0 * (lam / 100.0) ** beta
    k, kg = jax.vmap(aperture_k, in_axes=(0, 0, 0, None, None))(rl, frad, fgeo, quad, model)
    return rl, k, kg


def aperture_values(ra, dec, z, lam, fmap: DeviceMap, cfg, *, model: FilterModel | None = None,
                    quad: RadialQuad | None = None, ref_band: str | None = None,
                    batch: int = 16384) -> dict:
    """R_LAMBDA, SCALEVAL and MASKFRAC (float64 arrays) of clusters of richness ``lam`` at
    redshift ``z`` placed at (``ra``, ``dec``) [deg].

    ``cfg``: the catalogue's configuration (percolation aperture, selection, NFW profile).
    ``fmap``: :func:`footprint_map` of the footprint. Positions are processed in batches of
    ``batch`` (the last one padded, so that every call has the same shape).

    >>> from rema.config import RemaConfig
    >>> from rema.sky.maps import SparseMap
    >>> npix = 12 * 64**2                     # a deep full-sky map: nothing lost
    >>> fm = SparseMap(64, np.arange(npix), {"FRACGOOD": np.ones(npix, np.float32),
    ...                "SIGF_Z": np.full(npix, 1e-3, np.float32)}).device(("FRACGOOD", "SIGF_Z"))
    >>> out = aperture_values([10.0], [0.0], [0.3], [20.0], fm, RemaConfig(), batch=4)
    >>> round(float(out["SCALEVAL"][0]), 4), round(float(out["MASKFRAC"][0]), 4)
    (1.0, 0.0)
    """
    ra, dec, z, lam = (np.atleast_1d(np.asarray(a, np.float64)) for a in (ra, dec, z, lam))
    n = ra.size
    model = model if model is not None else aperture_model(cfg)
    quad = quad if quad is not None else percolation_quad(cfg)
    ref_band = ref_band or cfg.survey.ref_band
    pc = cfg.richness.percolation
    r0, beta = jnp.float32(pc.r0), jnp.float32(pc.beta)
    out = {k: np.empty(n, np.float64) for k in ("R_LAMBDA", "SCALEVAL", "MASKFRAC")}
    t0, every = time.time(), max(1, (n // batch) // 20)
    for ib, s0 in enumerate(range(0, n, batch)):
        if ib and ib % every == 0:
            log.info("aperture: %d/%d (%.0fs)", s0, n, time.time() - t0)
        idx = np.arange(s0, s0 + batch).clip(max=n - 1)       # pad with the last row
        frad, fgeo = footprint_completeness(ra[idx], dec[idx], z[idx], fmap, quad, model, cfg, ref_band)
        rl, k, kg = (np.asarray(a, np.float64) for a in
                     _k_batch(jnp.asarray(lam[idx], jnp.float32), r0, beta, frad, fgeo, quad, model))
        m = min(batch, n - s0)
        sl = slice(s0, s0 + m)
        out["R_LAMBDA"][sl] = rl[:m]
        with np.errstate(divide="ignore"):
            out["SCALEVAL"][sl] = np.where(k[:m] > 0, 1.0 / k[:m], np.inf)
        out["MASKFRAC"][sl] = 1.0 - kg[:m]
    return out
