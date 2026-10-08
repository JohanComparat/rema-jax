"""Null tests: galaxy tables whose real structures have lost their signature.

Positions and reference magnitudes are kept, so the angular clustering, the depth and the
magnitude distribution are those of the data:

- ``"photoz"`` permutes the (ZPHOT, ZPHOT_STD) pairs among the galaxies of the same reference
  magnitude bin. A real cluster keeps its angular overdensity but its members get redshifts drawn
  from the field, so detections in this table measure the false detections of a photo-z finder
  (chance alignments and projections of the angular clustering).
- ``"colour"`` permutes the fluxes in the same way, each row rescaled to the reference flux of the
  galaxy that receives it (the inverse variances by the square of the factor). The red sequence of
  real clusters is lost: the same test for the red-sequence filter.

The permutations are seeded (:class:`rema.config.NullConfig`), so a null run is reproducible.
"""

from __future__ import annotations

import numpy as np

from ..config import NullConfig

MODES = ("none", "photoz", "colour")


def _permutation(refmag, magbin: float, rng) -> np.ndarray:
    """A permutation of the rows that only exchanges galaxies of the same magnitude bin."""
    m = np.asarray(refmag, np.float64)
    key = np.floor(np.where(np.isfinite(m), m, 0.0) / magbin).astype(np.int64)
    order = np.argsort(key, kind="stable")
    k = key[order]
    perm = np.empty(m.size, np.int64)
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    ends = np.r_[starts[1:], m.size]
    for a, b in zip(starts, ends):
        rows = order[a:b]
        perm[rows] = rows[rng.permutation(b - a)]
    return perm


def null_shuffle(gal: dict, ncfg: NullConfig, iref: int) -> dict:
    """A copy of ``gal`` shuffled as ``ncfg.shuffle`` says (``gal`` itself for "none").

    ``iref``: the column of the reference band in FLUX and FLUX_IVAR.
    """
    if ncfg.shuffle not in MODES:
        raise ValueError(f"null.shuffle {ncfg.shuffle!r}: expected one of {MODES}")
    if ncfg.shuffle == "none":
        return gal
    rng = np.random.default_rng(ncfg.seed)
    perm = _permutation(gal["REFMAG"], ncfg.magbin, rng)
    out = dict(gal)
    if ncfg.shuffle == "photoz":
        for col in ("ZPHOT", "ZPHOT_STD"):
            if col not in gal:
                raise KeyError(f"null.shuffle photoz: the galaxy table has no {col}")
            out[col] = np.asarray(gal[col])[perm]
        return out
    flux = np.asarray(gal["FLUX"], np.float64)
    ivar = np.asarray(gal["FLUX_IVAR"], np.float64)
    fref = flux[:, iref]
    src = fref[perm]
    scale = np.where(src > 0, fref / np.where(src > 0, src, 1.0), 1.0)
    out["FLUX"] = (flux[perm] * scale[:, None]).astype(np.asarray(gal["FLUX"]).dtype)
    out["FLUX_IVAR"] = (ivar[perm] / (scale[:, None] ** 2)).astype(np.asarray(gal["FLUX_IVAR"]).dtype)
    # the reference band is the galaxy's own (same flux, same error)
    out["FLUX"][:, iref] = np.asarray(gal["FLUX"])[:, iref]
    out["FLUX_IVAR"][:, iref] = np.asarray(gal["FLUX_IVAR"])[:, iref]
    return out
