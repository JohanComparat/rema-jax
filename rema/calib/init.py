"""Initial red-sequence model from spectroscopic galaxies.

Starting from a template (BC03 colours, or a redMaPPer pars file), the red sequence is located
in the spectroscopic galaxies by iterative clipping: in sliding redshift bins, galaxies whose
colours lie within ``nsig`` x (current width) of the current model are kept, the per-bin medians
of the colour residuals update the mean colours and the robust widths update the scatter, and
the width shrinks from ``sigma_start`` towards the measured scatter. Pivot magnitudes are the
per-bin median reference magnitudes of the retained galaxies; slopes start at zero.

This mirrors redMaPPer's selection of spectroscopic red galaxies (``SelectSpecRedGalaxies``)
and produces the starting point of the EM calibration (:mod:`rema.calib.fit`).
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

from ..model.likelihood import MAG_PER_LN
from ..model.redsequence import RSModel
from ..model.splines import NaturalSpline

log = logging.getLogger(__name__)


def observed_colours(flux, ivar, iref, zeropoint: float = 22.5):
    """Adjacent colours from asinh magnitudes (softened at each band's 1-sigma error)."""
    b = 1.0 / np.sqrt(np.maximum(ivar, 1e-20))
    mu = zeropoint - MAG_PER_LN * (np.arcsinh(flux / (2 * b)) + np.log(b))
    return mu[:, :-1] - mu[:, 1:], mu[:, iref]


def initial_model(flux, ivar, refmag, zspec, rs0: RSModel, mstar_fn, *, zrange=(0.05, 0.9),
                  dz_bin: float = 0.04, dmag: float = 1.75, sigma_start: float = 0.15,
                  nsig: float = 2.0, niter: int = 6, min_gal: int = 30) -> tuple[RSModel, np.ndarray]:
    """Refine ``rs0`` on spectroscopic galaxies. Returns (model, mask of retained galaxies)."""
    flux = np.asarray(flux, np.float64)
    ivar = np.asarray(ivar, np.float64)
    zspec = np.asarray(zspec, np.float64)
    refmag = np.asarray(refmag, np.float64)
    ok = (zspec > zrange[0] - 0.02) & (zspec < zrange[1] + 0.02)
    ok &= refmag < np.asarray(mstar_fn(np.clip(zspec, 0.01, 2.0))) + dmag
    col, _ = observed_colours(flux, ivar, rs0.iref)
    # Colour errors from the asinh-magnitude errors (diagonal approximation).
    b = 1.0 / np.sqrt(np.maximum(ivar, 1e-20))
    smu = MAG_PER_LN * b / np.sqrt(flux**2 + 4 * b**2)
    cerr = np.sqrt(smu[:, :-1] ** 2 + smu[:, 1:] ** 2)
    ncol = rs0.ncol
    zm = np.asarray(rs0.z_mean)
    zs = np.asarray(rs0.z_sigma)
    zp = np.asarray(rs0.z_pivot)
    mean = np.asarray(rs0.at(jnp.asarray(zm)).mean, np.float64)
    width = np.full((zm.size, ncol), sigma_start)
    keep = ok.copy()
    spl = NaturalSpline(zm)
    at0 = rs0.at(jnp.asarray(np.clip(zspec, zm[0], zm[-1])))
    slope_term = np.asarray(at0.slope, np.float64) * (refmag - np.asarray(at0.pivot, np.float64))[:, None]
    for it in range(niter):
        model_c = spl(np.clip(zspec, zm[0], zm[-1]), mean) + slope_term
        w_at = np.stack([np.interp(zspec, zm, width[:, j]) for j in range(ncol)], axis=1)
        tot = np.sqrt(w_at**2 + cerr**2)
        resid = col - model_c
        keep = ok & np.all(np.abs(resid) < nsig * tot, axis=1)
        new_mean, new_width = mean.copy(), width.copy()
        for k, z0 in enumerate(zm):
            sel = keep & (np.abs(zspec - z0) < dz_bin)
            if sel.sum() < min_gal:
                continue
            r = resid[sel]
            new_mean[k] = mean[k] + np.median(r, axis=0)
            mad = 1.4826 * np.median(np.abs(r - np.median(r, axis=0)), axis=0)
            intr = np.sqrt(np.maximum(mad**2 - np.median(cerr[sel] ** 2, axis=0), 0.01**2))
            new_width[k] = np.maximum(intr, 0.5 * width[k])
        mean, width = new_mean, new_width
        log.info("initial model iteration %d: %d galaxies retained", it, int(keep.sum()))
    sig = np.stack([np.interp(zs, zm, width[:, j]) for j in range(ncol)], axis=1)
    piv = np.asarray(rs0.at(jnp.asarray(zp)).pivot, np.float64)
    for k, z0 in enumerate(zp):
        sel = keep & (np.abs(zspec - z0) < 0.05)
        if sel.sum() >= min_gal:
            piv[k] = np.median(refmag[sel])
    rs = RSModel.from_arrays(
        nodes={"mean": zm, "slope": rs0.z_slope, "sigma": zs, "corr": rs0.z_corr, "pivot": zp},
        values={"mean": mean, "slope": np.asarray(rs0.slope), "log_sigma": np.log(sig),
                "corr": np.asarray(rs0.corr), "pivot": piv},
        bands=rs0.bands, ref_band=rs0.ref_band, min_sigma=rs0.min_sigma)
    return rs, keep
