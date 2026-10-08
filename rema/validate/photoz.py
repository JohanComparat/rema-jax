"""Calibration of the galaxy photo-z widths against spectroscopic redshifts.

For the galaxies with both a photo-z and a spectroscopic redshift, x = (ZPHOT - ZSPEC)/ZPHOT_STD
should be a unit Gaussian if ZPHOT_STD were the photo-z scatter. In magnitude bins, the robust
width of x (1.4826 times its median absolute deviation, which ignores the outliers) is the factor
``err_scale`` of :class:`rema.config.PhotozConfig`; the outlier fraction and the median offset are
reported alongside. The DR11 photo-z are cross-validated (each galaxy's photo-z comes from a
forest not trained on it), so the spectroscopic galaxies give a fair estimate, for galaxies like
them.
"""

from __future__ import annotations

import numpy as np

from ..model.photoz import photoz_valid


def nmad(x) -> float:
    """1.4826 median |x - median(x)| (nan for an empty array)."""
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def fit_err_scale(zphot, zphot_std, zspec, refmag, edges=(14.0, 19.0, 20.0, 21.0, 22.0, 24.0),
                  outlier: float = 0.15, min_count: int = 30) -> dict:
    """Per magnitude bin: the node (median magnitude), the scale (NMAD of x), the NMAD of
    dz/(1+z), the outlier fraction (|dz|/(1+z) > ``outlier``), the median dz/(1+z) and the count.

    Bins with fewer than ``min_count`` galaxies are dropped. The returned ``err_scale`` and
    ``err_scale_mag`` are the PhotozConfig values.

    >>> rng = np.random.default_rng(1)
    >>> m = rng.uniform(18, 22, 20000)
    >>> zs = rng.uniform(0.1, 0.8, m.size)
    >>> sig = 0.02 * (1 + zs)
    >>> out = fit_err_scale(zs + sig * rng.normal(size=m.size), 1.5 * sig, zs, m, edges=(18, 20, 22))
    >>> [abs(s - 1 / 1.5) < 0.02 for s in out["err_scale"]]
    [True, True]
    """
    zphot, zphot_std, zspec, refmag = (np.asarray(a, np.float64) for a in (zphot, zphot_std, zspec, refmag))
    ok = photoz_valid(zphot, zphot_std) & np.isfinite(zspec) & (zspec > 0) & np.isfinite(refmag)
    x = (zphot - zspec) / np.where(ok, zphot_std, 1.0)
    dz = (zphot - zspec) / (1.0 + zspec)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = ok & (refmag >= lo) & (refmag < hi)
        if b.sum() < min_count:
            continue
        rows.append(dict(mag=float(np.median(refmag[b])), lo=float(lo), hi=float(hi), n=int(b.sum()),
                         scale=nmad(x[b]), nmad_dz=nmad(dz[b]),
                         outliers=float(np.mean(np.abs(dz[b]) > outlier)), bias=float(np.median(dz[b]))))
    return {"bins": rows, "err_scale": [r["scale"] for r in rows],
            "err_scale_mag": [r["mag"] for r in rows]}


def yaml_overlay(fit: dict, digits: int = 3) -> str:
    """The ``photoz`` section of a configuration overlay for ``rema ... --set @FILE``.

    >>> print(yaml_overlay({"err_scale": [0.85, 0.6], "err_scale_mag": [18.3, 21.4]}), end="")
    photoz:
      err_scale: [0.85, 0.6]
      err_scale_mag: [18.3, 21.4]
    """
    sc = ", ".join(f"{v:.{digits}g}" for v in fit["err_scale"])
    mg = ", ".join(f"{v:.{digits + 1}g}" for v in fit["err_scale_mag"])
    return f"photoz:\n  err_scale: [{sc}]\n  err_scale_mag: [{mg}]\n"
