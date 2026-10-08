"""Shared settings of the cosmology sensitivity scripts (docs/cosmology_sensitivity.rst).

Two directories, from environment variables, as in the DR11 notebooks:

REMA_DATA  shared, read only: the DR11 south directory, with the production in ``rema/``
           (``rema_dr11_v0.2.0`` combined, and the two parts). Default: the DR11 data system at
           CC-IN2P3 when /sps is mounted, else ``~/data/legacysurvey/dr11/south``.
REMA_WORK  yours, writable (default ``~/rema_work``): the reduced tables of
           ``docs/figures/redmapper_dr11/prepare.py`` are read from ``$REMA_WORK/redmapper``, and
           these scripts write to ``$REMA_WORK/cosmo_sens``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

CC = Path("/sps/lsst/datasets/desi/legacysurveys/dr11/south")
REMA_DATA = Path(os.environ.get("REMA_DATA", CC if CC.exists() else Path.home() / "data" / "legacysurvey" / "dr11" / "south"))
REMA_WORK = Path(os.environ.get("REMA_WORK", Path.home() / "rema_work"))
PRODUCTS = REMA_DATA / "rema"
REDUCED = REMA_WORK / "redmapper"
OUT = REMA_WORK / "cosmo_sens"
COMBINED = PRODUCTS / "rema_dr11_v0.2.0"
CALIB = PRODUCTS / "rema_dr11_v0.2.0_ra0-240" / "calib" / "calib.fits"

#: Binning of the data vector: lambda >= 20; z_lambda in 0.1-0.6 (volume limited over 96% of the
#: area; the z_lambda spikes at 0.82-0.86 are left out, the one at 0.38 sits inside a bin).
LAM_EDGES = [20.0, 30.0, 45.0, 60.0, 100.0, np.inf]
Z_EDGES = [0.1, 0.2, 0.3, 0.45, 0.6]
#: Clusters (and pixels) closer than this to a seam between the two parts are left out [deg].
SEAM = 2.33
#: z-band 10 sigma depth separating the deep (DES) and shallow (DECaLS) sky [mag].
DEPTH_SPLIT = 22.5


def seam_distance(ra, dec):
    """Angular distance [deg] to the boundaries between the parts: the meridians RA = 0 and 240
    above Dec -85, and the parallel Dec = -85 between RA 0 and 240 (scripts/dr11/combine_dr11.py)."""
    ra, dec = np.radians(np.asarray(ra, np.float64)), np.radians(np.asarray(dec, np.float64))
    d = np.full(ra.shape, np.inf)
    for ra0 in (0.0, 240.0):
        dra = (ra - np.radians(ra0) + np.pi) % (2 * np.pi) - np.pi
        dm = np.degrees(np.arcsin(np.clip(np.cos(dec) * np.abs(np.sin(dra)), 0, 1)))
        dm = np.where(np.abs(dra) < np.pi / 2, dm, np.inf)
        end = np.degrees(np.arccos(np.clip(np.sin(dec) * np.sin(np.radians(-85)) +
                                           np.cos(dec) * np.cos(np.radians(-85)) * np.cos(dra), -1, 1)))
        d = np.minimum(d, np.where(np.degrees(dec) >= -85, dm, end))
    ra_deg = np.degrees(ra) % 360
    dp = np.abs(np.degrees(dec) + 85.0)
    return np.minimum(d, np.where(ra_deg < 240, dp, np.inf))


def load_reduced(columns=("LAMBDA", "Z_LAMBDA", "Z_LAMBDA_E", "RA", "DEC", "ZVLIM", "SEAM_DIST",
                          "DEPTH_Z10", "PART"), lambda_min: float = 20.0):
    """Clusters of the reduced DR11 table (``prepare.py``) with lambda >= lambda_min."""
    d = REDUCED / "clusters"
    if not d.exists():
        raise SystemExit(f"{d} not found: run docs/figures/redmapper_dr11/prepare.py with the same REMA_WORK")
    lam = np.load(d / "LAMBDA.npy", mmap_mode="r")
    keep = np.flatnonzero(np.asarray(lam) >= lambda_min)
    return {c: np.asarray(np.load(d / f"{c}.npy", mmap_mode="r"))[keep] for c in columns}


def load_map():
    from rema.abundance.area import ZvlimMap

    return ZvlimMap.from_columns(REDUCED / "maps")


def data_vector(variant: str = "all", lambda_min: float = 20.0, lam_edges=None, z_edges=None):
    """The DR11 data vector of a variant: "all", "deep" or "shallow" (z-band depth against
    :data:`DEPTH_SPLIT`); seams excluded from the clusters and the area."""
    from rema.abundance.data import build_data_vector

    zmap = load_map()
    cat = load_reduced(lambda_min=lambda_min)
    ra, dec = zmap.radec()
    pix_keep = seam_distance(ra, dec) > SEAM
    depth = np.asarray(zmap.extra["DEPTH_Z10"])
    if variant == "deep":
        pix_keep &= depth >= DEPTH_SPLIT
    elif variant == "shallow":
        pix_keep &= depth < DEPTH_SPLIT
    elif variant != "all":
        raise ValueError(f"unknown variant {variant!r}")
    le = LAM_EDGES if lam_edges is None else lam_edges
    le = [max(lambda_min, le[0])] + [e for e in le[1:] if e > lambda_min]
    dv = build_data_vector(cat, zmap, le, Z_EDGES if z_edges is None else z_edges,
                           pix_keep=pix_keep, cluster_keep=cat["SEAM_DIST"] > SEAM)
    dv.meta.update(VARIANT=variant, LAMMIN=lambda_min)
    return dv, zmap, pix_keep
