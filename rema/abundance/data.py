"""The data vector: cluster counts N(lambda_i, z_j) in the volume-limited survey.

A cluster is counted in redshift bin j when its z_lambda is in the bin and the depth at its
position reaches the bin's upper edge (ZVLIM >= z_hi); the area of the bin is that of the pixels
with z_vlim >= z_hi (:meth:`rema.abundance.area.ZvlimMap.area_above`). Clusters outside the
pixels kept (``pix_keep``) or flagged by ``cluster_keep`` (e.g. near a seam) are left out, and so
must their area be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .area import ZvlimMap


@dataclass
class DataVector:
    """Counts [n_lambda, n_z] with the bin edges, the area of each z bin [deg2], and per bin the
    mean lambda, mean z_lambda and median z_lambda error of the clusters counted."""

    lam_edges: np.ndarray
    z_edges: np.ndarray
    counts: np.ndarray
    area: np.ndarray
    lam_mean: np.ndarray
    z_mean: np.ndarray
    sigma_z: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.counts.shape

    def write(self, path):
        from ..io.tables import table_hdu, write_fits

        bins = table_hdu({"COUNTS": self.counts.astype(np.float64), "LAMBDA_MEAN": self.lam_mean,
                          "Z_MEAN": self.z_mean, "SIGMA_Z": self.sigma_z}, extname="COUNTS")
        edges = table_hdu({"LAMBDA_EDGES": self.lam_edges}, extname="LAMBDA_EDGES")
        zed = table_hdu({"Z_EDGES": self.z_edges[:-1], "Z_HI": self.z_edges[1:], "AREA": self.area},
                        extname="Z_BINS")
        return write_fits(path, [bins, edges, zed], {k[:8].upper(): v for k, v in self.meta.items()})

    @classmethod
    def read(cls, path) -> "DataVector":
        from astropy.io import fits

        from ..io.tables import read_table

        c = read_table(path, hdu="COUNTS")
        le = read_table(path, hdu="LAMBDA_EDGES")["LAMBDA_EDGES"]
        zb = read_table(path, hdu="Z_BINS")
        ze = np.concatenate([zb["Z_EDGES"], zb["Z_HI"][-1:]])
        meta = {k: v for k, v in fits.getheader(path).items() if k not in ("SIMPLE", "BITPIX", "NAXIS", "EXTEND")}
        return cls(np.asarray(le, np.float64), np.asarray(ze, np.float64), c["COUNTS"], zb["AREA"],
                   c["LAMBDA_MEAN"], c["Z_MEAN"], c["SIGMA_Z"], meta)


def build_data_vector(cat: Mapping[str, np.ndarray], zmap: ZvlimMap, lam_edges: Sequence[float],
                      z_edges: Sequence[float], *, pix_keep=None, cluster_keep=None,
                      zvlim_col: str = "ZVLIM") -> DataVector:
    """Counts of ``cat`` (LAMBDA, Z_LAMBDA, Z_LAMBDA_E, RA, DEC, and ZVLIM or the map's value at
    the position) in the volume-limited sky of ``zmap``, restricted to the pixels ``pix_keep``
    (boolean over the map's pixels) and the clusters ``cluster_keep``."""
    le, ze = np.asarray(lam_edges, np.float64), np.asarray(z_edges, np.float64)
    lam, z = np.asarray(cat["LAMBDA"], np.float64), np.asarray(cat["Z_LAMBDA"], np.float64)
    ze_err = np.asarray(cat.get("Z_LAMBDA_E", np.full(lam.size, np.nan)), np.float64)
    idx = zmap.index_of(cat["RA"], cat["DEC"])
    keep = idx >= 0
    if pix_keep is not None:
        keep &= np.asarray(pix_keep, bool)[np.maximum(idx, 0)]
    if cluster_keep is not None:
        keep &= np.asarray(cluster_keep, bool)
    zv = np.asarray(cat[zvlim_col], np.float64) if zvlim_col in cat else zmap.zvlim[np.maximum(idx, 0)]
    il, iz = np.digitize(lam, le) - 1, np.digitize(z, ze) - 1
    nl, nz = le.size - 1, ze.size - 1
    inside = keep & (il >= 0) & (il < nl) & (iz >= 0) & (iz < nz)
    deep = inside & (zv >= ze[np.clip(iz + 1, 0, nz)])
    counts = np.zeros((nl, nz))
    lmean, zmean, sz = (np.full((nl, nz), np.nan) for _ in range(3))
    for i in range(nl):
        for j in range(nz):
            s = deep & (il == i) & (iz == j)
            counts[i, j] = s.sum()
            if s.any():
                lmean[i, j], zmean[i, j] = lam[s].mean(), z[s].mean()
                sz[i, j] = np.nanmedian(ze_err[s]) if np.isfinite(ze_err[s]).any() else np.nan
    area = zmap.area_above(ze[1:], mask=pix_keep)
    return DataVector(le, ze, counts, np.atleast_1d(area), lmean, zmean, sz,
                      {"NCLUSTER": int(deep.sum()), "AREATOT": float(np.sum(zmap.area if pix_keep is None
                                                                             else zmap.area[np.asarray(pix_keep, bool)]))})
