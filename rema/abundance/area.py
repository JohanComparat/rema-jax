"""Area of the survey against redshift: the depth (z_vlim) map.

As in Kluge et al. (2024, Fig. B.3), z_vlim of a HEALPix pixel is the redshift at which a galaxy
of the richness luminosity limit (0.2 L*, m*(z) + 1.75 in the reference band) reaches the 10 sigma
galaxy depth: the catalogue is volume limited there below z_vlim. The area available to a
redshift bin is that of the pixels whose z_vlim is above the bin's upper edge; the clusters
counted in it are those in such pixels (their own ZVLIM above the edge).

:class:`ZvlimMap` holds a sparse map (pixels, area in deg2, ZVLIM and other columns), read from
the reduced products of ``docs/figures/redmapper_dr11/prepare.py`` (a directory of ``.npy``
columns) or from a FITS table.

>>> import numpy as np
>>> zm = ZvlimMap(nside=256, pix=np.arange(4), area=np.full(4, 0.05), zvlim=np.array([0.3, 0.5, 0.7, 0.9]))
>>> round(zm.area_above(0.6), 3), list(np.round(zm.area_above([0.2, 0.8]), 3))
(0.1, [0.2, 0.05])
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def zvlim_of(depth10, mstar, lval: float = 0.2):
    """Redshift where m*(z) - 2.5 log10(lval) reaches the depth ``depth10`` (NaN when the depth is
    brighter than the limit at the lowest redshift of the m* table, its highest redshift beyond)."""
    from ..model.profiles import maxmag_from_mstar

    z = np.asarray(mstar.z, np.float64)
    mlim = np.asarray(maxmag_from_mstar(np.asarray(mstar.m), lval), np.float64)
    return np.interp(np.asarray(depth10, np.float64), mlim, z, left=np.nan, right=z[-1])


@dataclass
class ZvlimMap:
    """Sparse HEALPix (NESTED) map: ``pix``, ``area`` [deg2], ``zvlim`` and ``extra`` columns."""

    nside: int
    pix: np.ndarray
    area: np.ndarray
    zvlim: np.ndarray
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_columns(cls, directory, nside: int | None = None) -> "ZvlimMap":
        """The ``maps`` directory of ``.npy`` columns (PIX, AREA, ZVLIM, ...); NSIDE from the
        ``meta.json`` one level up when not given."""
        d = Path(directory)
        if nside is None:
            meta = d.parent / "meta.json"
            nside = int(json.loads(meta.read_text())["nside"]) if meta.exists() else 256
        cols = {p.stem: np.load(p) for p in sorted(d.glob("*.npy"))}
        return cls(nside, cols.pop("PIX").astype(np.int64), cols.pop("AREA").astype(np.float64),
                   cols.pop("ZVLIM").astype(np.float64), cols)

    @classmethod
    def read(cls, path) -> "ZvlimMap":
        from astropy.io import fits

        from ..io.tables import read_table

        t = read_table(path)
        return cls(int(fits.getheader(path, 1)["NSIDE"]), t.pop("PIX").astype(np.int64),
                   t.pop("AREA").astype(np.float64), t.pop("ZVLIM").astype(np.float64), t)

    def write(self, path):
        from ..io.tables import write_table

        return write_table(path, {"PIX": self.pix, "AREA": self.area, "ZVLIM": self.zvlim, **self.extra},
                           header={"NSIDE": self.nside, "ORDERING": "NESTED"}, extname="ZVLIM_MAP")

    def select(self, keep) -> "ZvlimMap":
        """The pixels where ``keep`` (a boolean array over the pixels) is true."""
        keep = np.asarray(keep, bool)
        return ZvlimMap(self.nside, self.pix[keep], self.area[keep], self.zvlim[keep],
                        {k: np.asarray(v)[keep] for k, v in self.extra.items()})

    def area_above(self, z, mask=None):
        """Area [deg2] where z_vlim >= z (scalar or array), within the pixels ``mask``."""
        a = self.area if mask is None else np.where(mask, self.area, 0.0)
        zv = np.where(np.isfinite(self.zvlim), self.zvlim, -np.inf)
        order = np.argsort(zv)
        cum = np.concatenate([np.cumsum(a[order][::-1])[::-1], [0.0]])
        out = cum[np.searchsorted(zv[order], np.asarray(z, np.float64), side="left")]
        return float(out) if np.ndim(out) == 0 else out

    def index_of(self, ra, dec) -> np.ndarray:
        """Index into the map of the pixel of each position (-1 outside)."""
        import healpy as hp

        p = hp.ang2pix(self.nside, np.asarray(ra), np.asarray(dec), nest=True, lonlat=True)
        order = np.argsort(self.pix)
        k = np.clip(np.searchsorted(self.pix, p, sorter=order), 0, max(self.pix.size - 1, 0))
        i = order[k] if self.pix.size else np.zeros(np.shape(p), np.int64)
        return np.where(self.pix[i] == p, i, -1) if self.pix.size else np.full(np.shape(p), -1)

    def value_at(self, ra, dec, col: str = "ZVLIM") -> np.ndarray:
        """A column of the map at positions (NaN outside)."""
        i = self.index_of(ra, dec)
        v = self.zvlim if col == "ZVLIM" else (self.area if col == "AREA" else np.asarray(self.extra[col]))
        return np.where(i >= 0, v[np.maximum(i, 0)], np.nan)

    def radec(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel centres."""
        import healpy as hp

        return hp.pix2ang(self.nside, self.pix, nest=True, lonlat=True)

    def full_mask(self, values=None) -> np.ndarray:
        """Full-sky NESTED map of ``values`` (default the area fraction of each pixel), zero
        elsewhere, for footprint statistics (e.g. the super-sample variance)."""
        import healpy as hp

        out = np.zeros(hp.nside2npix(self.nside))
        if values is None:
            values = np.clip(self.area / hp.nside2pixarea(self.nside, degrees=True), 0.0, 1.0)
        out[self.pix] = values
        return out
