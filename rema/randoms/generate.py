"""Copies of clusters at random positions (redMaPPer's GenerateRandoms).

The positions are DR11 random points (the survey randoms) that pass the galaxies' cuts
(:func:`rema.sky.maps.good_randoms`: MASKBITS, NOBS, E(B-V)), lie in the run's domain and, for
an externally selected sample, in the external survey's footprint. Each carries the z_vlim of
its pixel. A copy of a cluster at redshift z goes to a point drawn uniformly among those with
z_vlim > z, the set D(z) where the cluster would be in the volume-limited catalogue: as
redMaPPer, which draws positions uniformly over the zmax mask (weighted by the good fraction)
and keeps one only where z < zmax. Random points are uniform over the unmasked area, so the
fraction-good weighting and the sub-pixel masks (stars, bleed trails) come with them.

The pool is sorted by z_vlim, so D(z) is a tail of it and a copy is one uniform index:
``lo = searchsorted(zvlim, z, 'right')``, ``i = lo + floor(u (N - lo))``. Points are drawn with
replacement; the pool should be much larger than the number of copies (DR11: 2,500 points per
deg2 and file).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..abundance.area import ZvlimMap
from ..sky.maps import good_randoms
from .domain import RunDomain

FLAG_NOAREA = 1            # no pool point with z_vlim > z: the cluster cannot be copied


def iter_survey_randoms(sources, cfg, chunk: int = 5_000_000):
    """Chunks (dicts of the footprint columns) of DR11 randoms ``sources``: FITS files, or the
    stems of a randoms index given as ``index_dir/stem`` (see :func:`rema.sky.maps.index_randoms`)."""
    from astropy.io import fits

    from ..sky.maps import _randoms_columns

    cols = _randoms_columns(cfg)
    for src in sources:
        src = Path(src)
        npy = src.with_name(src.name + ".npy") if not src.suffix == ".fits" else None
        if npy is not None and npy.exists():
            arr = np.load(npy, mmap_mode="r")
            for lo in range(0, arr.shape[0], chunk):
                rows = np.asarray(arr[lo:lo + chunk])
                yield {c: np.ascontiguousarray(rows[c]) for c in cols}
            continue
        with fits.open(src, memmap=True) as h:
            data = h[1].data
            for lo in range(0, len(data), chunk):
                sl = slice(lo, min(len(data), lo + chunk))
                yield {c: np.asarray(data[c][sl]).astype(data[c].dtype.newbyteorder("=")) for c in cols}


@dataclass
class PositionPool:
    """Random points sorted by ZVLIM (ascending), with their RA, DEC."""

    ra: np.ndarray
    dec: np.ndarray
    zvlim: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        o = np.argsort(self.zvlim, kind="stable")
        self.ra = np.asarray(self.ra, np.float64)[o]
        self.dec = np.asarray(self.dec, np.float64)[o]
        self.zvlim = np.asarray(self.zvlim, np.float64)[o]

    @property
    def size(self) -> int:
        return self.ra.size

    @classmethod
    def from_chunks(cls, chunks, cfg, zmap: ZvlimMap, domain: RunDomain | None = None,
                    extra_mask=None, max_points: int | None = None, rng=None) -> "PositionPool":
        """Pool of the random points of ``chunks`` (see :func:`iter_survey_randoms`) that pass
        :func:`good_randoms`, ``domain`` and ``extra_mask(ra, dec) -> bool``, inside the z_vlim map
        (finite ZVLIM). ``max_points``: keep a random subset of at most that many."""
        ra, dec, zv, nread = [], [], [], 0
        for r in chunks:
            nread += r["RA"].size
            ok = good_randoms(r, cfg)
            a, d = np.asarray(r["RA"], np.float64)[ok], np.asarray(r["DEC"], np.float64)[ok]
            if domain is not None:
                k = domain.contains(a, d)
                a, d = a[k], d[k]
            if extra_mask is not None:
                k = np.asarray(extra_mask(a, d), bool)
                a, d = a[k], d[k]
            z = zmap.value_at(a, d)
            k = np.isfinite(z)
            ra.append(a[k]), dec.append(d[k]), zv.append(z[k])
        ra, dec, zv = (np.concatenate(x) if x else np.zeros(0) for x in (ra, dec, zv))
        if max_points is not None and ra.size > max_points:
            rng = rng or np.random.default_rng(0)
            k = np.sort(rng.choice(ra.size, max_points, replace=False))
            ra, dec, zv = ra[k], dec[k], zv[k]
        return cls(ra, dec, zv, {"NREAD": nread, "NPOOL": int(ra.size)})

    def restrict(self, mask) -> "PositionPool":
        """The points where ``mask`` (boolean over the pool, or a function of ra, dec) is true."""
        keep = np.asarray(mask(self.ra, self.dec) if callable(mask) else mask, bool)
        return PositionPool(self.ra[keep], self.dec[keep], self.zvlim[keep],
                            {**self.meta, "NPOOL": int(keep.sum())})

    def n_above(self, z) -> np.ndarray:
        """Number of points with z_vlim > z."""
        return self.size - np.searchsorted(self.zvlim, np.asarray(z, np.float64), side="right")

    def draw(self, z, rng) -> np.ndarray:
        """One pool index per redshift in ``z``, uniform over the points with z_vlim > z
        (-1 where there is none)."""
        z = np.asarray(z, np.float64)
        lo = np.searchsorted(self.zvlim, z, side="right")
        n = self.size - lo
        i = lo + np.floor(rng.random(z.size) * n).astype(np.int64)
        return np.where(n > 0, np.minimum(i, self.size - 1), -1)


def generate(source: dict, pool: PositionPool, n_samp, rng, z_col: str = "Z_LAMBDA",
             carry=()) -> tuple[dict, dict]:
    """``n_samp`` copies (an int or one per source row) of each source cluster.

    ``source``: columns of the clusters. Returns (randoms, sources): randoms have SRC (row in
    ``source``), RA, DEC, ZVLIM and the source columns named in ``carry``; sources gain N_SAMP,
    N_POOL (pool points in D(z)) and FLAG (FLAG_NOAREA).
    """
    z = np.asarray(source[z_col], np.float64)
    n = z.size
    ns = np.broadcast_to(np.asarray(n_samp, np.int64), (n,)).copy()
    npool = pool.n_above(z)
    flag = np.where(npool > 0, 0, FLAG_NOAREA).astype(np.int16)
    ns[flag != 0] = 0
    src = np.repeat(np.arange(n), ns)
    idx = pool.draw(z[src], rng)
    rand = {"SRC": src.astype(np.int64), "RA": pool.ra[idx], "DEC": pool.dec[idx],
            "ZVLIM": pool.zvlim[idx].astype(np.float32)}
    for k in carry:
        rand[k] = np.asarray(source[k])[src]
    sources = {**{k: np.asarray(v) for k, v in source.items()}, "N_SAMP": ns.astype(np.int32),
               "N_POOL": npool.astype(np.int64), "FLAG": flag}
    return rand, sources
