"""Neighbour search on the sphere and padding into fixed-shape batches.

The galaxies of a region go into a KD-tree of unit vectors (scipy). A query returns, for each
centre, the galaxies within an angular radius; the ragged lists are padded to K, the next power of
two of the largest list (at least ``kmin``), so each K compiles once. Angular separations are
computed in float64 on the host and passed to the device in float32.

>>> import numpy as np
>>> idx = NeighborIndex(np.array([10.0, 10.01, 10.5]), np.array([0.0, 0.0, 0.0]))
>>> pad = idx.query(np.array([10.0]), np.array([0.0]), 0.1)
>>> pad.idx.shape, int(pad.valid.sum()), np.round(pad.theta[0, :2], 3).tolist()
((1, 128), 2, [0.0, 0.01])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def unit_vectors(ra, dec) -> np.ndarray:
    ra = np.radians(np.asarray(ra, np.float64))
    dec = np.radians(np.asarray(dec, np.float64))
    c = np.cos(dec)
    return np.stack([c * np.cos(ra), c * np.sin(ra), np.sin(dec)], axis=-1)


def radec_from_unit(xyz) -> tuple[float, float]:
    """(RA, Dec) in degrees of a unit vector (any norm > 0)."""
    x, y, z = (float(v) for v in np.asarray(xyz, np.float64))
    return float(np.degrees(np.arctan2(y, x)) % 360.0), float(np.degrees(np.arctan2(z, np.hypot(x, y))))


def n_workers() -> int:
    """CPUs this process may use (the SLURM allocation, not the whole node)."""
    import os

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:                       # not Linux
        return os.cpu_count() or 1


def next_pow2(n: int, kmin: int = 128) -> int:
    return max(kmin, 1 << int(np.ceil(np.log2(max(n, 1)))))


@dataclass
class Padded:
    """Padded neighbour lists: ``idx`` [B, K] (0 where invalid), ``valid``, ``theta`` [deg]."""

    idx: np.ndarray
    valid: np.ndarray
    theta: np.ndarray
    counts: np.ndarray


class NeighborIndex:
    def __init__(self, ra, dec):
        self.xyz = unit_vectors(ra, dec)
        self.tree = cKDTree(self.xyz)
        self.n = self.xyz.shape[0]

    def query_lists(self, ra_c, dec_c, radius_deg) -> list[np.ndarray]:
        xc = unit_vectors(ra_c, dec_c)
        rad = np.broadcast_to(np.asarray(radius_deg, np.float64), (xc.shape[0],))
        chord = 2.0 * np.sin(np.radians(rad) / 2.0)
        return list(self.tree.query_ball_point(xc, chord, workers=n_workers(), return_sorted=False))

    def query(self, ra_c, dec_c, radius_deg, k: int | None = None, kmin: int = 128) -> Padded:
        """Neighbours within ``radius_deg`` (scalar or per centre), padded to K.

        If ``k`` is given and a list is longer, the farthest neighbours are dropped.
        """
        xc = unit_vectors(ra_c, dec_c)
        lists = self.query_lists(ra_c, dec_c, radius_deg)
        counts = np.array([len(l) for l in lists], dtype=np.int64)
        K = next_pow2(int(counts.max(initial=1)), kmin) if k is None else int(k)
        B = len(lists)
        idx = np.zeros((B, K), np.int64)
        valid = np.zeros((B, K), bool)
        theta = np.zeros((B, K), np.float64)
        for b, l in enumerate(lists):
            if not l:
                continue
            l = np.asarray(l, np.int64)
            d = np.linalg.norm(self.xyz[l] - xc[b], axis=1)
            th = np.degrees(2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0)))
            order = np.argsort(th)[:K]
            n = order.size
            idx[b, :n] = l[order]
            valid[b, :n] = True
            theta[b, :n] = th[order]
        return Padded(idx, valid, theta, np.minimum(counts, K))


def bucket_order(counts: np.ndarray, kmin: int = 128) -> list[tuple[int, np.ndarray]]:
    """Group items by the power-of-two bucket of their neighbour count: [(K, indices), ...]."""
    ks = np.array([next_pow2(int(c), kmin) for c in counts])
    return [(int(k), np.flatnonzero(ks == k)) for k in np.unique(ks)]
