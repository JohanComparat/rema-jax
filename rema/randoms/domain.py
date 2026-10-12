"""The area a cluster catalogue covers: the own boxes of the regions that finished.

A merged run (``rema merge``) keeps the clusters centred in the own box of their region, of the
regions that finished, at |b| >= ``glat_min``; the galaxies and the randoms carry the per-object
cuts (MASKBITS, NOBS, E(B-V)). :class:`RunDomain` holds those boxes for one or several runs
(the parts of a combined catalogue) and tells which positions belong to it.

Where boxes of different parts touch, each part found its clusters independently (a cluster on
the line can be found twice, see FLAG_SEAM of the combined catalogue). These seams are found
from the boxes, and ``seam_buffer_deg`` removes the area (and the clusters) closer to them.

>>> d = RunDomain(np.array([[0.0, 10.0, -5.0, 5.0], [10.0, 20.0, -5.0, 5.0]]),
...               parts=np.array([0, 1]), regions=np.array([0, 0]), glat_min=0.0)
>>> d.contains(np.array([5.0, 15.0, 25.0]), np.array([0.0, 0.0, 0.0])).tolist()
[True, True, False]
>>> d.part_of(np.array([5.0, 15.0]), np.array([0.0, 0.0])).tolist()
[0, 1]
>>> np.round(d.seam_distance(np.array([9.0]), np.array([0.0])), 6).tolist()   # seam at RA 10
[1.0]
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io.tables import read_header, read_table, table_hdu, write_fits

_EPS = 1e-9


def _in_ra(ra, r0, r1):
    return ((ra - r0) % 360.0) < (r1 - r0)


def _gc_dist(ra1, dec1, ra2, dec2):
    """Great-circle distance [deg]."""
    r1, d1, r2, d2 = (np.radians(np.asarray(a, np.float64)) for a in (ra1, dec1, ra2, dec2))
    s = np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(s, 0.0, 1.0))))


def _seg_dist(ra, dec, seg):
    """Distance [deg] of positions to a seam segment (kind, value, lo, hi): kind 0 is the meridian
    RA = value for lo <= Dec <= hi, kind 1 the parallel Dec = value for RA in [lo, lo + (hi - lo))."""
    kind, v, lo, hi = seg
    if kind == 0:
        dra = np.radians((ra - v + 180.0) % 360.0 - 180.0)
        along = np.degrees(np.arcsin(np.minimum(1.0, np.abs(np.sin(dra)) * np.cos(np.radians(dec)))))
        inside = (dec >= lo) & (dec <= hi) & (np.abs(dra) <= np.pi / 2)
        ends = np.minimum(_gc_dist(ra, dec, v, lo), _gc_dist(ra, dec, v, hi))
        return np.where(inside, along, ends)
    inside = _in_ra(ra, lo, hi) | (np.abs(((ra - hi) + 180.0) % 360.0 - 180.0) < _EPS)
    ends = np.minimum(_gc_dist(ra, dec, lo, v), _gc_dist(ra, dec, hi, v))
    return np.where(inside, np.abs(dec - v), ends)


def _seams(boxes, parts):
    """Segments where boxes of different parts share an edge."""
    out = []
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if parts[i] >= parts[j]:
                continue
            a, b = boxes[i], boxes[j]
            for ea, eb in ((a[1], b[0]), (a[0], b[1])):            # RA edges (mod 360)
                if abs(((ea - eb) + 180.0) % 360.0 - 180.0) < _EPS:
                    lo, hi = max(a[2], b[2]), min(a[3], b[3])
                    if hi > lo:
                        out.append((0, ea % 360.0, lo, hi))
            for ea, eb in ((a[3], b[2]), (a[2], b[3])):            # Dec edges
                if abs(ea - eb) < _EPS:
                    # RA overlap: it starts at whichever box start lies inside the other box
                    for lo in (a[0], b[0]):
                        if _in_ra(np.array([lo]), a[0], a[1])[0] and _in_ra(np.array([lo]), b[0], b[1])[0]:
                            width = min(a[0] + (a[1] - a[0]) - lo if lo == a[0] else (a[1] - lo) % 360.0,
                                        b[0] + (b[1] - b[0]) - lo if lo == b[0] else (b[1] - lo) % 360.0)
                            out.append((1, ea, lo % 360.0, lo % 360.0 + width))
                            break
    return out


@dataclass
class RunDomain:
    """Own boxes [N, 4] (RA0, RA1, DEC0, DEC1; RA0 < RA1, RA taken modulo 360) with their part
    and region, the |b| cut and the seam buffer."""

    boxes: np.ndarray
    parts: np.ndarray
    regions: np.ndarray
    glat_min: float = 15.0
    seam_buffer_deg: float = 0.0

    def __post_init__(self):
        self.boxes = np.asarray(self.boxes, np.float64).reshape(-1, 4)
        self.parts = np.asarray(self.parts, np.int16)
        self.regions = np.asarray(self.regions, np.int32)
        self.seams = _seams(self.boxes, self.parts)

    @classmethod
    def from_runs(cls, runs, glat_min: float | None = None, seam_buffer_deg: float = 0.0) -> "RunDomain":
        """Domain of merged runs (directories with ``regions.fits`` and
        ``clusters_dr11_qa.json``; part k = the k-th run). ``glat_min``: default the runs' own."""
        from ..pipeline import read_plan

        boxes, parts, regions, glats = [], [], [], []
        for k, run in enumerate(runs):
            run = Path(run)
            plan, _ = read_plan(run / "regions.fits")
            qa = json.loads(next(run.glob("*_qa.json")).read_text())
            done = ~np.isin(plan["REGION_ID"], qa.get("missing", []))
            boxes.append(np.stack([plan[f"OWN_{a}"][done] for a in ("RA0", "RA1", "DEC0", "DEC1")], axis=1))
            parts.append(np.full(done.sum(), k))
            regions.append(plan["REGION_ID"][done])
            if qa.get("glat_min") is not None:
                glats.append(float(qa["glat_min"]))
        if glat_min is None:
            glat_min = max(glats) if glats else 0.0
        return cls(np.concatenate(boxes), np.concatenate(parts), np.concatenate(regions),
                   glat_min=float(glat_min), seam_buffer_deg=float(seam_buffer_deg))

    # ------------------------------------------------------------------ membership
    def box_of(self, ra, dec) -> np.ndarray:
        """Index of the box containing each position (-1 for none)."""
        ra = np.asarray(ra, np.float64) % 360.0
        dec = np.asarray(dec, np.float64)
        out = np.full(ra.shape, -1, np.int64)
        for i, (r0, r1, d0, d1) in enumerate(self.boxes):
            inside = _in_ra(ra, r0, r1) & (dec >= d0) & (dec < d1)
            out[inside & (out < 0)] = i
        return out

    def part_of(self, ra, dec) -> np.ndarray:
        b = self.box_of(ra, dec)
        return np.where(b >= 0, self.parts[np.maximum(b, 0)], -1).astype(np.int16)

    def seam_distance(self, ra, dec) -> np.ndarray:
        """Distance [deg] to the nearest seam (inf without seams)."""
        ra, dec = np.asarray(ra, np.float64) % 360.0, np.asarray(dec, np.float64)
        d = np.full(ra.shape, np.inf)
        for s in self.seams:
            d = np.minimum(d, _seg_dist(ra, dec, s))
        return d

    def contains(self, ra, dec) -> np.ndarray:
        """In a box, at |b| >= glat_min and farther than the seam buffer from the seams.

        Positions in a pixel (NSIDE 256) whose centre, corners and edge midpoints all agree take
        that answer; the others are tested exactly.
        """
        from ..sky.healpix import ang2pix_nest_np

        ra, dec = np.asarray(ra, np.float64), np.asarray(dec, np.float64)
        if ra.size < 100_000:
            return self.contains_exact(ra, dec)
        cls = self._coarse_classes()
        c = cls[ang2pix_nest_np(self._COARSE_NSIDE, ra, dec)]
        out = c == 1
        edge = np.flatnonzero(c == 2)
        if edge.size:
            out[edge] = self.contains_exact(ra[edge], dec[edge])
        return out

    _COARSE_NSIDE = 256

    def _coarse_classes(self) -> np.ndarray:
        """Per NSIDE-256 pixel: 0 outside, 1 inside, 2 undecided."""
        if getattr(self, "_cls", None) is None:
            import healpy as hp

            ns = self._COARSE_NSIDE
            pix = np.arange(12 * ns**2)
            ra, dec = hp.pix2ang(ns, pix, nest=True, lonlat=True)
            c = self.contains_exact(ra, dec)
            vec = hp.boundaries(ns, pix, step=2, nest=True)              # [n, 3, 8]
            bra, bdec = hp.vec2ang(np.moveaxis(vec, 1, 2).reshape(-1, 3), lonlat=True)
            b = self.contains_exact(bra, bdec).reshape(pix.size, -1)
            agree = np.all(b == c[:, None], axis=1)
            if self.seam_buffer_deg > 0:      # a buffer band can be narrower than a pixel
                rad = float(np.degrees(hp.max_pixrad(ns)))
                agree &= self.seam_distance(ra, dec) > self.seam_buffer_deg + 1.01 * rad
            self._cls = np.where(agree, c.astype(np.int8), np.int8(2))
        return self._cls

    def contains_exact(self, ra, dec) -> np.ndarray:
        from ..pipeline import galactic_latitude

        ra, dec = np.asarray(ra, np.float64), np.asarray(dec, np.float64)
        ok = self.box_of(ra, dec) >= 0
        if self.glat_min > 0 and ok.any():
            ok[ok] = np.abs(galactic_latitude(ra[ok], dec[ok])) >= self.glat_min
        if self.seam_buffer_deg > 0 and ok.any():
            ok[ok] = self.seam_distance(ra[ok], dec[ok]) >= self.seam_buffer_deg
        return ok

    # ------------------------------------------------------------------ io
    def digest(self) -> str:
        h = hashlib.sha1(np.ascontiguousarray(self.boxes).tobytes())
        h.update(np.ascontiguousarray(self.parts).tobytes())
        h.update(f"{self.glat_min} {self.seam_buffer_deg}".encode())
        return h.hexdigest()[:12]

    def write(self, path) -> Path:
        cols = {"PART": self.parts, "REGION_ID": self.regions,
                **{k: self.boxes[:, i] for i, k in enumerate(("RA0", "RA1", "DEC0", "DEC1"))}}
        hdr = {"GLATMIN": self.glat_min, "SEAMBUF": self.seam_buffer_deg, "DIGEST": self.digest()}
        return write_fits(path, [table_hdu(cols, hdr, "BOXES")])

    @classmethod
    def read(cls, path) -> "RunDomain":
        t = read_table(path, hdu="BOXES")
        h = read_header(path, "BOXES")
        boxes = np.stack([t[k] for k in ("RA0", "RA1", "DEC0", "DEC1")], axis=1)
        return cls(boxes, t["PART"], t["REGION_ID"], float(h["GLATMIN"]), float(h["SEAMBUF"]))
