"""Sky regions: RA/Dec boxes aligned with the Legacy Survey sweep files.

A sweep ``sweep-AAAcBBB-CCCdDDD.fits`` covers AAA <= RA < CCC and cBBB <= Dec < dDDD (degrees,
``m``/``p`` for the sign). A :class:`Box` handles RA wrap-around at 0/360, and a
:class:`BoxUnion` is a set of disjoint boxes (e.g. a set of sweeps that is not a rectangle).

>>> sweep_box("sweep-000m010-005m005.fits")
Box(ra_min=0.0, ra_max=5.0, dec_min=-10.0, dec_max=-5.0)
>>> Box(0, 5, -5, 0).buffered(1.0).contains(359.5, -2.0)
True
>>> Box(0, 5, -88, -85).buffered(2.0)          # the buffer reaches the pole: a full RA ring
Box(ra_min=0.0, ra_max=360.0, dec_min=-90.0, dec_max=-83.0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SWEEP = re.compile(r"sweep-(\d{3})([mp])(\d{3})-(\d{3})([mp])(\d{3})")


@dataclass(frozen=True)
class Box:
    """RA/Dec box in degrees; ``ra_max`` may exceed 360 or ``ra_min`` be negative (wrap)."""

    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float

    def __post_init__(self):
        for name in ("ra_min", "ra_max", "dec_min", "dec_max"):
            object.__setattr__(self, name, float(getattr(self, name)))

    @property
    def is_ring(self) -> bool:
        return self.ra_max - self.ra_min >= 360.0

    def contains(self, ra, dec):
        ra = np.asarray(ra, dtype=np.float64)
        dec = np.asarray(dec, dtype=np.float64)
        width = self.ra_max - self.ra_min
        if width >= 360.0:
            in_ra = np.ones(np.shape(ra), dtype=bool)
        else:
            in_ra = np.mod(ra - self.ra_min, 360.0) < width
        return in_ra & (dec >= self.dec_min) & (dec < self.dec_max)

    def buffered(self, deg: float) -> "Box":
        """The box grown so that it holds every point within ``deg`` degrees of this one.

        Dec grows by ``deg``. RA grows by the exact half-width of a cap of radius ``deg`` at the
        box's largest |Dec|, asin(sin deg / cos |Dec|max). A box whose buffer reaches a pole (or
        would wrap all the way round) becomes a full RA ring.
        """
        dec_lo, dec_hi = self.dec_min - deg, self.dec_max + deg
        ring = Box(0.0, 360.0, max(-90.0, dec_lo), min(90.0, dec_hi))
        if self.is_ring or dec_lo <= -90.0 or dec_hi >= 90.0:
            return ring
        s = np.sin(np.radians(deg)) / np.cos(np.radians(max(abs(self.dec_min), abs(self.dec_max))))
        if s >= 1.0:
            return ring
        dra = float(np.degrees(np.arcsin(s)))
        if self.ra_max - self.ra_min + 2.0 * dra >= 360.0:
            return ring
        return Box(self.ra_min - dra, self.ra_max + dra, dec_lo, dec_hi)

    def intersection(self, other: "Box") -> list["Box"]:
        """Intersection with ``other`` as 0, 1 or 2 boxes (two when RA ranges meet at both ends).

        >>> Box(350, 370, 0, 5).intersection(Box(5, 355, -1, 1))
        [Box(ra_min=350.0, ra_max=355.0, dec_min=0.0, dec_max=1.0), Box(ra_min=5.0, ra_max=10.0, dec_min=0.0, dec_max=1.0)]
        """
        d0, d1 = max(self.dec_min, other.dec_min), min(self.dec_max, other.dec_max)
        if d1 <= d0:
            return []
        if self.is_ring and other.is_ring:
            return [Box(0.0, 360.0, d0, d1)]
        if self.is_ring or other.is_ring:
            b = other if self.is_ring else self
            return [Box(b.ra_min, b.ra_max, d0, d1)]
        a0, a1 = self.ra_min % 360.0, self.ra_min % 360.0 + (self.ra_max - self.ra_min)
        out = []
        for k in (-1, 0, 1):
            b0 = other.ra_min % 360.0 + 360.0 * k
            b1 = b0 + (other.ra_max - other.ra_min)
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                shift = 360.0 * np.floor(lo / 360.0)
                out.append(Box(lo - shift, hi - shift, d0, d1))
        return out

    def union(self, other: "Box") -> "Box":
        """Bounding box of two boxes (see :func:`bounding_box` for RA wrap)."""
        return bounding_box([self, other])

    def overlaps(self, other) -> bool:
        if isinstance(other, BoxUnion):
            return other.overlaps(self)
        if self.dec_max <= other.dec_min or other.dec_max <= self.dec_min:
            return False
        a0, aw = self.ra_min % 360.0, self.ra_max - self.ra_min
        b0, bw = other.ra_min % 360.0, other.ra_max - other.ra_min
        if aw >= 360 or bw >= 360:
            return True
        d = (b0 - a0) % 360.0
        return d < aw or (360.0 - d) < bw

    def area_deg2(self) -> float:
        dra = np.radians(min(360.0, self.ra_max - self.ra_min))
        return float(np.degrees(np.degrees(dra * (np.sin(np.radians(self.dec_max))
                                                   - np.sin(np.radians(self.dec_min))))))

    def bounding(self) -> "Box":
        return self

    @property
    def boxes(self) -> tuple["Box", ...]:
        return (self,)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.ra_min, self.ra_max, self.dec_min, self.dec_max)


@dataclass(frozen=True)
class BoxUnion:
    """A union of disjoint boxes (for example a set of sweeps that is not a rectangle)."""

    boxes: tuple[Box, ...]

    def __post_init__(self):
        object.__setattr__(self, "boxes", tuple(self.boxes))
        if not self.boxes:
            raise ValueError("BoxUnion needs at least one box")

    def contains(self, ra, dec):
        out = self.boxes[0].contains(ra, dec)
        for b in self.boxes[1:]:
            out = out | b.contains(ra, dec)
        return out

    def overlaps(self, other) -> bool:
        return any(o.overlaps(b) for b in self.boxes for o in other.boxes)

    def intersect(self, box: Box) -> "BoxUnion | Box | None":
        """Intersection with a box: a Box, a BoxUnion, or None when empty."""
        return _union_or_box([p for b in self.boxes for p in b.intersection(box)])

    def area_deg2(self) -> float:
        """Sum of the areas (the boxes are disjoint)."""
        return float(sum(b.area_deg2() for b in self.boxes))

    def bounding(self) -> Box:
        return bounding_box(list(self.boxes))

    @property
    def is_ring(self) -> bool:
        return False


def _union_or_box(pieces: list[Box]):
    if not pieces:
        return None
    return pieces[0] if len(pieces) == 1 else BoxUnion(tuple(pieces))


def intersect(a, b):
    """Intersection of two boxes or unions: a Box, a BoxUnion, or None when empty."""
    return _union_or_box([p for x in a.boxes for y in b.boxes for p in x.intersection(y)])


def sweep_box(name: str | Path) -> Box:
    """Sky box covered by a sweep (or photo-z sweep, or per-sweep galaxy table) file name."""
    m = _SWEEP.search(Path(name).name)
    if m is None:
        raise ValueError(f"not a sweep file name: {name}")
    ra0, s0, d0, ra1, s1, d1 = m.groups()
    sign = {"m": -1.0, "p": 1.0}
    return Box(float(ra0), float(ra1), sign[s0] * float(d0), sign[s1] * float(d1))


def sweep_union(names) -> Box | BoxUnion:
    """Sky covered by a set of sweeps: one Box when they tile a rectangle, else a BoxUnion.

    >>> sweep_union(["sweep-000m005-005p000.fits", "sweep-000m010-005m005.fits"])
    Box(ra_min=0.0, ra_max=5.0, dec_min=-10.0, dec_max=0.0)
    """
    boxes = sorted({sweep_box(n) for n in names}, key=lambda b: (b.dec_min, b.ra_min))
    if not boxes:
        raise ValueError("no sweeps")
    bb = bounding_box(boxes)
    if abs(sum(b.area_deg2() for b in boxes) - bb.area_deg2()) < 1e-6 * bb.area_deg2():
        return bb
    return BoxUnion(tuple(boxes))


def sweeps_overlapping(box, sweep_dir: str | Path, pattern: str = "sweep-*.fits") -> list[Path]:
    """Files in ``sweep_dir`` (sweeps, or per-sweep galaxy tables) whose box overlaps ``box``."""
    out = []
    for p in sorted(Path(sweep_dir).glob(pattern)):
        if p.name.endswith("-pz.fits"):
            continue
        try:
            if box.overlaps(sweep_box(p)):
                out.append(p)
        except ValueError:
            continue
    return out


def bounding_box(boxes: list[Box]) -> Box:
    """Smallest RA/Dec box holding all the boxes, with RA wrap: the box starts after the largest
    RA gap between them (so sweeps at RA 355-360 and 0-5 give RA 355-365, not 0-360).

    >>> bounding_box([Box(355, 360, 0, 5), Box(0, 5, 0, 5)])
    Box(ra_min=355.0, ra_max=365.0, dec_min=0.0, dec_max=5.0)
    """
    d0 = min(b.dec_min for b in boxes)
    d1 = max(b.dec_max for b in boxes)
    ring = Box(0.0, 360.0, d0, d1)
    if any(b.is_ring for b in boxes):
        return ring
    merged = []                                  # RA intervals merged along the line
    for lo, hi in sorted((b.ra_min % 360.0, b.ra_min % 360.0 + (b.ra_max - b.ra_min))
                         for b in boxes):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    n = len(merged)
    # Gap after interval i, up to the start of the next one round the circle.
    gaps = [merged[(i + 1) % n][0] + (360.0 if i == n - 1 else 0.0) - merged[i][1]
            for i in range(n)]
    i = int(np.argmax(gaps))
    if gaps[i] <= 0.0:
        return ring
    start = merged[(i + 1) % n][0]
    end = merged[i][1] + (360.0 if i < n - 1 else 0.0)
    if start >= 360.0:
        start, end = start - 360.0, end - 360.0
    return Box(start, end, d0, d1)


def sky_header(sky) -> dict:
    """FITS header keys describing a Box (BOXRA0, BOXRA1, BOXDEC0, BOXDEC1) or a BoxUnion
    (NBOX and B01RA0, B01RA1, B01DEC0, B01DEC1, ...)."""
    if sky is None:
        return {}
    if isinstance(sky, Box):
        return {"BOXRA0": sky.ra_min, "BOXRA1": sky.ra_max, "BOXDEC0": sky.dec_min,
                "BOXDEC1": sky.dec_max}
    out = {"NBOX": len(sky.boxes)}
    for i, b in enumerate(sky.boxes, 1):
        out.update({f"B{i:02d}RA0": b.ra_min, f"B{i:02d}RA1": b.ra_max,
                    f"B{i:02d}DEC0": b.dec_min, f"B{i:02d}DEC1": b.dec_max})
    return out


def sky_from_header(hdr) -> Box | BoxUnion | None:
    """Inverse of :func:`sky_header` (None when the header describes no sky)."""
    if "NBOX" in hdr:
        return BoxUnion(tuple(Box(float(hdr[f"B{i:02d}RA0"]), float(hdr[f"B{i:02d}RA1"]),
                                  float(hdr[f"B{i:02d}DEC0"]), float(hdr[f"B{i:02d}DEC1"]))
                              for i in range(1, int(hdr["NBOX"]) + 1)))
    if "BOXRA0" in hdr:
        return Box(float(hdr["BOXRA0"]), float(hdr["BOXRA1"]), float(hdr["BOXDEC0"]),
                   float(hdr["BOXDEC1"]))
    return None


def sky_union(boxes) -> Box | BoxUnion:
    """One Box, or a BoxUnion of several (for repeated ``--box`` options)."""
    boxes = list(boxes)
    return boxes[0] if len(boxes) == 1 else BoxUnion(tuple(boxes))
