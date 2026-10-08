"""Footprint, mask and depth maps built from Legacy Survey randoms.

DR11 randoms are uniform points over the processed bricks (2,500 per deg^2 per file) that carry
NOBS, GALDEPTH and MASKBITS at their exact positions. They give, per NESTED HEALPix pixel of
``nside_fine`` (1024 by default):

- ``NRAND``: number of randoms (any) — zero outside the processed area or outside the data box;
- ``COVER``: fraction of the pixel inside the data box (1 when no box is given);
- ``FRACGOOD``: fraction of the randoms that pass the same MASKBITS / NOBS cuts as the
  galaxies, times ``COVER``;
- ``SIGF_<band>``: median dereddened 1-sigma flux error of a canonical galaxy, from GALDEPTH,
  the inverse variance (1/nanomaggies^2, not corrected for extinction) of its flux:
  sigma_f = 1 / sqrt(GALDEPTH) / MW_TRANSMISSION (the 5-sigma depth is
  22.5 - 2.5 log10(5 / sqrt(GALDEPTH)) mag).

Pixels without randoms are absent (fraction good 0). Depths of good-less pixels fall back to the
parent pixel at ``nside_fine / 4``, then to the global median.

>>> m = SparseMap(16, np.array([3, 7, 9]), {"V": np.array([1.0, 2.0, 3.0])})
>>> m.lookup_np(np.array([0.0]), np.array([90.0]), "V")   # pixel 1 of nside 16 is absent
array([0.])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

from ..config import EXTINCTION_DECAM, RemaConfig
from ..io.tables import read_table, table_hdu, write_fits
from .healpix import ang2pix_nest, ang2pix_nest_np, pix_area_deg2
from .regions import Box, BoxUnion, sky_from_header, sky_header

# SIGF of pixels without any good random anywhere in the input (FRACGOOD = 0 there, so the value
# only has to be finite): a 1-sigma error of 1e3 nanomaggies, i.e. no depth.
SIGF_NONE = 1e3


@dataclass
class SparseMap:
    """NESTED HEALPix map stored as sorted pixel numbers and named value columns."""

    nside: int
    pixels: np.ndarray
    values: dict[str, np.ndarray] = field(default_factory=dict)
    fill: float = 0.0

    def __post_init__(self):
        order = np.argsort(self.pixels)
        self.pixels = np.asarray(self.pixels, dtype=np.int64)[order]
        self.values = {k: np.asarray(v)[order] for k, v in self.values.items()}

    # lookups ---------------------------------------------------------------
    def index_np(self, pix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i = np.clip(np.searchsorted(self.pixels, pix), 0, max(self.pixels.size - 1, 0))
        found = (self.pixels.size > 0) & (self.pixels[i] == pix)
        return i, found

    def lookup_np(self, ra, dec, name: str) -> np.ndarray:
        pix = ang2pix_nest_np(self.nside, ra, dec)
        i, found = self.index_np(pix)
        return np.where(found, self.values[name][i], self.fill)

    def device(self, names: tuple[str, ...]) -> "DeviceMap":
        """Arrays for jitted lookups (pixel numbers fit in int32 for nside <= 8192)."""
        return DeviceMap(jnp.asarray(self.pixels.astype(np.int32)),
                         {n: jnp.asarray(self.values[n]) for n in names}, self.nside, self.fill)

    # io ----------------------------------------------------------------------
    def hdu(self, extname: str, header: dict | None = None):
        cols = {"PIXEL": self.pixels, **self.values}
        hdr = {"NSIDE": self.nside, "ORDERING": "NESTED", "FILLVAL": self.fill, **(header or {})}
        return table_hdu(cols, hdr, extname)

    @classmethod
    def from_hdu(cls, path: str | Path, extname: str) -> "SparseMap":
        from astropy.io import fits

        t = read_table(path, hdu=extname)
        with fits.open(path) as h:
            hdr = h[extname].header
            nside, fill = int(hdr["NSIDE"]), float(hdr.get("FILLVAL", 0.0))
        pix = t.pop("PIXEL")
        return cls(nside, pix, t, fill)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DeviceMap:
    """Jit-friendly view of a :class:`SparseMap` (a pytree; nside and fill are static)."""

    pixels: jnp.ndarray
    values: dict
    nside: int = field(default=1024, metadata=dict(static=True))
    fill: float = field(default=0.0, metadata=dict(static=True))

    def lookup(self, ra, dec, name: str):
        pix = ang2pix_nest(self.nside, ra, dec)
        i = jnp.clip(jnp.searchsorted(self.pixels, pix), 0, self.pixels.size - 1)
        found = self.pixels[i] == pix
        return jnp.where(found, self.values[name][i], self.fill)


def sigma_flux_from_depth(depth_ivar, ebv, band: str):
    """Dereddened 1-sigma flux error (nanomaggies) from a GALDEPTH/PSFDEPTH inverse variance.

    >>> round(float(sigma_flux_from_depth(66.84, 0.0, "z")), 4)   # 5-sigma depth 23.03 mag
    0.1223
    """
    mw = 10.0 ** (-0.4 * EXTINCTION_DECAM[band] * np.asarray(ebv))
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.sqrt(np.asarray(depth_ivar, np.float64)) / mw


def depth_ivar_from_mag(depth5_mag, zeropoint: float = 22.5):
    """GALDEPTH-style inverse variance (1/nanomaggies^2) of a 5-sigma AB depth."""
    f5 = 10.0 ** (-0.4 * (np.asarray(depth5_mag, np.float64) - zeropoint))
    return (5.0 / f5) ** 2


@dataclass
class Footprint:
    """Fine map with NRAND, FRACGOOD and SIGF_<band>; see the module docstring."""

    fine: SparseMap
    bands: tuple[str, ...]
    density: float
    box: Box | BoxUnion | None = None

    @property
    def nside(self) -> int:
        return self.fine.nside

    def area_deg2(self) -> float:
        """Unmasked area: sum of fraction good times pixel area."""
        return float(np.sum(self.fine.values["FRACGOOD"]) * pix_area_deg2(self.nside))

    def effective_area(self, mag, band: str, snr_min: float = 5.0, mag_max: float = np.inf,
                       zeropoint: float = 22.5, sky=None) -> np.ndarray:
        """Area (deg^2) over which a galaxy of true magnitude ``mag`` passes the selection.

        Selection: measured flux above max(snr_min sigma_f, flux(mag_max)), with Gaussian noise.
        ``sky``: count only the pixels centred in this box (or union of boxes), for galaxies read
        over a part of the map's area.
        """
        mag = np.atleast_1d(np.asarray(mag, dtype=np.float64))
        sig = self.fine.values[f"SIGF_{band.upper()}"].astype(np.float64)
        frac = self.fine.values["FRACGOOD"].astype(np.float64)
        if sky is not None and sky != self.box:
            inside = self.pixels_in(sky)
            sig, frac = sig[inside], frac[inside]
        f = 10.0 ** (-0.4 * (mag - zeropoint))
        fcut = np.maximum(snr_min * sig, 10.0 ** (-0.4 * (mag_max - zeropoint)))
        psel = ndtr((f[:, None] - fcut[None, :]) / sig[None, :])
        return np.sum(psel * frac[None, :], axis=1) * pix_area_deg2(self.nside)

    def pixels_in(self, sky) -> np.ndarray:
        """Mask of the map's pixels whose centres lie in ``sky``."""
        import healpy as hp

        ra, dec = hp.pix2ang(self.nside, self.fine.pixels, nest=True, lonlat=True)
        return np.asarray(sky.contains(ra, dec), bool)

    def digest(self) -> str:
        """Hash of the map (pixels and values), for checkpoint keys and provenance."""
        import hashlib

        h = hashlib.sha1(f"{self.nside} {self.bands} {self.density}".encode())
        h.update(np.ascontiguousarray(self.fine.pixels).tobytes())
        for k in sorted(self.fine.values):
            h.update(k.encode())
            h.update(np.ascontiguousarray(self.fine.values[k]).tobytes())
        return h.hexdigest()

    def write(self, path: str | Path) -> Path:
        hdr = {"BANDS": ",".join(self.bands), "DENSITY": self.density, **sky_header(self.box)}
        return write_fits(path, [self.fine.hdu("FINE", hdr)])

    @classmethod
    def read(cls, path: str | Path) -> "Footprint":
        from astropy.io import fits

        fine = SparseMap.from_hdu(path, "FINE")
        with fits.open(path) as h:
            hdr = h["FINE"].header
            bands = tuple(hdr["BANDS"].split(","))
            density = float(hdr["DENSITY"])
            box = sky_from_header(hdr)
        return cls(fine, bands, density, box)


def good_randoms(r: dict, cfg: RemaConfig) -> np.ndarray:
    """Randoms passing the galaxies' footprint cuts (MASKBITS and NOBS in every band)."""
    reject = np.int64(sum(1 << int(b) for b in cfg.survey.maskbits_reject))
    ok = (r["MASKBITS"].astype(np.int64) & reject) == 0
    for b in cfg.survey.bands:
        ok &= r[f"NOBS_{b.upper()}"] >= cfg.survey.nobs_min
        ok &= np.isfinite(r[f"GALDEPTH_{b.upper()}"]) & (r[f"GALDEPTH_{b.upper()}"] > 0)
    if cfg.survey.ebv_max is not None:
        ok &= r["EBV"] < cfg.survey.ebv_max
    return ok


def build_footprint(randoms: dict, cfg: RemaConfig | None = None, box: Box | BoxUnion | None = None,
                    density: float | None = None) -> Footprint:
    """Build the fine footprint/depth map from a randoms table (dict of columns).

    ``box`` (a Box or a BoxUnion) is the data box: randoms outside it are ignored and pixels it
    cuts are weighted by their covered fraction. The result does not depend on the row order.
    """
    cfg = cfg or RemaConfig()
    nside = cfg.mask.nside_fine
    bands = tuple(b.upper() for b in cfg.survey.bands)
    ra, dec = np.asarray(randoms["RA"], np.float64), np.asarray(randoms["DEC"], np.float64)
    keep = np.ones(ra.size, bool) if box is None else box.contains(ra, dec)
    r = {k: np.asarray(v)[keep] for k, v in randoms.items()}
    ra, dec = ra[keep], dec[keep]
    good = good_randoms(r, cfg)
    pix = ang2pix_nest_np(nside, ra, dec)
    upix, inv, nrand = np.unique(pix, return_inverse=True, return_counts=True)
    ngood = np.bincount(inv, weights=good, minlength=upix.size)
    # Pixels cut by the data box are only partly covered: weight by the covered fraction.
    cover = np.ones(upix.size) if box is None else box_cover(upix, nside, box)
    values = {"NRAND": nrand.astype(np.int32), "COVER": cover.astype(np.float32),
              "FRACGOOD": (ngood / nrand * cover).astype(np.float32)}
    # Depth fallback for pixels without good randoms: the parent pixel (nside_fine / 4).
    parent = upix // 16
    gpar = pix[good] // 16
    upar, ipar = np.unique(gpar, return_inverse=True)
    j = np.clip(np.searchsorted(upar, parent), 0, max(upar.size - 1, 0))
    has_parent = (upar.size > 0) & (upar[j] == parent) if upar.size else np.zeros(parent.size, bool)
    for b in bands:
        sig = sigma_flux_from_depth(r[f"GALDEPTH_{b}"][good], r["EBV"][good], b.lower())
        med = _grouped_median(inv[good], sig, upix.size)
        pmed = _grouped_median(ipar, sig, upar.size)
        fill_parent = np.where(has_parent, pmed[j] if upar.size else np.nan, np.nan)
        med = np.where(np.isfinite(med), med, fill_parent)
        med = np.where(np.isfinite(med), med, np.nanmedian(sig) if sig.size else SIGF_NONE)
        values[f"SIGF_{b}"] = med.astype(np.float32)
    fine = SparseMap(nside, upix, values, fill=0.0)
    return Footprint(fine, bands, float(density or cfg.mask.randoms_density), box)


def box_cover(pix: np.ndarray, nside: int, box: Box | BoxUnion, nsub: int = 8) -> np.ndarray:
    """Fraction of each NESTED pixel inside ``box`` (or a union), from nsub^2 sub-pixel centres."""
    import healpy as hp

    k = nsub * nsub
    children = (np.asarray(pix, np.int64)[:, None] * k + np.arange(k)[None, :]).ravel()
    ra, dec = hp.pix2ang(nside * nsub, children, nest=True, lonlat=True)
    return box.contains(ra, dec).reshape(-1, k).mean(axis=1)


def _grouped_median(group: np.ndarray, x: np.ndarray, ngroups: int) -> np.ndarray:
    """Median of x within integer groups 0..ngroups-1 (NaN for empty groups)."""
    out = np.full(ngroups, np.nan)
    if x.size == 0:
        return out
    order = np.lexsort((x, group))
    g, xs = group[order], x[order]
    starts = np.searchsorted(g, np.arange(ngroups), side="left")
    ends = np.searchsorted(g, np.arange(ngroups), side="right")
    n = ends - starts
    has = n > 0
    lo = starts + (n - 1) // 2
    hi = starts + n // 2
    out[has] = 0.5 * (xs[lo[has]] + xs[hi[has]])
    return out


def read_randoms(path: str | Path, cfg: RemaConfig | None = None, box: Box | None = None,
                 chunk: int = 5_000_000) -> dict:
    """Read the randoms columns needed for the footprint, optionally inside ``box`` only."""
    from astropy.io import fits

    cfg = cfg or RemaConfig()
    bands = [b.upper() for b in cfg.survey.bands]
    cols = ["RA", "DEC", "MASKBITS", "EBV"] + [f"NOBS_{b}" for b in bands] + \
           [f"GALDEPTH_{b}" for b in bands]
    out = {c: [] for c in cols}
    with fits.open(path, memmap=True) as h:
        data = h[1].data
        n = len(data)
        for lo in range(0, n, chunk):
            sl = slice(lo, min(n, lo + chunk))
            ra = np.asarray(data["RA"][sl], np.float64)
            dec = np.asarray(data["DEC"][sl], np.float64)
            keep = np.ones(ra.size, bool) if box is None else box.contains(ra, dec)
            if not np.any(keep):
                continue
            for c in cols:
                v = np.asarray(data[c][sl])[keep]
                out[c].append(v.astype(v.dtype.newbyteorder("=")))
    return {c: (np.concatenate(v) if v else np.zeros(0)) for c, v in out.items()}


# --------------------------------------------------------------------------- randoms index
# The rows of a DR11 randoms file are in random sky order, so the randoms of any box need a full
# read of the 23 GB file. index_randoms reads it once and writes a pixel-sorted copy of the
# columns read_randoms returns; read_randoms_index then reads only the rows near a box.

RANDOMS_INDEX_NSIDE = 64


def _randoms_columns(cfg: RemaConfig) -> list[str]:
    bands = [b.upper() for b in cfg.survey.bands]
    return ["RA", "DEC", "MASKBITS", "EBV"] + [f"NOBS_{b}" for b in bands] + \
           [f"GALDEPTH_{b}" for b in bands]


def index_randoms(path: str | Path, outdir: str | Path, cfg: RemaConfig | None = None,
                  box: Box | BoxUnion | None = None, nside_index: int = RANDOMS_INDEX_NSIDE,
                  chunk: int = 5_000_000) -> Path:
    """Pixel-sorted copy of a randoms file, for fast reads of any sky box.

    Writes, in ``outdir``, ``<stem>.npy`` (one structured array of the :func:`read_randoms`
    columns, sorted by NESTED pixel at ``mask.nside_fine``), ``<stem>.offsets.npy`` (the first
    row of every NESTED pixel at ``nside_index``, plus one past the last) and ``<stem>.json``
    (source file, size and time, nside values, columns, number of rows), written last. An index
    of the same source and settings is kept. Peak memory is about 120 bytes per random.
    """
    import json
    import os

    cfg = cfg or RemaConfig()
    path, outdir = Path(path), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = path.name.removesuffix(".fits")
    meta_path = outdir / f"{stem}.json"
    st = path.stat()
    meta = {"source": str(path.resolve()), "size": st.st_size, "mtime": int(st.st_mtime),
            "nside_fine": cfg.mask.nside_fine, "nside_index": nside_index,
            "columns": _randoms_columns(cfg), "box": sky_header(box)}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if all(old.get(k) == v for k, v in meta.items()):
            return outdir / f"{stem}.npy"
    r = read_randoms(path, cfg, box, chunk)
    pix = ang2pix_nest_np(cfg.mask.nside_fine, r["RA"], r["DEC"])
    order = np.argsort(pix, kind="stable")
    arr = np.empty(order.size, dtype=[(c, r[c].dtype) for c in meta["columns"]])
    for c in meta["columns"]:
        arr[c] = r[c][order]
    shift = 2 * int(round(np.log2(cfg.mask.nside_fine // nside_index)))
    coarse = pix[order] >> shift
    offsets = np.searchsorted(coarse, np.arange(12 * nside_index**2 + 1))
    for name, a in ((f"{stem}.npy", arr), (f"{stem}.offsets.npy", offsets)):
        tmp = outdir / f".{name}.tmp.npy"
        np.save(tmp, a)
        os.replace(tmp, outdir / name)
    meta["nrows"] = int(order.size)
    tmp = outdir / f".{stem}.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, meta_path)
    return outdir / f"{stem}.npy"


def randoms_index_files(index_dir: str | Path) -> list[str]:
    """Stems of the complete randoms indexes in ``index_dir``, in natural order (…-2 before …-10)."""
    import re

    stems = [p.name.removesuffix(".json") for p in Path(index_dir).glob("*.json")
             if not p.name.startswith(".")]
    key = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
    return sorted(stems, key=key)


def _coarse_candidates(box: Box | BoxUnion, nside_index: int) -> np.ndarray:
    """NESTED pixels at ``nside_index`` that can hold points of ``box``: those whose centre lies
    within the maximum pixel radius of it."""
    import healpy as hp

    npix = 12 * nside_index**2
    ra, dec = hp.pix2ang(nside_index, np.arange(npix), nest=True, lonlat=True)
    rad = float(np.degrees(hp.max_pixrad(nside_index))) * 1.01
    keep = np.zeros(npix, bool)
    for b in box.boxes:
        keep |= b.buffered(rad).contains(ra, dec)
    return np.flatnonzero(keep)


def read_randoms_index(index_dir: str | Path, cfg: RemaConfig | None = None,
                       box: Box | BoxUnion | None = None, files: list[str] | None = None) -> dict:
    """The randoms of ``box`` from the indexes in ``index_dir``: the same rows (in another order)
    as :func:`read_randoms` on each indexed file, concatenated. ``files``: the index stems to use
    (default all, see :func:`randoms_index_files`)."""
    import json

    cfg = cfg or RemaConfig()
    index_dir = Path(index_dir)
    stems = files if files is not None else randoms_index_files(index_dir)
    if not stems:
        raise FileNotFoundError(f"no randoms index in {index_dir}")
    parts = []
    for stem in stems:
        meta = json.loads((index_dir / f"{stem}.json").read_text())
        arr = np.load(index_dir / f"{stem}.npy", mmap_mode="r")
        if box is None:
            parts.append(np.asarray(arr))
            continue
        off = np.load(index_dir / f"{stem}.offsets.npy")
        cand = _coarse_candidates(box, int(meta["nside_index"]))
        cand = cand[off[cand + 1] > off[cand]]
        sub = np.concatenate([arr[off[c]:off[c + 1]] for c in cand]) if cand.size \
            else np.asarray(arr[:0])
        parts.append(sub[box.contains(sub["RA"], sub["DEC"])])
    rows = np.concatenate(parts)
    return {c: np.ascontiguousarray(rows[c]) for c in rows.dtype.names}
