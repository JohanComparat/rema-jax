"""Runs the scan mode and the spectroscopic post-processing on a list of positions, joined to the input.

:func:`scan_positions` takes a table with RA and DEC (any number of positions, anywhere in the
survey) and returns it with the results of :func:`rema.modes.scan.run_scan` and
:func:`rema.modes.specpost.process` added, row for row: every input column renamed
``INPUT_<NAME>``, the pipeline columns under their rema names, and ``SCANNED`` (false for a
position outside the galaxy tables). It builds what the scan needs around the positions only:

1. the sweeps overlapping the positions' boxes (each position buffered by ``buffer`` deg) are
   ingested whole, with their photo-z, into ``work/galaxies`` (kept and reused by later runs);
2. the randoms of the union of the boxes are indexed once per run (``work/randoms_index/<hash>``),
   unless an existing index is given (``index``: e.g. the full-sky one of ``rema randoms-index``);
3. the positions are grouped into sky cells of ``cell`` deg; for each group the galaxies and the
   footprint of its boxes are read, the region is built with the calibration and the positions
   are scanned.

The members keep ``MEM_MATCH_ID`` = the input row (0-based), which is also the cluster's
``MEM_MATCH_ID``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np

from ..sky.regions import Box, bounding_box, sky_union

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- geometry
def group_positions(ra, dec, cell: float = 10.0) -> list[np.ndarray]:
    """Return index arrays of the positions grouped into sky cells: Dec bands of ``cell`` deg, cut in RA
    into cells of about ``cell`` deg on the sky (wider in RA towards the poles).

    >>> [g.tolist() for g in group_positions([1.0, 2.0, 359.0, 100.0], [0.0, 1.0, 0.0, -89.0])]
    [[3], [0, 1], [2]]
    """
    ra = np.mod(np.asarray(ra, np.float64), 360.0)
    dec = np.asarray(dec, np.float64)
    band = np.floor((dec + 90.0) / cell).astype(np.int64)
    centre = -90.0 + (band + 0.5) * cell
    nra = np.maximum(1, np.floor(360.0 * np.cos(np.radians(np.clip(centre, -89.9, 89.9))) / cell)).astype(np.int64)
    col = np.minimum(np.floor(ra / (360.0 / nra)).astype(np.int64), nra - 1)
    key = band * 100_000 + col
    return [np.flatnonzero(key == k) for k in np.unique(key)]


def data_box(ra, dec, buffer: float = 1.2, max_boxes: int = 50):
    """Return the union of the positions' boxes, each a point buffered by ``buffer`` deg (a Box for one
    position, else a BoxUnion); with more than ``max_boxes`` positions, their bounding box (a
    union of thousands of boxes makes every ``contains`` slow).

    >>> data_box([10.0], [0.0], 1.0)
    Box(ra_min=9.0, ra_max=11.0, dec_min=-1.0, dec_max=1.0)
    >>> data_box([10.0, 12.0, 14.0], [0.0, 0.0, 0.0], 1.0, max_boxes=2)
    Box(ra_min=9.0, ra_max=15.0, dec_min=-1.0, dec_max=1.0)
    """
    boxes = [Box(float(r), float(r), float(d), float(d)).buffered(buffer)
             for r, d in zip(np.mod(np.asarray(ra, np.float64), 360.0), np.asarray(dec, np.float64))]
    if len(boxes) > max_boxes:
        return bounding_box(boxes)
    return sky_union(boxes)


def _parts(sky):
    return list(getattr(sky, "boxes", [sky]))


def _box_hash(sky) -> str:
    from ..sky.regions import sky_header

    return hashlib.sha1(repr(sorted(sky_header(sky).items())).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- join
def _fill(dtype, shape):
    kind = np.dtype(dtype).kind
    if kind in "US":
        return np.zeros(shape, dtype)
    if kind == "b":
        return np.zeros(shape, bool)
    return np.full(shape, -1, dtype)


def join_to_input(table: dict, cat: dict, nrows: int | None = None) -> dict:
    """Return the input columns as ``INPUT_<NAME>``, then the pipeline columns placed by the input row in
    ``cat["MEM_MATCH_ID"]``, then ``SCANNED``; rows without a result get fill values (-1 for
    numbers, empty strings, False).

    >>> out = join_to_input({"ra": np.array([1.0, 2.0])}, {"MEM_MATCH_ID": np.array([1]),
    ...                     "LAMBDA": np.array([30.0])})
    >>> sorted(out), out["LAMBDA"].tolist(), out["SCANNED"].tolist()
    (['INPUT_RA', 'LAMBDA', 'MEM_MATCH_ID', 'SCANNED'], [-1.0, 30.0], [False, True])
    """
    n = nrows if nrows is not None else len(next(iter(table.values())))
    out = {f"INPUT_{k.upper()}": np.asarray(v) for k, v in table.items()}
    rows = np.asarray(cat.get("MEM_MATCH_ID", np.zeros(0, np.int64)), np.int64)
    for k, v in cat.items():
        v = np.asarray(v)
        col = _fill(v.dtype, (n,) + v.shape[1:])
        col[rows] = v
        out[k] = col
    if "MEM_MATCH_ID" in out:
        out["MEM_MATCH_ID"][:] = np.arange(n)
    scanned = np.zeros(n, bool)
    scanned[rows] = True
    out["SCANNED"] = scanned
    return out


# --------------------------------------------------------------------------- run
def scan_positions(table: dict, *, ra_col: str = "RA", dec_col: str = "DEC", dr11=None, calib=None,
                   work=None, cfg=None, buffer: float = 1.2, cell: float = 10.0, nrand: int = 4,
                   batch: int = 128, specpost: bool = True, galaxies=None, footprint=None,
                   index=None, info: dict | None = None):
    """Scan the positions of ``table`` (dict of columns) and return ``(joined, members)`` (see the module
    docstring).

    ``dr11``: the DR11 south directory (``sweep/11.0``, ``sweep/11.0-photo-z``,
    ``randoms/randoms-south-1-*.fits``); ``calib``: the calibration file; ``work``: the work
    directory; ``cfg``: the configuration (default: the calibration's); ``galaxies`` (a galaxy
    table or a directory of per-sweep tables) and ``footprint`` (a Footprint) replace the
    ingestion and the randoms; ``index``: a randoms index directory covering the positions, used
    instead of indexing the randoms; ``nrand``: randoms files used for the footprints.
    """
    from ..calibration import Calibration
    from ..io.legacy import ingest_sweeps, read_galaxies
    from ..sky.maps import build_footprint, index_randoms, randoms_index_files, read_randoms_index
    from ..sky.regions import sweeps_overlapping
    from .common import Region
    from .scan import run_scan
    from . import specpost as sp

    t0 = time.time()
    cols = {k.upper(): k for k in table}
    for c in (ra_col, dec_col):
        if c.upper() not in cols:
            raise KeyError(f"the input has no column {c!r}; columns: {sorted(table)}")
    ra = np.mod(np.asarray(table[cols[ra_col.upper()]], np.float64), 360.0)
    dec = np.asarray(table[cols[dec_col.upper()]], np.float64)
    n = ra.size
    cal = Calibration.read(calib)
    cfg = cfg or cal.config
    ok = np.isfinite(ra) & np.isfinite(dec) & (np.abs(dec) <= 90)
    stats = {"n_positions": int(n), "n_groups": 0, "n_sweeps": 0}
    work = Path(work) if work is not None else None

    groups = [np.flatnonzero(ok)[g] for g in group_positions(ra[ok], dec[ok], cell)]
    boxes = [data_box(ra[rows], dec[rows], buffer) for rows in groups]
    # the sweeps and the randoms of every position, once
    sky_all = sky_union([b for sky in boxes for b in _parts(sky)]) if groups else None
    gal_src = galaxies
    if gal_src is None and sky_all is not None:
        if work is None:
            raise ValueError("give a work directory (or galaxies and a footprint)")
        sweeps = sweeps_overlapping(sky_all, Path(dr11) / "sweep" / "11.0")
        sweeps = [s for s in sweeps if not s.name.endswith("-pz.fits")]
        stats["n_sweeps"] = len(sweeps)
        log.info("ingesting %d sweeps into %s", len(sweeps), work / "galaxies")
        ingest_sweeps(sweeps, work / "galaxies", cfg, require_pz=True)
        gal_src = work / "galaxies"
    index_dir, stems = None, None
    if footprint is None and sky_all is not None:
        if index is not None:
            index_dir = Path(index)
        else:
            index_dir = work / "randoms_index" / _box_hash(sky_all)
            files = sorted((Path(dr11) / "randoms").glob("randoms-south-1-*.fits"))[:nrand]
            if not files:
                raise FileNotFoundError(f"no randoms in {Path(dr11) / 'randoms'}")
            for f in files:
                log.info("indexing %s over the positions' boxes", f.name)
                index_randoms(f, index_dir, cfg, box=sky_all)
        stems = randoms_index_files(index_dir)[:nrand]
        if not stems:
            raise FileNotFoundError(f"no randoms index in {index_dir}")
    stats["t_prepare"] = time.time() - t0

    cats, mems = [], []
    for rows, sky in zip(groups, boxes):
        try:
            gal = read_galaxies(gal_src, sky, cfg)
        except FileNotFoundError:
            gal = {}
        if not gal or np.size(gal["RA"]) == 0:
            log.info("group of %d positions: no galaxies (outside the survey)", rows.size)
            continue
        fp = footprint
        if fp is None:
            rnd = read_randoms_index(index_dir, cfg, sky, files=stems)
            fp = build_footprint(rnd, cfg, box=sky, density=cfg.mask.randoms_density * len(stems))
        reg = Region.build(gal, cal.rs, cfg, footprint=fp, zredcorr=cal.zredcorr, bkg=cal.bkg,
                           zlcorr=cal.zlcorr, zbkg=cal.zbkg, wcen_params=cal.wcen, sky=sky)
        log.info("group of %d positions: %d galaxies", rows.size, np.size(gal["RA"]))
        cat, mem = run_scan(reg, ra[rows], dec[rows], ids=rows, batch=batch)
        if specpost and cat:
            sc = cfg.spec
            cat, mem = sp.process(cat, mem, min_members=sc.min_members, vmax_init=sc.vmax_init,
                                  nsigma_clip=sc.nsigma_clip, niter=sc.niter, gapper_nmax=sc.gapper_nmax,
                                  c_location=sc.c_location, c_scale=sc.c_scale, nboot=sc.nboot,
                                  seed=sc.seed)
        cats.append(cat)
        mems.append(mem)
        stats["n_groups"] += 1
    cat = _concat(cats)
    mem = _concat(mems)
    joined = join_to_input(table, cat, n)
    stats.update(n_scanned=int(joined["SCANNED"].sum()),
                 n_found=int(np.sum(joined.get("LAMBDA", np.zeros(n)) > 0)), t_total=time.time() - t0)
    log.info("scanned %d of %d positions, %d with lambda > 0, %.1fs", stats["n_scanned"], n,
             stats["n_found"], stats["t_total"])
    if info is not None:
        info.update(stats)
    return joined, mem


def _concat(parts):
    parts = [p for p in parts if p]
    if not parts:
        return {}
    keys = [k for k in parts[0] if all(k in p for p in parts)]
    return {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in keys}
