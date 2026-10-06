"""Production runs over large areas: region planning, status, merging and boundary QA.

A large area is cut into regions. Each region has an *own box* and a *data box*, which is the own
box grown by a buffer. A region reads the galaxies and randoms of its data box and keeps the
clusters whose final centre lies in its own box. Own boxes are made of whole 5 deg sweep tiles
and never overlap, so the region catalogues concatenate into one catalogue without duplicates
(``MEM_MATCH_ID = region << 32 | rank``).

:func:`plan_regions` builds the regions from the per-sweep galaxy counts
(:func:`rema.io.legacy.galaxy_counts`):

- **Dec bands:** each polar cap (|Dec| > ``polar_cap``) is one full-RA ring. In between, the
  bands are ``band_height`` deg high, aligned to the sweep rows.
- **Own boxes:** in each band they are runs of 5 deg RA tiles. The runs start after the band's
  largest RA gap, so RA 0/360 is handled, and are long enough to hold about ``target_area``
  deg^2.
- **Splitting:** a region whose data box would hold more than ``max_gal`` galaxies, or more than
  ``max_pairs`` estimated percolation dependency pairs, is split in RA, then in Dec rows.
- **Order:** regions are numbered by decreasing estimated cost, so the slowest jobs start first.

:func:`merge_regions` concatenates the region catalogues and runs the QA:

- duplicate IDs;
- centres outside their own box;
- close pairs across regions;
- members without a cluster;
- galaxies with more than unit total membership;
- mixed code versions, calibrations or configurations;
- z_lambda against the spectroscopic redshifts, per region.

:func:`edge_profile` gives the cluster density against the distance to internal region edges.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .config import RemaConfig
from .io.tables import Table, read_catalog, read_header, read_table, write_table
from .sky.regions import Box, BoxUnion, intersect, sky_from_header, sky_header

log = logging.getLogger(__name__)

TILE = 5.0                      # sweep tiles are 5 x 5 deg
# Development area (75 deg^2): 3.48 M galaxies gave 57 M percolation dependency pairs; pairs
# scale as N^2 / area.
_PAIRS_REF = (57e6, 3.48e6, 75.0)


def file_sha1(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- planning
def _covered_area(tiles: list[tuple[float, float]], ngal: np.ndarray, box) -> tuple[float, float]:
    """(galaxies, area with galaxies) of the tiles inside ``box`` (a Box or BoxUnion), each tile
    counted by the fraction of its area inside the box."""
    n = a = 0.0
    for (ra0, dec0), g in zip(tiles, ngal):
        t = Box(ra0, ra0 + TILE, dec0, dec0 + TILE)
        part = intersect(t, box)
        if part is None:
            continue
        f = (part.area_deg2() if isinstance(part, (Box, BoxUnion)) else 0.0) / t.area_deg2()
        n += f * g
        a += f * t.area_deg2() if g > 0 else 0.0
    return n, a


def _pairs(ngal: float, area: float) -> float:
    p0, n0, a0 = _PAIRS_REF
    return p0 * (ngal / n0) ** 2 * (a0 / max(area, 1e-9))


def _ra_runs(cols: np.ndarray, k: int) -> list[tuple[float, float]]:
    """Runs of ``k`` 5-deg columns covering the occupied columns ``cols`` (RA starts), starting
    after the largest circular gap: [(ra_start, ra_end)], ra_end possibly > 360."""
    cols = np.unique(np.mod(cols, 360.0))
    if cols.size == 1:
        return [(float(cols[0]), float(cols[0]) + k * TILE)]
    gaps = np.diff(np.append(cols, cols[0] + 360.0))
    start = cols[(int(np.argmax(gaps)) + 1) % cols.size]
    rel = np.mod(cols - start, 360.0)
    groups = np.unique(np.floor(rel / (k * TILE) + 1e-9).astype(int))
    return [(float(start + g * k * TILE), float(min(start + (g + 1) * k * TILE, start + 360.0)))
            for g in groups]


def galactic_latitude(ra, dec) -> np.ndarray:
    """Galactic latitude b [deg] of ICRS positions [deg]."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    return SkyCoord(np.asarray(ra, float) * u.deg, np.asarray(dec, float) * u.deg, frame="icrs").galactic.b.deg


def box_abs_glat(box: Box, n: int = 41) -> tuple[float, float]:
    """(smallest, largest) |b| [deg] over an n x n grid of ``box``."""
    ra = np.linspace(box.ra_min, box.ra_max, n) % 360.0
    dec = np.linspace(box.dec_min, box.dec_max, n)
    r, d = np.meshgrid(ra, dec)
    b = np.abs(galactic_latitude(r.ravel(), d.ravel()))
    return float(b.min()), float(b.max())


def plan_regions(counts: Table, *, target_area: float = 100.0, buffer: float = 2.0,
                 band_height: float = 10.0, polar_cap: float = 85.0, max_gal: float | None = None,
                 max_pairs: float | None = None, sky: Box | BoxUnion | None = None,
                 glat_min: float | None = None):
    """Regions for a blind run over the tiles in ``counts``. Returns (plan table, meta dict).

    Parameters
    ----------
    counts : per-sweep table with NAME, RA0, DEC0 and NGAL (see :func:`galaxy_counts`). Tiles
        without galaxies get no region.
    target_area : own area per region (deg^2), reached with whole tiles.
    buffer : data box = own box grown by ``buffer`` deg (:meth:`Box.buffered`), cut to ``sky``
        when given (e.g. the sweeps available in a test).
    band_height : height of the Dec bands (deg, a multiple of 5).
    polar_cap : |Dec| above which a cap is one full-RA region.
    max_gal, max_pairs : split regions whose data box exceeds these estimates.
    glat_min : drop the regions whose own box lies entirely at |b| < ``glat_min`` deg (the
        catalogue is then cut at |b| >= ``glat_min`` by :func:`cut_glat`).
    """
    ra0 = np.asarray(counts["RA0"], np.float64)
    dec0 = np.asarray(counts["DEC0"], np.float64)
    ngal = np.asarray(counts["NGAL"], np.float64)
    occupied = ngal > 0
    tiles = list(zip(ra0, dec0))
    if not occupied.any():
        raise ValueError("no tile with galaxies")

    # Dec bands: caps, then band_height bands between them.
    lo, hi = -polar_cap, polar_cap
    edges = [-90.0] + list(np.arange(lo, hi, band_height)) + [hi, 90.0]
    edges = sorted(set(float(e) for e in edges))
    regions = []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        in_band = occupied & (dec0 >= b0) & (dec0 < b1)
        if not in_band.any():
            continue
        if b0 <= -90.0 or b1 >= 90.0:                    # polar cap: one ring
            regions.append(Box(0.0, 360.0, b0, b1))
            continue
        col_area = Box(0.0, TILE, b0, b1).area_deg2()
        k = max(1, int(round(target_area / col_area)))
        if k * TILE >= 360.0:
            regions.append(Box(0.0, 360.0, b0, b1))
            continue
        for r0, r1 in _ra_runs(ra0[in_band], k):
            regions.append(Box(r0, r1, b0, b1))

    def describe(own: Box) -> dict:
        data = own.buffered(buffer)
        if sky is not None:
            data_sky = intersect(data, sky)
        else:
            data_sky = data
        n_own, _ = _covered_area(tiles, ngal, own)
        n_data, a_data = _covered_area(tiles, ngal, data_sky) if data_sky is not None else (0, 0)
        return {"own": own, "data": data, "NGAL_OWN": n_own, "NGAL_DATA": n_data,
                "AREA_COVERED": a_data, "PAIRS_EST": _pairs(n_data, a_data)}

    def too_big(d: dict) -> bool:
        return ((max_gal is not None and d["NGAL_DATA"] > max_gal)
                or (max_pairs is not None and d["PAIRS_EST"] > max_pairs))

    def split(own: Box) -> list[Box]:
        ncol = int(round((own.ra_max - own.ra_min) / TILE))
        nrow = int(round((own.dec_max - own.dec_min) / TILE))
        if own.is_ring:
            ncol = 72
        if ncol > 1:
            h = (ncol // 2) * TILE
            return [Box(own.ra_min, own.ra_min + h, own.dec_min, own.dec_max),
                    Box(own.ra_min + h, own.ra_max if not own.is_ring else 360.0,
                        own.dec_min, own.dec_max)]
        if nrow > 1:
            h = (nrow // 2) * TILE
            return [Box(own.ra_min, own.ra_max, own.dec_min, own.dec_min + h),
                    Box(own.ra_min, own.ra_max, own.dec_min + h, own.dec_max)]
        return [own]

    done, todo = [], [describe(r) for r in regions]
    while todo:
        d = todo.pop()
        parts = split(d["own"]) if too_big(d) else [d["own"]]
        if len(parts) == 1:
            if too_big(d):
                log.warning("region %s exceeds the caps but is a single tile", d["own"])
            if d["NGAL_OWN"] > 0:
                done.append(d)
        else:
            todo += [describe(p) for p in parts]

    if glat_min is not None:
        low = [d for d in done if box_abs_glat(d["own"])[1] < glat_min]
        if low:
            log.info("%d regions lie entirely at |b| < %g deg and are left out", len(low), glat_min)
        done = [d for d in done if box_abs_glat(d["own"])[1] >= glat_min]
    done.sort(key=lambda d: (-d["PAIRS_EST"], d["own"].dec_min, d["own"].ra_min))
    plan = {"REGION_ID": np.arange(len(done), dtype=np.int64)}
    for name, key in (("OWN", "own"), ("DATA", "data")):
        for i, attr in enumerate(("RA0", "RA1", "DEC0", "DEC1")):
            plan[f"{name}_{attr}"] = np.array([d[key].as_tuple()[i] for d in done])
    plan["AREA_OWN"] = np.array([d["own"].area_deg2() for d in done])
    plan["AREA_COVERED"] = np.array([d["AREA_COVERED"] for d in done])
    for k in ("NGAL_OWN", "NGAL_DATA", "PAIRS_EST"):
        plan[k] = np.array([d[k] for d in done])
    meta = {"TARGET": target_area, "BUFFER": buffer, "BANDH": band_height, "CAP": polar_cap,
            "MAXGAL": -1.0 if max_gal is None else float(max_gal),
            "MAXPAIR": -1.0 if max_pairs is None else float(max_pairs),
            "NREGION": len(done), **sky_header(sky)}
    if glat_min is not None:
        meta["GLATMIN"] = float(glat_min)
    h = hashlib.sha1(json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                 for k, v in meta.items()}, sort_keys=True).encode())
    for k in sorted(plan):
        h.update(np.ascontiguousarray(plan[k]).tobytes())
    meta["PLANHASH"] = h.hexdigest()[:12]
    return plan, meta


def write_plan(path: str | Path, plan: Table, meta: dict) -> Path:
    return write_table(path, plan, header=meta, extname="REGIONS")


def read_plan(path: str | Path) -> tuple[Table, dict]:
    hdr = read_header(path, "REGIONS")
    meta = {k: hdr[k] for k in hdr if k in ("TARGET", "BUFFER", "BANDH", "CAP", "MAXGAL", "MAXPAIR",
                                             "NREGION", "PLANHASH")}
    sky = sky_from_header(hdr)
    if sky is not None:
        meta.update(sky_header(sky))
    return read_table(path, hdu="REGIONS"), meta


def plan_boxes(plan: Table, region_id: int, meta: dict | None = None):
    """(own Box, data Box or BoxUnion) of a region; the data box is cut to the plan's sky."""
    i = int(np.flatnonzero(plan["REGION_ID"] == region_id)[0])
    own = Box(*(float(plan[f"OWN_{a}"][i]) for a in ("RA0", "RA1", "DEC0", "DEC1")))
    data = Box(*(float(plan[f"DATA_{a}"][i]) for a in ("RA0", "RA1", "DEC0", "DEC1")))
    sky = sky_from_header(meta or {})
    if sky is not None:
        data = intersect(data, sky)
    return own, data


def required_buffer(cfg: RemaConfig, lam: float = 100.0, z: float | None = None) -> dict:
    """Buffers (deg) that a region needs around a cluster of richness ``lam`` at redshift ``z``
    (default: the low end of ``model.zrange``, where angles are largest).

    ``exact``: percolation's dependency radius, read + reach from 2 lambda as
    :func:`rema.modes.blind.percolate` bounds them. ``first_order``: the cluster's aperture plus
    a higher-ranked neighbour's mask radius and seed offset, which is what changes its members
    directly.
    """
    from .model.cosmo import CosmoTable

    rc, pc = cfg.richness.percolation, cfg.percolation
    z = cfg.model.zrange[0] if z is None else z
    cosmo = CosmoTable.create(cfg.cosmology.Omega_m, cfg.cosmology.h)
    deg = lambda r, zz: float(r / float(cosmo.mpc_per_deg(max(zz, cfg.model.zrange[0]))))
    maxrad = cfg.richness.maxrad_factor * rc.r0 * 3.0**rc.beta
    lam2 = 2.0 * lam
    rl2 = rc.r0 * (lam2 / 100.0) ** rc.beta
    rm2 = pc.rmask_0 * (lam2 / 100.0) ** pc.rmask_beta
    zlo = z - 0.05
    exact = deg(max(maxrad, rm2) + rl2, zlo) + deg(rm2 + rl2, zlo)
    rl = rc.r0 * (lam / 100.0) ** rc.beta
    rm = pc.rmask_0 * (lam / 100.0) ** pc.rmask_beta
    return {"exact": exact, "first_order": deg(maxrad + rm + rl, z)}


def uncovered_tiles(counts: Table, index_dir: str | Path, sky=None) -> list[str]:
    """Tiles with randoms but no per-sweep galaxy table (within ``sky`` when given): data missing
    from the mirror, which the footprint would count as observed."""
    import healpy as hp

    from .sky.maps import randoms_index_files

    stems = randoms_index_files(index_dir)
    if not stems:
        return []
    meta = json.loads((Path(index_dir) / f"{stems[0]}.json").read_text())
    off = np.load(Path(index_dir) / f"{stems[0]}.offsets.npy")
    nside = int(meta["nside_index"])
    pix = np.flatnonzero(np.diff(off) > 0)
    ra, dec = hp.pix2ang(nside, pix, nest=True, lonlat=True)
    have = set(zip(np.asarray(counts["RA0"], float), np.asarray(counts["DEC0"], float)))
    missing = set()
    for r, d in zip(np.floor(ra / TILE) * TILE, np.floor(dec / TILE) * TILE):
        if (float(r), float(d)) in have:
            continue
        if sky is not None and not sky.overlaps(Box(r, r + TILE, d, d + TILE)):
            continue
        missing.add((float(r), float(d)))
    return [_tile_name(r, d) for r, d in sorted(missing)]


def _tile_name(ra0: float, dec0: float) -> str:
    def dec(v):
        return f"{'m' if v < 0 else 'p'}{abs(int(round(v))):03d}"
    return f"sweep-{int(ra0):03d}{dec(dec0)}-{int(ra0 + TILE):03d}{dec(dec0 + TILE)}.fits"


def calib_suggest(counts: Table, area: float = 400.0, ebv_max: float = 0.05,
                  n: int = 5) -> list[dict]:
    """Calibration areas: square blocks of tiles of about ``area`` deg^2, every tile with
    galaxies and mean E(B-V) < ``ebv_max``, ranked by their number of spectroscopic redshifts
    (NZSPEC); the ``n`` best that do not overlap."""
    ra0 = np.asarray(counts["RA0"], float)
    dec0 = np.asarray(counts["DEC0"], float)
    tab = {(r, d): (g, z, e) for r, d, g, z, e in zip(ra0, dec0, counts["NGAL"], counts["NZSPEC"],
                                                       counts["EBVMEAN"])}
    m = max(1, int(round(np.sqrt(area) / TILE)))
    cands = []
    for r, d in tab:
        cells = [((r + TILE * i) % 360.0, d + TILE * j) for i in range(m) for j in range(m)]
        vals = [tab.get(c) for c in cells]
        if any(v is None or v[0] <= 0 or v[2] >= ebv_max for v in vals):
            continue
        box = Box(r, r + m * TILE, d, d + m * TILE)
        cands.append({"box": box, "NZSPEC": int(sum(v[1] for v in vals)),
                      "NGAL": int(sum(v[0] for v in vals)), "AREA": box.area_deg2(),
                      "EBVMAX": float(max(v[2] for v in vals))})
    cands.sort(key=lambda c: -c["NZSPEC"])
    out = []
    for c in cands:
        if all(not c["box"].overlaps(o["box"]) for o in out):
            out.append(c)
        if len(out) == n:
            break
    return out


# --------------------------------------------------------------------------- status and merge
def region_path(runs_dir: str | Path, region_id: int) -> Path:
    return Path(runs_dir) / f"{int(region_id):04d}" / "clusters.fits"


def region_status(plan_path: str | Path, runs_dir: str | Path, calib: str | Path | None = None,
                  version: str | None = None) -> dict:
    """done / missing / stale region IDs, and ``prime``: the missing region of median cost.

    A region is stale when its catalogue was made with another plan (PLANHASH), calibration
    (CALSHA1, when ``calib`` is given) or rema version (REMAVER, when ``version`` is given).
    """
    plan, meta = read_plan(plan_path)
    cal = file_sha1(calib) if calib is not None else None
    out = {"done": [], "missing": [], "stale": []}
    for rid in plan["REGION_ID"]:
        p = region_path(runs_dir, rid)
        if not p.exists():
            out["missing"].append(int(rid))
            continue
        h = read_header(p, 0)
        ok = h.get("PLANHASH") == meta["PLANHASH"]
        ok &= cal is None or h.get("CALSHA1") == cal
        ok &= version is None or h.get("REMAVER") == version
        out["done" if ok else "stale"].append(int(rid))
    todo = sorted(out["missing"] + out["stale"])
    if todo:
        cost = {int(r): float(c) for r, c in zip(plan["REGION_ID"], plan["PAIRS_EST"])}
        by_cost = sorted(todo, key=lambda r: cost[r])
        out["prime"] = by_cost[len(by_cost) // 2]
    else:
        out["prime"] = None
    return out


def close_pairs(cat: Table, radius_arcmin: float = 1.0, dz: float = 0.02):
    """Index pairs (i, j), i < j, of clusters within ``radius_arcmin`` and |dz| < dz (1 + z)."""
    from scipy.spatial import cKDTree

    from .sky.neighbors import unit_vectors

    if not cat or len(cat["RA"]) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    xyz = unit_vectors(cat["RA"], cat["DEC"])
    chord = 2.0 * np.sin(np.radians(radius_arcmin / 60.0) / 2.0)
    pairs = np.array(sorted(cKDTree(xyz).query_pairs(chord)), dtype=int).reshape(-1, 2)
    if pairs.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    z = np.asarray(cat["Z_LAMBDA"], float)
    i, j = pairs[:, 0], pairs[:, 1]
    keep = np.abs(z[i] - z[j]) < dz * (1.0 + 0.5 * (z[i] + z[j]))
    return i[keep], j[keep]


def _zstats(cat: Table) -> tuple[int, float, float]:
    if not cat or "SPEC_Z_BOOT" not in cat:
        return 0, np.nan, np.nan
    sel = (cat["LAMBDA"] >= 20) & (cat["N_MEMBERS"] >= 3) & np.isfinite(cat["SPEC_Z_BOOT"])
    if sel.sum() < 3:
        return int(sel.sum()), np.nan, np.nan
    zs = cat["SPEC_Z_BOOT"][sel]
    d = (cat["Z_LAMBDA"][sel] - zs) / (1 + zs)
    b = float(np.median(d))
    return int(sel.sum()), b, float(1.4826 * np.median(np.abs(d - b)))


def cut_glat(cat: Table, mem: Table, glat_min: float) -> tuple[Table, Table, int]:
    """Clusters at |b| >= ``glat_min`` deg and their members; also returns the number removed."""
    if not cat:
        return cat, mem, 0
    keep = np.abs(galactic_latitude(cat["RA"], cat["DEC"])) >= glat_min
    ids = np.asarray(cat["MEM_MATCH_ID"])[keep]
    cat = {k: np.asarray(v)[keep] for k, v in cat.items()}
    if mem:
        km = np.isin(np.asarray(mem["MEM_MATCH_ID"]), ids)
        mem = {k: np.asarray(v)[km] for k, v in mem.items()}
    return cat, mem, int(np.sum(~keep))


def merge_regions(plan_path: str | Path, runs_dir: str | Path, *, allow_missing: bool = False,
                  calib: str | Path | None = None):
    """Concatenate the region catalogues. Returns (clusters, members, regions table, qa dict).

    Raises when regions are missing or stale, unless ``allow_missing``. See the module docstring
    for the QA.
    """
    plan, meta = read_plan(plan_path)
    st = region_status(plan_path, runs_dir, calib)
    if (st["missing"] or st["stale"]) and not allow_missing:
        raise RuntimeError(f"{len(st['missing'])} missing and {len(st['stale'])} stale regions "
                           f"(e.g. {(st['missing'] + st['stale'])[:10]}); use allow_missing")
    cats, mems, rows = [], [], []
    versions, cals, configs = set(), set(), set()
    for rid in st["done"]:
        cat, mem, hdr = read_catalog(region_path(runs_dir, rid))
        versions.add(hdr.get("REMAVER"))
        cals.add(hdr.get("CALSHA1"))
        configs.add(hdr.get("CFGSHA1"))
        nz, zb, zn = _zstats(cat)
        rows.append((rid, len(cat.get("LAMBDA", [])), len(mem.get("ID", [])),
                     int(hdr.get("NGAL", -1)), int(hdr.get("NSEED", -1)), int(hdr.get("NCAND", -1)),
                     int(hdr.get("NDEP", -1)), float(hdr.get("TTOTAL", np.nan)),
                     float(hdr.get("PEAKRSS", np.nan)), str(hdr.get("DEVICE", "")), nz, zb, zn))
        if cat:
            cats.append(cat)
        if mem:
            mems.append(mem)
    names = ("REGION_ID", "NCLUSTER", "NMEMBER", "NGAL", "NSEED", "NCAND", "NDEP", "TTOTAL",
             "PEAKRSS", "DEVICE", "NZSPEC", "ZBIAS", "ZNMAD")
    regions = {n: np.array(c) for n, c in zip(names, zip(*rows))} if rows else {n: np.zeros(0) for n in names}
    keys = [k for k in cats[0] if all(k in c for c in cats)] if cats else []
    cat = {k: np.concatenate([c[k] for c in cats]) for k in keys}
    mkeys = [k for k in mems[0] if all(k in m for m in mems)] if mems else []
    mem = {k: np.concatenate([m[k] for m in mems]) for k in mkeys}

    qa = {"n_regions": int(len(plan["REGION_ID"])), "n_done": len(st["done"]),
          "missing": st["missing"], "stale": st["stale"],
          "n_empty": int(sum(1 for r in rows if r[1] == 0)), "n_clusters": len(cat.get("RA", [])),
          "n_members": len(mem.get("ID", [])), "versions": sorted(map(str, versions)),
          "calibrations": sorted(map(str, cals)), "configs": sorted(map(str, configs))}
    if cat:
        mmid = np.asarray(cat["MEM_MATCH_ID"], np.int64)
        qa["duplicate_ids"] = int(mmid.size - np.unique(mmid).size)
        reg = mmid >> 32
        outside = 0
        for rid in np.unique(reg):
            own, _ = plan_boxes(plan, int(rid), meta)
            sel = reg == rid
            outside += int(np.sum(~own.contains(cat["RA"][sel], cat["DEC"][sel])))
        qa["centres_outside_own"] = outside
        i, j = close_pairs(cat)
        cross = reg[i] != reg[j]
        qa["close_pairs"] = int(i.size)
        qa["close_pairs_cross_region"] = int(cross.sum())
        qa["close_pairs_examples"] = [(int(mmid[a]), int(mmid[b])) for a, b in
                                      zip(i[cross][:20], j[cross][:20])]
    if mem and cat:
        mid = np.asarray(mem["MEM_MATCH_ID"], np.int64)
        qa["members_without_cluster"] = int(np.sum(~np.isin(mid, cat["MEM_MATCH_ID"])))
        if "PMEM" in mem:
            ids, inv = np.unique(mem["ID"], return_inverse=True)
            tot = np.bincount(inv, weights=np.asarray(mem["PMEM"], float))
            # Regions contributing to each galaxy: distinct (galaxy, region) keys.
            key = np.unique((inv.astype(np.int64) << 20) | (mid >> 32))
            nreg = np.bincount((key >> 20).astype(np.int64), minlength=ids.size)
            over = tot > 1.001
            qa["galaxies_pmem_over_1"] = int(over.sum())
            qa["galaxies_pmem_over_1_cross_region"] = int(np.sum(over & (nreg > 1)))
    return cat, mem, regions, qa


def edge_profile(cat: Table, plan: Table, meta: dict | None = None,
                 bins: np.ndarray | None = None, nrand: int = 200_000, seed: int = 1) -> Table:
    """Cluster density against the distance (deg) to the nearest internal edge of their own box
    (an edge shared with another region), relative to points spread uniformly over the own
    boxes: a dip or a bump near 0 reveals a boundary effect."""
    bins = np.arange(0.0, 3.01, 0.25) if bins is None else np.asarray(bins)
    rids = np.asarray(plan["REGION_ID"])
    owns = {int(r): plan_boxes(plan, int(r), meta)[0] for r in rids}

    def internal_sides(own: Box) -> list[str]:
        sides = []
        eps = 1e-3
        ras = np.linspace(own.ra_min, own.ra_max, 9)[1:-1]
        decs = np.linspace(own.dec_min, own.dec_max, 9)[1:-1]
        probes = {"dec_min": (ras, np.full(ras.size, own.dec_min - eps)),
                  "dec_max": (ras, np.full(ras.size, own.dec_max + eps))}
        if not own.is_ring:
            probes["ra_min"] = (np.full(decs.size, own.ra_min - eps), decs)
            probes["ra_max"] = (np.full(decs.size, own.ra_max + eps), decs)
        for side, (ra, dec) in probes.items():
            if abs(dec[0]) >= 90:
                continue
            if any(np.any(o.contains(ra, dec)) for o in owns.values() if o != own):
                sides.append(side)
        return sides

    def dist(ra, dec, own: Box, sides) -> np.ndarray:
        d = np.full(np.size(ra), np.inf)
        for s in sides:
            if s == "dec_min":
                d = np.minimum(d, dec - own.dec_min)
            elif s == "dec_max":
                d = np.minimum(d, own.dec_max - dec)
            elif s == "ra_min":
                d = np.minimum(d, np.mod(ra - own.ra_min, 360.0) * np.cos(np.radians(dec)))
            else:
                d = np.minimum(d, np.mod(own.ra_max - ra, 360.0) * np.cos(np.radians(dec)))
        return d

    reg = np.asarray(cat["MEM_MATCH_ID"], np.int64) >> 32
    rng = np.random.default_rng(seed)
    dc, dr = [], []
    area = np.array([owns[int(r)].area_deg2() for r in rids])
    nr = rng.multinomial(nrand, area / area.sum())
    for r, n in zip(rids, nr):
        own = owns[int(r)]
        sides = internal_sides(own)
        sel = reg == r
        dc.append(dist(cat["RA"][sel], cat["DEC"][sel], own, sides))
        u = rng.uniform(np.sin(np.radians(own.dec_min)), np.sin(np.radians(own.dec_max)), n)
        dec = np.degrees(np.arcsin(u))
        ra = rng.uniform(own.ra_min, own.ra_min + min(360.0, own.ra_max - own.ra_min), n)
        dr.append(dist(ra, dec, own, sides))
    dc, dr = np.concatenate(dc), np.concatenate(dr)
    hc, _ = np.histogram(dc, bins)
    hr, _ = np.histogram(dr, bins)
    norm = dc.size / max(dr.size, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = hc / (hr * norm)
    return {"D_LO": bins[:-1], "D_HI": bins[1:], "N": hc, "N_UNIFORM": hr * norm, "RATIO": ratio}
