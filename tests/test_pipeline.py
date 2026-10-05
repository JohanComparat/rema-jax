"""Large runs: sky boxes and unions, per-sweep tables, the randoms index, the region planner, the
merge and its QA."""

import numpy as np
import pytest

from rema.config import RemaConfig
from rema.io.legacy import read_galaxies, survey_hash
from rema.io.tables import read_catalog, write_catalog, write_table
from rema.pipeline import (TILE, calib_suggest, close_pairs, edge_profile, merge_regions,
                           plan_boxes, plan_regions, read_plan, region_status, required_buffer,
                           write_plan)
from rema.sky.maps import build_footprint, index_randoms, read_randoms, read_randoms_index
from rema.sky.regions import (Box, BoxUnion, bounding_box, intersect, sky_from_header, sky_header,
                              sweep_box, sweep_union)


@pytest.fixture
def rng():
    """A local generator: the session one is shared with other modules, whose draws must not
    depend on these tests."""
    return np.random.default_rng(7)


def _sphere_offset(ra, dec, d, pa):
    """Points at angular distance d (deg) and position angle pa (rad) from (ra, dec)."""
    r, de, t = np.radians(ra), np.radians(dec), np.radians(d)
    s = np.sin(de) * np.cos(t) + np.cos(de) * np.sin(t) * np.cos(pa)
    dec2 = np.degrees(np.arcsin(np.clip(s, -1, 1)))
    ra2 = np.degrees(r + np.arctan2(np.sin(pa) * np.sin(t) * np.cos(de), np.cos(t) - np.sin(de) * s))
    return ra2, dec2


@pytest.mark.parametrize("box", [Box(10, 20, -5, 5), Box(355, 365, 60, 70), Box(0, 5, -80, -75),
                                 Box(100, 140, -84, -75)])
def test_buffered_holds_every_point_within_the_buffer(rng, box):
    d = 2.0
    n = 20000
    ra = rng.uniform(box.ra_min, box.ra_max, n)
    dec = rng.uniform(box.dec_min, box.dec_max, n)
    ra2, dec2 = _sphere_offset(ra, dec, rng.uniform(0, d, n), rng.uniform(0, 2 * np.pi, n))
    big = box.buffered(d)
    assert np.all(big.contains(ra2, dec2))
    # Exact half-width at the largest |Dec| (not wider than needed).
    dmax = max(abs(box.dec_min), abs(box.dec_max))
    assert big.ra_max - box.ra_max == pytest.approx(np.degrees(np.arcsin(np.sin(np.radians(d))
                                                                         / np.cos(np.radians(dmax)))))


def test_buffered_reaching_the_pole_is_a_ring():
    assert Box(0, 5, -88, -85).buffered(2).is_ring
    assert Box(10, 20, 86, 89).buffered(2) == Box(0, 360, 84, 90)
    assert Box(0, 360, -90, -85).buffered(2) == Box(0, 360, -90, -83)


def test_intersection_union_and_headers():
    assert Box(350, 370, 0, 5).intersection(Box(5, 355, -1, 1)) == [Box(350, 355, 0, 1),
                                                                    Box(5, 10, 0, 1)]
    assert Box(0, 10, 0, 1).intersection(Box(20, 30, 0, 1)) == []
    assert bounding_box([Box(355, 360, 0, 5), Box(0, 5, 0, 5)]) == Box(355, 365, 0, 5)
    assert bounding_box([Box(0, 200, 0, 5), Box(150, 370, 0, 5)]).is_ring
    u = sweep_union(["sweep-000m005-005p000.fits", "sweep-005m010-010m005.fits"])
    assert isinstance(u, BoxUnion) and u.area_deg2() == pytest.approx(49.75, rel=1e-3)
    assert list(u.contains([2, 7, 2], [-2, -7, -7])) == [True, True, False]
    assert u.bounding() == Box(0, 10, -10, 0)
    assert sky_from_header(sky_header(u)) == u
    assert sky_from_header(sky_header(Box(1, 2, 3, 4))) == Box(1, 2, 3, 4)
    assert sweep_union(["sweep-355m005-360p000.fits", "sweep-000m005-005p000.fits"]) == Box(355, 365, -5, 0)
    assert intersect(u, Box(3, 8, -8, -3)).area_deg2() == pytest.approx(
        Box(3, 5, -5, -3).area_deg2() + Box(5, 8, -8, -5).area_deg2())


def _write_sweep_tables(tmp_path, names, rng, cfg, n=500):
    for s in names:
        b = sweep_box(s)
        t = {"ID": rng.integers(0, 2**40, n), "RA": rng.uniform(b.ra_min, b.ra_max, n) % 360.0,
             "DEC": rng.uniform(b.dec_min, b.dec_max, n), "ZSPEC": np.full(n, -1.0, np.float32)}
        write_table(tmp_path / s, t, header={"SURVHASH": survey_hash(cfg), "NZSPEC": 0,
                                             "EBVMEAN": 0.02, **sky_header(b)}, extname="GALAXIES")


def test_read_galaxies_from_per_sweep_tables(tmp_path, rng):
    cfg = RemaConfig()
    names = ["sweep-355m005-360p000.fits", "sweep-000m005-005p000.fits", "sweep-005m005-010p000.fits",
             "sweep-000m010-005m005.fits"]
    _write_sweep_tables(tmp_path, names, rng, cfg)
    box = Box(357, 367, -7, -1)                          # across RA 0, three tiles
    g = read_galaxies(tmp_path, box, cfg)
    from rema.io.tables import read_table
    ref = [read_table(tmp_path / s) for s in sorted(names)]
    ra = np.concatenate([t["RA"] for t in ref])
    dec = np.concatenate([t["DEC"] for t in ref])
    assert g["ID"].size == np.sum(box.contains(ra, dec))
    assert np.all(box.contains(g["RA"], g["DEC"]))
    with pytest.raises(ValueError, match="give a box"):
        read_galaxies(tmp_path)
    with pytest.raises(ValueError, match="SURVHASH"):
        read_galaxies(tmp_path, box, cfg.replace(survey={"ebv_max": 0.2}))


def _randoms_file(tmp_path, rng, n=200_000, box=Box(-2, 3, -2, 2)):
    from rema.sky.maps import depth_ivar_from_mag

    ra = rng.uniform(box.ra_min, box.ra_max, n) % 360.0
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(box.dec_min)), np.sin(np.radians(box.dec_max)), n)))
    cols = {"RA": ra, "DEC": dec, "MASKBITS": np.where(rng.random(n) < 0.05, 2, 0).astype(np.int16),
            "EBV": rng.uniform(0, 0.1, n).astype(np.float32)}
    for b, d in zip("GRIZ", (24.9, 24.7, 24.2, 23.6)):
        cols[f"NOBS_{b}"] = rng.integers(0, 4, n).astype(np.int16)
        cols[f"GALDEPTH_{b}"] = np.full(n, depth_ivar_from_mag(d), np.float32) * rng.uniform(0.5, 2, n).astype(np.float32)
    return write_table(tmp_path / "randoms-south-1-0.fits", cols)


def test_randoms_index_equals_direct_read(tmp_path, rng):
    cfg = RemaConfig().replace(mask={"nside_fine": 256})
    path = _randoms_file(tmp_path, rng)
    idx = tmp_path / "index"
    index_randoms(path, idx, cfg, nside_index=16)
    for box in (Box(-1, 2, -1, 1), Box(358, 361, -1.5, 0.5), BoxUnion((Box(0, 1, 0, 1), Box(1, 2, -1, 0)))):
        a, b = read_randoms_index(idx, cfg, box), read_randoms(path, cfg, box)
        oa, ob = np.lexsort((a["DEC"], a["RA"])), np.lexsort((b["DEC"], b["RA"]))
        for k in b:
            np.testing.assert_array_equal(a[k][oa], b[k][ob])
        assert build_footprint(a, cfg, box=box).digest() == build_footprint(b, cfg, box=box).digest()
    # The E(B-V) cut removes randoms from the good fraction, as from the galaxies.
    fp = build_footprint(b, cfg.replace(survey={"ebv_max": 0.05}), box=box)
    assert fp.area_deg2() < build_footprint(b, cfg, box=box).area_deg2()


def _counts(rows):
    names = []
    for ra0, dec0, n in rows:
        d = lambda v: f"{'m' if v < 0 else 'p'}{abs(int(v)):03d}"
        names.append(f"sweep-{int(ra0):03d}{d(dec0)}-{int(ra0 + 5):03d}{d(dec0 + 5)}.fits")
    a = np.array(rows, float)
    return {"NAME": np.array(names), "RA0": a[:, 0], "DEC0": a[:, 1], "RA1": a[:, 0] + 5,
            "DEC1": a[:, 1] + 5, "NGAL": a[:, 2], "NZSPEC": a[:, 2] / 50, "EBVMEAN": np.full(len(rows), 0.02)}


def test_plan_regions_tiles_the_sky_once(rng):
    rows = [(ra, dec, 1e6) for ra in range(0, 360, 5) for dec in (-10, -5, 0, 5)]
    rows += [(ra, -90, 2e5) for ra in range(0, 360, 5)]                        # polar cap
    rows += [(ra, -50, 0) for ra in range(0, 60, 5)]                           # no galaxies
    counts = _counts(rows)
    plan, meta = plan_regions(counts, target_area=100, buffer=2)
    own = [plan_boxes(plan, int(r), meta)[0] for r in plan["REGION_ID"]]
    # Every tile with galaxies in exactly one own box; edges on the 5 deg grid; one polar ring.
    cra, cdec = counts["RA0"] + 2.5, counts["DEC0"] + 2.5
    n_in = sum(o.contains(cra, cdec).astype(int) for o in own)
    assert np.all(n_in[counts["NGAL"] > 0] == 1) and np.all(n_in[counts["NGAL"] == 0] == 0)
    for o in own:
        assert all(abs(v / TILE - round(v / TILE)) < 1e-9 for v in o.as_tuple())
    assert sum(o.is_ring for o in own) == 1
    for r in plan["REGION_ID"]:
        o, d = plan_boxes(plan, int(r), meta)
        assert d == o.buffered(2)
    # Deterministic, ordered by decreasing cost; caps split regions.
    plan2, meta2 = plan_regions(counts, target_area=100, buffer=2)
    assert meta2["PLANHASH"] == meta["PLANHASH"]
    assert np.all(np.diff(plan["PAIRS_EST"]) <= 0)
    cap = 0.6 * plan["NGAL_DATA"].max()
    plan3, _ = plan_regions(counts, target_area=100, buffer=2, max_gal=cap)
    assert len(plan3["REGION_ID"]) > len(plan["REGION_ID"])
    big = plan3["NGAL_DATA"] > cap
    assert np.all((plan3["OWN_RA1"][big] - plan3["OWN_RA0"][big] <= 5) & (plan3["OWN_DEC1"][big] - plan3["OWN_DEC0"][big] <= 5))


def test_plan_roundtrip_sky_and_suggest(tmp_path):
    counts = _counts([(ra, dec, 1e6) for ra in (0, 5, 10) for dec in (-15, -10, -5)])
    sky = Box(0, 15, -15, 0)
    plan, meta = plan_regions(counts, target_area=25, band_height=5, sky=sky)
    assert len(plan["REGION_ID"]) == 9
    write_plan(tmp_path / "p.fits", plan, meta)
    plan2, meta2 = read_plan(tmp_path / "p.fits")
    assert meta2["PLANHASH"] == meta["PLANHASH"]
    own, data = plan_boxes(plan2, 0, meta2)
    assert data == intersect(own.buffered(2), sky)
    s = calib_suggest(counts, area=100)
    assert s and s[0]["box"].area_deg2() == pytest.approx(Box(0, 10, -15, -5).area_deg2(), rel=0.1)


def test_required_buffer():
    cfg = RemaConfig()
    b = [required_buffer(cfg, lam) for lam in (30, 100, 300)]
    assert b[0]["exact"] < b[1]["exact"] < b[2]["exact"]
    assert all(x["first_order"] < x["exact"] for x in b)
    assert required_buffer(cfg, 100, z=0.3)["exact"] < b[1]["exact"]
    assert 1.5 < b[1]["first_order"] < 2.0 < b[1]["exact"] < 2.5      # the 2 deg default


def _region_catalogue(path, rid, own, rng, meta, calib, n=20, extra=None):
    ra = rng.uniform(own.ra_min + 0.1, own.ra_max - 0.1, n)
    dec = rng.uniform(own.dec_min + 0.1, own.dec_max - 0.1, n)
    if extra is not None:
        ra, dec = np.append(ra, extra[0]), np.append(dec, extra[1])
    m = ra.size
    cat = {"MEM_MATCH_ID": (np.int64(rid) << 32) | np.arange(m, dtype=np.int64), "RA": ra, "DEC": dec,
           "Z_LAMBDA": np.full(m, 0.3), "LAMBDA": np.full(m, 10.0)}
    mem = {"MEM_MATCH_ID": np.repeat(cat["MEM_MATCH_ID"], 2), "ID": np.arange(2 * m) + 1000 * rid,
           "PMEM": np.full(2 * m, 0.6)}
    from rema.pipeline import file_sha1
    write_catalog(path, cat, mem, None, {"PLANHASH": meta["PLANHASH"], "CALSHA1": file_sha1(calib)})


def test_merge_qa(tmp_path, rng):
    counts = _counts([(ra, dec, 1e6) for ra in (0, 5) for dec in (-5, 0)])
    plan, meta = plan_regions(counts, target_area=25, band_height=5)
    write_plan(tmp_path / "plan.fits", plan, meta)
    calib = tmp_path / "calib.fits"
    calib.write_bytes(b"c")
    runs = tmp_path / "runs"
    rids = [int(r) for r in plan["REGION_ID"]]
    owns = {r: plan_boxes(plan, r, meta)[0] for r in rids}
    # Region 0 also keeps a cluster outside its own box, next to one of region 1.
    o1 = owns[rids[1]]
    for r in rids[:-1]:
        extra = (o1.ra_min + 0.5, o1.dec_min + 0.5) if r == rids[0] else None
        _region_catalogue(runs / f"{r:04d}" / "clusters.fits", r, owns[r], rng, meta, calib, extra=extra)
    with pytest.raises(RuntimeError, match="missing"):
        merge_regions(tmp_path / "plan.fits", runs, calib=calib)
    write_catalog(runs / f"{rids[-1]:04d}" / "clusters.fits", {}, {}, None,
                  {"PLANHASH": meta["PLANHASH"], "CALSHA1": "x"})
    st = region_status(tmp_path / "plan.fits", runs, calib)
    assert st["stale"] == [rids[-1]]
    cat, mem, regions, qa = merge_regions(tmp_path / "plan.fits", runs, allow_missing=True, calib=calib)
    assert qa["duplicate_ids"] == 0 and qa["centres_outside_own"] == 1
    assert qa["n_clusters"] == 3 * 20 + 1 and qa["members_without_cluster"] == 0
    # A close pair across regions: copy a region-1 cluster position into region 0.
    cat2 = {k: v.copy() for k, v in cat.items()}
    j = np.flatnonzero((cat2["MEM_MATCH_ID"] >> 32) == rids[1])[0]
    cat2["RA"][0], cat2["DEC"][0] = cat2["RA"][j], cat2["DEC"][j] + 0.5 / 60
    i, k = close_pairs(cat2)
    assert any(((cat2["MEM_MATCH_ID"][a] >> 32) != (cat2["MEM_MATCH_ID"][b] >> 32)) for a, b in zip(i, k))
    prof = edge_profile(cat, plan, meta)
    assert np.isfinite(prof["RATIO"]).any()
