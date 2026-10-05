"""Edge cases of sky boxes and unions, neighbour padding, HEALPix, the randoms readers and FITS
tables."""

import json

import numpy as np
import pytest

from rema.config import RemaConfig
from rema.io.tables import nrows, read_table, write_table
from rema.sky import maps as M
from rema.sky.healpix import ang2pix_nest, ang2pix_nest_np
from rema.sky.neighbors import NeighborIndex, bucket_order, next_pow2, radec_from_unit, unit_vectors
from rema.sky.regions import (Box, BoxUnion, bounding_box, sky_union, sweep_box, sweep_union,
                              sweeps_overlapping)


# --------------------------------------------------------------------------- boxes
def test_buffered_wide_box_becomes_a_ring():
    assert Box(0, 350, 0, 10).buffered(10.0) == Box(0, 360, -10, 20)
    assert Box(0, 360, 0, 10).buffered(1.0) == Box(0, 360, -1, 11)
    b = Box(10, 20, 0, 10).buffered(1.0)
    assert not b.is_ring and b.dec_min == -1.0


def test_intersection_with_rings():
    ring = Box(0, 360, -10, 10)
    assert ring.intersection(Box(0, 360, 5, 20)) == [Box(0, 360, 5, 10)]
    assert ring.intersection(Box(350, 365, 0, 20)) == [Box(350, 365, 0, 10)]
    assert Box(350, 365, 0, 20).intersection(ring) == [Box(350, 365, 0, 10)]
    assert Box(0, 10, 0, 5).intersection(Box(0, 10, 5, 8)) == []


def test_union_overlaps_and_bounding():
    a, b = Box(0, 5, 0, 5), Box(10, 15, -5, 0)
    assert a.union(b) == Box(0, 15, -5, 5)
    assert a.bounding() is a and a.boxes == (a,)
    u = BoxUnion((a, b))
    assert a.overlaps(u) and u.overlaps(Box(12, 13, -1, 1)) and not u.overlaps(Box(6, 9, -5, 5))
    assert not u.is_ring
    assert Box(0, 360, 0, 1).overlaps(Box(100, 101, 0, 1))
    # BoxUnion.intersect: a Box, a union, or None.
    assert u.intersect(Box(1, 2, 1, 2)) == Box(1, 2, 1, 2)
    assert isinstance(u.intersect(Box(0, 15, -5, 5)), BoxUnion)
    assert u.intersect(Box(100, 101, 0, 1)) is None
    with pytest.raises(ValueError, match="at least one"):
        BoxUnion(())
    # Bounding boxes: rings stay rings, intervals covering the whole circle give a ring.
    assert bounding_box([Box(0, 360, 0, 1), a]) == Box(0, 360, 0, 5)
    assert bounding_box([Box(0, 180, 0, 1), Box(180, 360, 0, 1)]).is_ring
    assert bounding_box([Box(355, 365, 0, 1)]) == Box(355, 365, 0, 1)
    assert sky_union([a]) is a and isinstance(sky_union([a, b]), BoxUnion)


def test_sweep_names(tmp_path):
    with pytest.raises(ValueError, match="not a sweep"):
        sweep_box("galaxies.fits")
    with pytest.raises(ValueError, match="no sweeps"):
        sweep_union([])
    assert isinstance(sweep_union(["sweep-000m005-005p000.fits", "sweep-010m005-015p000.fits"]), BoxUnion)
    for n in ("sweep-000m005-005p000.fits", "sweep-000m005-005p000-pz.fits", "sweep-bad.fits",
              "sweep-100m005-105p000.fits"):
        (tmp_path / n).touch()
    got = sweeps_overlapping(Box(1, 2, -1, 0), tmp_path)
    assert [p.name for p in got] == ["sweep-000m005-005p000.fits"]


# --------------------------------------------------------------------------- neighbours
def test_neighbor_padding():
    idx = NeighborIndex(np.array([10.0, 10.01, 10.02, 50.0]), np.zeros(4))
    pad = idx.query(np.array([10.0, 30.0]), np.array([0.0, 0.0]), np.array([0.05, 0.05]), k=2)
    assert pad.idx.shape == (2, 2) and pad.counts.tolist() == [2, 0]
    assert pad.idx[0].tolist() == [0, 1] and not pad.valid[1].any()        # nearest first
    assert next_pow2(0) == 128 and next_pow2(300) == 512 and next_pow2(3, kmin=1) == 4
    groups = bucket_order(np.array([5, 200, 130, 900]))
    assert [(k, i.tolist()) for k, i in groups] == [(128, [0]), (256, [1, 2]), (1024, [3])]
    ra, dec = radec_from_unit(2.0 * unit_vectors(359.0, -30.0))
    assert ra == pytest.approx(359.0) and dec == pytest.approx(-30.0)


# --------------------------------------------------------------------------- healpix
def test_healpix_device_matches_host_and_checks_nside():
    import healpy as hp

    rng = np.random.default_rng(1)
    ra, dec = rng.uniform(0, 360, 1000), rng.uniform(-90, 90, 1000)
    host = ang2pix_nest_np(256, ra, dec)
    np.testing.assert_array_equal(host, hp.ang2pix(256, ra, dec, nest=True, lonlat=True))
    np.testing.assert_array_equal(np.asarray(ang2pix_nest(256, ra, dec)), host)
    for bad in (100, 16384):
        with pytest.raises(ValueError, match="power of two"):
            ang2pix_nest(bad, ra, dec)


# --------------------------------------------------------------------------- randoms readers
def _randoms(tmp_path, n=2000):
    rng = np.random.default_rng(2)
    cols = {"RA": rng.uniform(10, 12, n), "DEC": rng.uniform(-1, 1, n),
            "MASKBITS": np.zeros(n, np.int16), "EBV": np.full(n, 0.02, np.float32)}
    for b in "GRIZ":
        cols[f"NOBS_{b}"] = np.full(n, 2, np.int16)
        cols[f"GALDEPTH_{b}"] = np.full(n, 100.0, np.float32)
    return write_table(tmp_path / "randoms-south-1-3.fits", cols), cols


def test_read_randoms_chunks_and_index_reuse(tmp_path, caplog):
    cfg = RemaConfig()
    path, cols = _randoms(tmp_path)
    # Small chunks, some of them without any random in the box.
    box = Box(10.0, 10.5, -1, 1)
    r = M.read_randoms(path, cfg, box, chunk=97)
    ref = M.read_randoms(path, cfg, box)
    np.testing.assert_array_equal(r["RA"], ref["RA"])
    assert r["RA"].size == np.sum(box.contains(cols["RA"], cols["DEC"]))
    assert M.read_randoms(path, cfg, Box(100, 101, 0, 1))["RA"].size == 0
    # An index is kept when the source and settings are unchanged, rebuilt otherwise.
    idx = tmp_path / "idx"
    out = M.index_randoms(path, idx, cfg, nside_index=16)
    stamp = out.stat().st_mtime_ns
    assert M.index_randoms(path, idx, cfg, nside_index=16) == out and out.stat().st_mtime_ns == stamp
    M.index_randoms(path, idx, cfg, nside_index=8)
    assert json.loads((idx / "randoms-south-1-3.json").read_text())["nside_index"] == 8
    allr = M.read_randoms_index(idx, cfg)
    assert allr["RA"].size == 2000
    assert M.randoms_index_files(idx) == ["randoms-south-1-3"]
    with pytest.raises(FileNotFoundError, match="no randoms index"):
        M.read_randoms_index(tmp_path / "empty", cfg, box)
    # A box without randoms gives an empty table with the columns.
    empty = M.read_randoms_index(idx, cfg, Box(200, 201, 50, 51))
    assert empty["RA"].size == 0 and "GALDEPTH_Z" in empty


def test_grouped_median_and_footprint_without_good_randoms():
    np.testing.assert_array_equal(M._grouped_median(np.zeros(0, int), np.zeros(0), 3),
                                  [np.nan] * 3)
    np.testing.assert_allclose(M._grouped_median(np.array([0, 0, 0, 2]), np.array([3.0, 1.0, 2.0, 5.0]), 3),
                               [2.0, np.nan, 5.0])
    n = 500
    rng = np.random.default_rng(3)
    r = {"RA": rng.uniform(10, 11, n), "DEC": rng.uniform(0, 1, n), "MASKBITS": np.full(n, 2, np.int16),
         "EBV": np.zeros(n, np.float32)}
    for b in "GRIZ":
        r[f"NOBS_{b}"] = np.ones(n, np.int16)
        r[f"GALDEPTH_{b}"] = np.full(n, 100.0, np.float32)
    fp = M.build_footprint(r, RemaConfig().replace(mask={"nside_fine": 64}))
    assert fp.area_deg2() == 0.0
    np.testing.assert_array_equal(fp.fine.values["SIGF_Z"], M.SIGF_NONE)


# --------------------------------------------------------------------------- tables
def test_table_strings_int8_and_rows(tmp_path):
    t = {"NAME": np.array([b"ab ", b"c  "]), "U": np.array(["x", "yz"]),
         "I8": np.array([-3, 4], np.int8)}
    p = write_table(tmp_path / "t.fits", t)
    assert nrows(p) == 2
    back = read_table(p)
    assert [str(s.decode() if isinstance(s, bytes) else s).strip() for s in back["NAME"]] == ["ab", "c"]
    np.testing.assert_array_equal(back["I8"], [-3, 4])
    low = read_table(p, ["i8"], upper=False)
    assert list(low) == ["I8"]
