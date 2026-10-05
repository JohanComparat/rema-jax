"""FITS tables, sky boxes, DR11 ingestion and the randoms footprint."""

import numpy as np
import pytest

from rema.config import RemaConfig
from rema.io.tables import read_table, write_table
from rema.sky.healpix import pix_area_deg2
from rema.sky.maps import Footprint, SparseMap, build_footprint, depth_ivar_from_mag, sigma_flux_from_depth
from rema.sky.regions import Box, sweep_box

from conftest import SWEEP, SWEEP_PZ, require


def test_table_roundtrip(tmp_path):
    cols = {"ID": np.arange(5, dtype=np.int64), "X": np.linspace(0, 1, 5).astype(np.float32),
            "V": np.ones((5, 4), np.float32), "M": np.ones((5, 2, 3)), "F": np.array([1, 0, 1, 1, 0], bool),
            "U": np.arange(5, dtype=np.uint8)}
    p = write_table(tmp_path / "t.fits", cols, header={"FOO": 3}, extname="T")
    back = read_table(p, hdu="T")
    for k, v in cols.items():
        np.testing.assert_array_equal(back[k], v)
        assert back[k].dtype.byteorder in ("=", "|")
    sub = read_table(p, ["x"], rows=np.array([1, 3]), hdu="T")
    np.testing.assert_allclose(sub["X"], [0.25, 0.75])


def test_box_wrap_and_area():
    b = Box(-2.0, 3.0, -1.0, 1.0)
    assert b.contains(359.0, 0.0) and b.contains(2.5, 0.5) and not b.contains(3.5, 0.0)
    assert Box(0, 360, -90, 90).area_deg2() == pytest.approx(41252.96, rel=1e-6)
    assert sweep_box("sweep-350p005-355p010.fits") == Box(350.0, 355.0, 5.0, 10.0)
    assert Box(355, 365, 0, 5).overlaps(Box(0, 5, 0, 5))
    assert not Box(10, 20, 0, 5).overlaps(Box(0, 5, 0, 5))


def _synthetic_randoms(rng, n=400_000, box=Box(10.0, 14.0, -2.0, 2.0)):
    ra = rng.uniform(box.ra_min, box.ra_max, n)
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(box.dec_min)),
                                           np.sin(np.radians(box.dec_max)), n)))
    star = (ra - 12.0) ** 2 + dec**2 < 0.5**2          # a 0.5 deg "bright star" mask
    r = {"RA": ra, "DEC": dec, "MASKBITS": np.where(star, 2, 0).astype(np.int32),
         "EBV": np.full(n, 0.03, np.float32)}
    for b, d in zip("GRIZ", (24.9, 24.7, 24.2, 23.6)):
        r[f"NOBS_{b}"] = np.full(n, 3, np.int16)
        r[f"GALDEPTH_{b}"] = np.full(n, depth_ivar_from_mag(d), np.float32)   # inverse variance
    return r, box, star


def test_footprint_from_randoms(tmp_path, rng):
    r, box, star = _synthetic_randoms(rng)
    cfg = RemaConfig().replace(mask={"nside_fine": 256})
    fp = build_footprint(r, cfg, box=box, density=r["RA"].size / box.area_deg2())
    # Unmasked area = box - disc (pixels straddling the edges are fractional).
    expected = box.area_deg2() - np.pi * 0.5**2
    assert fp.area_deg2() == pytest.approx(expected, rel=0.02)
    sig_z = sigma_flux_from_depth(depth_ivar_from_mag(23.6), 0.03, "z")
    np.testing.assert_allclose(fp.fine.values["SIGF_Z"], sig_z, rtol=1e-5)
    # Very bright galaxies are selected everywhere unmasked; galaxies at the 5-sigma limit in half of it.
    m5 = 22.5 - 2.5 * np.log10(5 * sig_z)
    a_bright, a_lim, a_faint = fp.effective_area([18.0, m5, m5 + 2], "z", snr_min=5)
    assert a_bright == pytest.approx(fp.area_deg2(), rel=1e-6)
    assert a_lim == pytest.approx(0.5 * fp.area_deg2(), rel=1e-3)
    assert a_faint < 1e-3 * fp.area_deg2()
    # Device lookup agrees with the host lookup, and a masked point has a small fraction good.
    dm = fp.fine.device(("FRACGOOD",))
    ra_t, dec_t = np.array([12.0, 13.5, 10.5]), np.array([0.0, 1.5, -1.5])
    host = fp.fine.lookup_np(ra_t, dec_t, "FRACGOOD")
    dev = np.asarray(dm.lookup(ra_t, dec_t, "FRACGOOD"))
    np.testing.assert_allclose(dev, host)
    assert host[0] < 0.05 and host[1] > 0.95
    back = Footprint.read(fp.write(tmp_path / "fp.fits"))
    np.testing.assert_array_equal(back.fine.pixels, fp.fine.pixels)
    assert back.bands == fp.bands and back.box == box


def test_sparse_map_absent_pixels():
    m = SparseMap(4, np.array([5, 1]), {"V": np.array([2.0, 1.0])}, fill=-1.0)
    assert list(m.pixels) == [1, 5]
    assert pix_area_deg2(4) == pytest.approx(41252.96 / 192, rel=1e-6)


@pytest.mark.data
def test_ingest_dr11_tile():
    require(SWEEP, SWEEP_PZ)
    from rema.io.legacy import ingest_sweep

    box = Box(1.0, 2.0, -3.0, -2.0)
    t = ingest_sweep(SWEEP, RemaConfig(), box=box)
    n = t["ID"].size
    assert 30_000 < n < 90_000                      # ~55k galaxies per deg^2 after cuts
    assert np.all(box.contains(t["RA"], t["DEC"]))
    assert np.all(t["TYPE"] != 0)                     # no PSF
    snr = t["FLUX"][:, 3] * np.sqrt(t["FLUX_IVAR"][:, 3])
    assert snr.min() >= 5.0 - 1e-4
    np.testing.assert_allclose(t["REFMAG"], 22.5 - 2.5 * np.log10(t["FLUX"][:, 3]), atol=1e-4)
    assert np.all((t["ZSPEC"] == -1) | (t["ZSPEC"] > 0.001))
    assert np.unique(t["ID"]).size == n
