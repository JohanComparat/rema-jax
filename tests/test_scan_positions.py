"""rema scan-positions: grouping, data boxes, the join to the input, and runs on synthetic DR11 files
(ingestion of the sweeps, randoms index, scan and spectroscopic post-processing)."""

import numpy as np
import pytest
from astropy.table import Table

from rema import cli
from rema.calibration import Calibration
from rema.config import RemaConfig
from rema.io.legacy import survey_hash
from rema.io.tables import read_catalog, write_table
from rema.model.cosmo import CosmoTable
from rema.model.profiles import MStar
from rema.model.redsequence import RSModel
from rema.modes.positions import data_box, group_positions, join_to_input
from rema.sky.regions import Box
from rema.validate import mocks

from test_cli_commands import DEPTH5, write_randoms
from test_legacy_io import SWEEP_NAME, fake_dr11


@pytest.fixture(autouse=True)
def no_jax_setup(monkeypatch):
    monkeypatch.setattr(cli, "_setup_jax", lambda: None)


def run(*argv):
    assert cli.main([str(a) for a in argv]) == 0


def test_group_positions_and_data_box():
    rng = np.random.default_rng(1)
    ra = np.r_[rng.uniform(0, 360, 200), 359.9, 0.1]
    dec = np.r_[rng.uniform(-89, 89, 200), 10.0, 10.0]
    groups = group_positions(ra, dec, 10.0)
    allrows = np.sort(np.concatenate(groups))
    np.testing.assert_array_equal(allrows, np.arange(ra.size))
    for g in groups:
        assert np.ptp(np.floor((dec[g] + 90) / 10)) == 0          # one Dec band per group
        box = data_box(ra[g], dec[g], 1.2)
        assert np.all(box.contains(ra[g], dec[g]))
    # a buffered position near RA 0 covers both sides of the wrap
    box = data_box([0.1], [10.0], 1.0)
    assert box.contains(np.array([359.5]), np.array([10.0]))[0]
    assert data_box([0.0, 0.0], [89.5, 89.5], 1.0).contains(np.array([180.0]), np.array([89.9]))[0]


def test_join_to_input():
    table = {"RA": np.array([1.0, 2.0, 3.0]), "name": np.array(["a", "b", "c"]),
             "flag": np.array([True, False, True])}
    cat = {"MEM_MATCH_ID": np.array([2, 0]), "RA": np.array([3.0, 1.0]),
           "LAMBDA": np.array([12.0, 30.0], np.float32), "PZ": np.ones((2, 4)),
           "BEST_Z_TYPE": np.array(["spec_z_boot", "photo_z"]), "ZMAX_EDGE": np.array([True, False])}
    out = join_to_input(table, cat, 3)
    np.testing.assert_array_equal(out["INPUT_RA"], table["RA"])
    np.testing.assert_array_equal(out["INPUT_NAME"], table["name"])
    np.testing.assert_array_equal(out["RA"], [1.0, -1.0, 3.0])
    np.testing.assert_array_equal(out["LAMBDA"], [30.0, -1.0, 12.0])
    assert out["LAMBDA"].dtype == np.float32 and out["PZ"].shape == (3, 4) and np.all(out["PZ"][1] == -1)
    assert out["BEST_Z_TYPE"].tolist() == ["photo_z", "", "spec_z_boot"]
    assert out["ZMAX_EDGE"].tolist() == [False, False, True]
    assert out["SCANNED"].tolist() == [True, False, True]
    np.testing.assert_array_equal(out["MEM_MATCH_ID"], [0, 1, 2])
    empty = join_to_input(table, {}, 3)
    assert set(empty) == {"INPUT_RA", "INPUT_NAME", "INPUT_FLAG", "SCANNED"} and not empty["SCANNED"].any()


def _dr11(tmp_path, rng):
    """A DR11 tree with one sweep (RA 10-15, Dec -5-0), its photo-z and one randoms file."""
    fake_dr11(tmp_path, rng, n=400)
    (tmp_path / "randoms").mkdir()
    write_randoms(tmp_path / "randoms" / "randoms-south-1-0.fits", rng, Box(9.0, 16.0, -6.0, 1.0),
                  density=4000.0)
    Calibration(rs=RSModel.from_template(), config=RemaConfig()).write(tmp_path / "cal.fits")
    return tmp_path


def test_cli_scan_positions_ingests_and_indexes(tmp_path):
    rng = np.random.default_rng(7)
    dr11 = _dr11(tmp_path, rng)
    Table({"ra": [12.5, 200.0, 13.0], "dec": [-2.5, 30.0, np.nan], "name": ["in", "out", "bad"]}).write(
        tmp_path / "pos.csv")
    run("scan-positions", tmp_path / "pos.csv", "--dr11", dr11, "--calib", dr11 / "cal.fits",
        "--work", tmp_path / "work", "--ra-col", "ra", "--dec-col", "dec", "--nrand", 1,
        "--out", tmp_path / "out.fits")
    cat, mem, hdr = read_catalog(tmp_path / "out.fits")
    assert hdr["MODE"] == "scan-positions" and hdr["NPOS"] == 3 and hdr["NSCANNED"] == 1
    assert cat["SCANNED"].tolist() == [True, False, False]
    assert [s.decode() if isinstance(s, bytes) else s for s in cat["INPUT_NAME"]] == ["in", "out", "bad"]
    np.testing.assert_allclose(cat["INPUT_RA"], [12.5, 200.0, 13.0])
    assert cat["RA"][0] == pytest.approx(12.5) and cat["RA"][1] == -1
    for col in ("LAMBDA", "Z_LAMBDA", "ZMAX", "NSPEC", "BEST_Z", "RA_OPT", "SNR"):
        assert col in cat
    assert (tmp_path / "work" / "galaxies" / SWEEP_NAME).exists()
    assert list((tmp_path / "work" / "randoms_index").glob("*/randoms-south-1-0.json"))
    if mem:
        assert set(np.unique(mem["MEM_MATCH_ID"])) <= {0}
    # a second run reuses the ingested sweep
    mtime = (tmp_path / "work" / "galaxies" / SWEEP_NAME).stat().st_mtime
    run("scan-positions", tmp_path / "pos.csv", "--dr11", dr11, "--calib", dr11 / "cal.fits",
        "--work", tmp_path / "work", "--ra-col", "ra", "--dec-col", "dec", "--nrand", 1,
        "--no-specpost", "--out", tmp_path / "out2.fits")
    assert (tmp_path / "work" / "galaxies" / SWEEP_NAME).stat().st_mtime == mtime
    out2 = read_catalog(tmp_path / "out2.fits")[0]
    assert "NSPEC" not in out2
    # an existing randoms index (of the whole file) gives the same footprint
    run("randoms-index", dr11 / "randoms" / "randoms-south-1-0.fits", "--outdir", tmp_path / "index")
    run("scan-positions", tmp_path / "pos.csv", "--dr11", dr11, "--calib", dr11 / "cal.fits",
        "--work", tmp_path / "work", "--ra-col", "ra", "--dec-col", "dec", "--index", tmp_path / "index",
        "--no-specpost", "--out", tmp_path / "out3.fits")
    np.testing.assert_allclose(read_catalog(tmp_path / "out3.fits")[0]["LAMBDA"], out2["LAMBDA"], rtol=1e-5)
    with pytest.raises(FileNotFoundError, match="no randoms index"):
        cli.main(["scan-positions", str(tmp_path / "pos.csv"), "--dr11", str(dr11), "--calib",
                  str(dr11 / "cal.fits"), "--work", str(tmp_path / "work"), "--ra-col", "ra", "--dec-col",
                  "dec", "--index", str(tmp_path / "work"), "--out", str(tmp_path / "o.fits")])


def test_cli_scan_positions_errors(tmp_path, monkeypatch):
    rng = np.random.default_rng(8)
    dr11 = _dr11(tmp_path, rng)
    Table({"x": [12.5], "y": [-2.5]}).write(tmp_path / "pos.fits")
    with pytest.raises(KeyError, match="no column"):
        cli.main(["scan-positions", str(tmp_path / "pos.fits"), "--dr11", str(dr11), "--calib",
                  str(dr11 / "cal.fits"), "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o.fits")])
    for var in ("DR11", "LEGACYSURVEY_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cli, "_default_dr11", lambda: None)
    with pytest.raises(SystemExit, match="--dr11"):
        cli.main(["scan-positions", str(tmp_path / "pos.fits"), "--out", str(tmp_path / "o.fits")])


def test_default_dr11(monkeypatch):
    monkeypatch.setenv("DR11", "/x/dr11")
    assert cli._default_dr11() == "/x/dr11"
    monkeypatch.delenv("DR11")
    monkeypatch.setenv("LEGACYSURVEY_DIR", "/y")
    assert cli._default_dr11() == "/y/dr11/south"


@pytest.mark.slow
def test_scan_positions_recovers_a_mock_cluster(tmp_path):
    """A mock cluster in a per-sweep galaxy table: its row gets the cluster's richness and
    redshift; the footprint comes from the indexed randoms."""
    rng = np.random.default_rng(53)
    cfg = RemaConfig()
    rs = RSModel.from_template()
    ms, cosmo = MStar(cfg.model.mstar), CosmoTable.create()
    box = Box(11.0, 14.0, -4.0, -1.0)
    field = mocks.mock_field(rng, rs, box, density=4000, depth5=DEPTH5, mag_range=(14.0, 22.0))
    cl = mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, 12.5, -2.5, 0.3, 30.0, DEPTH5, poisson=False,
                            central_dmag=-1.5)
    gal = mocks.concat(field, cl)
    n = gal["RA"].size
    gal["ID"] = np.arange(n, dtype=np.int64)
    gal["ZSPEC"] = np.full(n, -1.0, np.float32)
    rows = field["RA"].size + np.argsort(cl["REFMAG"])[:8]
    gal["ZSPEC"][rows] = 0.3 + 1.3 * rng.normal(0.0, 600.0, rows.size) / 299792.458
    (tmp_path / "gal").mkdir()
    write_table(tmp_path / "gal" / SWEEP_NAME, gal, header={"SURVHASH": survey_hash(cfg)}, extname="GALAXIES")
    (tmp_path / "randoms").mkdir()
    write_randoms(tmp_path / "randoms" / "randoms-south-1-0.fits", rng, Box(10.0, 15.0, -5.0, 0.0),
                  density=4000.0)
    Calibration(rs=rs, config=cfg).write(tmp_path / "cal.fits")
    Table({"RA": [12.5, 11.6], "DEC": [-2.5, -1.6]}).write(tmp_path / "pos.fits")
    run("scan-positions", tmp_path / "pos.fits", "--dr11", tmp_path, "--calib", tmp_path / "cal.fits",
        "--galaxies", tmp_path / "gal", "--work", tmp_path / "work", "--nrand", 1, "--out", tmp_path / "out.fits")
    cat, mem, _ = read_catalog(tmp_path / "out.fits")
    assert cat["SCANNED"].all()
    assert cat["LAMBDA"][0] > 15 and abs(cat["Z_LAMBDA"][0] - 0.3) < 0.03
    assert cat["LAMBDA"][1] < cat["LAMBDA"][0]
    assert cat["NSPEC"][0] >= 3 and abs(cat["BEST_Z"][0] - 0.3) < 0.01
    assert set(np.unique(mem["MEM_MATCH_ID"])) <= {0, 1}
    assert not (tmp_path / "work" / "galaxies").exists()
