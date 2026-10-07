"""Command line, run in-process on small synthetic inputs: ingest, randoms-index, maps,
calib-import, zred, background, scan, specpost, blind, regions, status, todo and merge."""

import json
import logging
import os

import jax
import numpy as np
import pytest
from astropy.io import fits

from rema import cli
from rema.calibration import Calibration
from rema.config import RemaConfig
from rema.io.legacy import survey_hash
from rema.io.tables import read_catalog, read_header, read_table, write_catalog, write_table
from rema.model.cosmo import CosmoTable
from rema.model.profiles import MStar
from rema.model.redsequence import RSModel
from rema.pipeline import file_sha1, plan_boxes, read_plan
from rema.sky.maps import Footprint, build_footprint, depth_ivar_from_mag, read_randoms
from rema.sky.regions import Box, sky_header, sweep_box
from rema.validate import mocks

from test_legacy_io import REJECT, fake_dr11

BOX = Box(10.0, 10.6, -0.3, 0.3)            # mock field
CLUSTER = (10.3, 0.0, 0.30, 30.0)           # RA, Dec, z, lambda
DEPTH5 = np.array([24.9, 24.7, 24.2, 23.6])


SETUP_JAX = cli._setup_jax


@pytest.fixture(autouse=True, scope="module")
def no_jax_setup():
    """main() points JAX's persistent compilation cache at ~/.cache/rema/jax; tests leave the
    process-wide JAX configuration alone (test_setup_jax checks that function)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "_setup_jax", lambda: None)
        yield


def run(*argv):
    assert cli.main([str(a) for a in argv]) == 0


def test_setup_jax(monkeypatch, tmp_path):
    old = (jax.config.jax_compilation_cache_dir, jax.config.jax_persistent_cache_min_compile_time_secs)
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "jax"))
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    try:
        SETUP_JAX()
        assert jax.config.jax_compilation_cache_dir == str(tmp_path / "jax")
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 1.0
        assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    finally:
        jax.config.update("jax_compilation_cache_dir", old[0])
        jax.config.update("jax_persistent_cache_min_compile_time_secs", old[1])


def test_array_spec():
    assert cli.array_spec([]) == ""
    assert cli.array_spec([5, 0, 2, 1, 2]) == "0-2,5"
    assert cli.array_spec([3, 7, 8, 9, 11]) == "3,7-9,11"


def test_sky_and_sweeps_helpers(tmp_path):
    assert cli._sky(None) is None and cli._box(None) is None
    assert cli._sky([["0", "5", "-5", "0"]]) == Box(0, 5, -5, 0)
    u = cli._sky([[0, 5, -5, 0], [5, 10, -5, 0]])
    assert u.area_deg2() == pytest.approx(2 * Box(0, 5, -5, 0).area_deg2())
    for n in ("sweep-000m005-005p000.fits", "sweep-005m005-010p000.fits"):
        (tmp_path / n).touch()
    (tmp_path / "sweep-000m005-005p000-pz.fits").touch()
    (tmp_path / "other.fits").touch()
    got = cli._sweeps([tmp_path, tmp_path / "x.fits"])
    assert [p.name for p in got] == ["sweep-000m005-005p000.fits", "sweep-005m005-010p000.fits", "x.fits"]


# --------------------------------------------------------------------------- ingest
def test_cli_ingest(tmp_path):
    rng = np.random.default_rng(51)
    names = ("sweep-010m005-015p000.fits", "sweep-015m005-020p000.fits")
    paths, _ = fake_dr11(tmp_path / "dr11", rng, names=names, n=40)
    sweeps = paths[0].parent
    ngal = 40 - len(REJECT)
    run("ingest", sweeps, "--out", tmp_path / "all.fits")
    assert read_table(tmp_path / "all.fits")["ID"].size == 2 * ngal
    run("ingest", sweeps, "--out", tmp_path / "box.fits", "--box", 16, 19, -4, -1)
    t = read_table(tmp_path / "box.fits")
    assert 0 < t["ID"].size < ngal and np.all(Box(16, 19, -4, -1).contains(t["RA"], t["DEC"]))
    assert read_header(tmp_path / "box.fits")["NSWEEPS"] == 2
    # Per-sweep tables, only of the sweeps overlapping the box.
    run("ingest", sweeps, "--outdir", tmp_path / "gal", "--box", 16, 19, -4, -1, "--require-pz")
    assert [p.name for p in sorted((tmp_path / "gal").iterdir())] == [names[1]]
    assert read_header(tmp_path / "gal" / names[1])["NGAL"] == ngal
    with pytest.raises(SystemExit, match="--out"):
        cli.main(["ingest", str(sweeps)])


# --------------------------------------------------------------------------- randoms and maps
def write_randoms(path, rng, box, density=20_000.0):
    """A randoms file over ``box`` (RA < 10.1 masked), with the columns of DR11 randoms."""
    n = int(density * box.area_deg2())
    ra = rng.uniform(box.ra_min, box.ra_max, n)
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(box.dec_min)),
                                           np.sin(np.radians(box.dec_max)), n)))
    cols = {"RA": ra, "DEC": dec, "MASKBITS": np.where(ra < 10.1, 2, 0).astype(np.int16),
            "EBV": np.full(n, 0.02, np.float32)}
    for b, d in zip("GRIZ", DEPTH5):
        cols[f"NOBS_{b}"] = np.full(n, 3, np.int16)
        cols[f"GALDEPTH_{b}"] = np.full(n, depth_ivar_from_mag(d), np.float32)
    return write_table(path, cols)


def test_cli_randoms_index_and_maps(tmp_path):
    rng = np.random.default_rng(52)
    big = Box(9.5, 11.0, -0.8, 0.8)
    rnd = write_randoms(tmp_path / "randoms-south-1-0.fits", rng, big)
    box = ("--box", 10, 10.6, -0.3, 0.3)
    run("maps", rnd, *box, "--out", tmp_path / "fp_direct.fits")
    run("randoms-index", rnd, "--outdir", tmp_path / "idx")
    assert {p.name for p in (tmp_path / "idx").iterdir()} >= {"randoms-south-1-0.json",
                                                            "randoms-south-1-0.npy",
                                                            "randoms-south-1-0.offsets.npy"}
    run("maps", "--index", tmp_path / "idx", "--nrand", 1, *box, "--out", tmp_path / "fp_index.fits")
    a, b = Footprint.read(tmp_path / "fp_direct.fits"), Footprint.read(tmp_path / "fp_index.fits")
    assert a.digest() == b.digest() and a.box == BOX
    # Area: the box minus the masked 0.1 deg strip.
    assert a.area_deg2() == pytest.approx(Box(10.1, 10.6, -0.3, 0.3).area_deg2(), rel=0.03)
    assert a.density == 2500.0
    with pytest.raises(SystemExit, match="randoms"):
        cli.main(["maps", "--out", str(tmp_path / "x.fits")])
    # The data box of a planned region gives the same map as --box.
    plan = mock_plan(tmp_path / "plan.fits")
    run("maps", rnd, "--regions", plan, "--region-id", 3, "--out", tmp_path / "fp_plan.fits")
    assert Footprint.read(tmp_path / "fp_plan.fits").digest() == a.digest()


def mock_plan(path, region_id=3):
    """A one-region plan whose own and data boxes are the mock field."""
    from rema.pipeline import write_plan

    plan = {"REGION_ID": np.array([region_id])}
    for pre in ("OWN", "DATA"):
        for k, v in zip(("RA0", "RA1", "DEC0", "DEC1"), BOX.as_tuple()):
            plan[f"{pre}_{k}"] = np.array([v])
    plan["PAIRS_EST"] = np.array([1.0])
    return write_plan(path, plan, {"NREGION": 1, "BUFFER": 0.0, "PLANHASH": "testplan0001"})


# --------------------------------------------------------------------------- calib-import
def write_pars(path, bands=("g", "r", "i", "z")):
    """A redMaPPer-style ``*_pars.fit`` file (reddest band as reference) with linear colours on
    different node sets, constant scatter and correlations 0.3."""
    ncol = len(bands) - 1
    zc = [np.linspace(0.05, 0.85, 9)] + [np.linspace(0.05, 0.85, 5)] * (ncol - 1)
    lines = [(1.0, 1.5), (0.3, 1.0), (0.2, 0.5)][:ncol]
    zs, zcov, zpiv = np.linspace(0.05, 0.85, 5), np.linspace(0.05, 0.85, 4), np.linspace(0.05, 0.85, 6)
    sigma = np.full((ncol, ncol, zcov.size), 0.3)
    for j, s in enumerate((0.05, 0.04, 0.03)[:ncol]):
        sigma[j, j] = s
    zcorr = np.linspace(0.05, 0.85, 7)
    row = {}
    for j in range(ncol):
        row[f"Z{j:02d}"], row[f"C{j:02d}"] = zc[j], lines[j][0] + lines[j][1] * zc[j]
        row[f"ZS{j:02d}"], row[f"SLOPE{j:02d}"] = zs, np.full(zs.size, -0.01 * (j + 1))
    row.update({"COVMAT_Z": zcov, "SIGMA": sigma, "PIVOTMAG_Z": zpiv, "PIVOTMAG": 17.0 + 5.0 * zpiv,
                "CORR": np.full(zcorr.size, 0.01), "CORR_SLOPE": np.zeros(zcorr.size),
                "CORR_R": np.full(zcorr.size, 1.1), "CORR_Z": zcorr, "CORR_SLOPE_Z": zcorr})
    cols = []
    for k, v in row.items():
        v = np.asarray(v, np.float64)
        dim = None if v.ndim == 1 else "(" + ",".join(str(s) for s in reversed(v.shape)) + ")"
        cols.append(fits.Column(name=k, format=f"{v.size}D", dim=dim, array=v[None]))
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header["NCOL"], hdu.header["BANDS"], hdu.header["REF_IND"] = ncol, ", ".join(bands), ncol
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    return path


def test_cli_calib_import(tmp_path):
    import jax.numpy as jnp

    pars = write_pars(tmp_path / "test_pars.fit")
    cfg = tmp_path / "cfg.yaml"
    RemaConfig().replace(spec={"nboot": 8}).to_yaml(cfg)
    run("calib-import", pars, "--out", tmp_path / "cal.fits", "--config", cfg)
    cal = Calibration.read(tmp_path / "cal.fits")
    assert cal.rs.bands == ("g", "r", "i", "z") and cal.rs.ref_band == "z"
    assert cal.config.spec.nboot == 8 and cal.meta["PARSFILE"] == "test_pars.fit"
    # Colour nodes: the union of the node sets; linear colours are kept exactly.
    assert len(cal.rs.z_mean) == 9
    zz = jnp.asarray([0.2, 0.5, 0.7])
    at = cal.rs.at(zz)
    expect = np.stack([1.0 + 1.5 * zz, 0.3 + zz, 0.2 + 0.5 * zz], 1)
    np.testing.assert_allclose(np.asarray(at.mean), expect, atol=1e-5)
    np.testing.assert_allclose(np.asarray(at.slope), [[-0.01, -0.02, -0.03]] * 3, atol=1e-6)
    np.testing.assert_allclose(np.asarray(at.pivot), 17.0 + 5.0 * np.asarray(zz), atol=1e-4)
    c = np.asarray(at.cint)
    sig = np.sqrt(np.diagonal(c, axis1=1, axis2=2))
    np.testing.assert_allclose(sig, [[0.05, 0.04, 0.03]] * 3, rtol=1e-4)
    np.testing.assert_allclose(c / (sig[:, :, None] * sig[:, None, :]),
                               np.broadcast_to(np.where(np.eye(3) > 0, 1.0, 0.3), c.shape), atol=1e-4)
    np.testing.assert_allclose(np.asarray(cal.zredcorr.corr_r), 1.1)
    np.testing.assert_allclose(np.asarray(cal.zredcorr.z_corr), np.linspace(0.05, 0.85, 7))


def test_two_colour_model_fits_roundtrip(tmp_path):
    """Regression: CORR of a two-colour model (three bands) is [n, 1] and must stay so in FITS."""
    import jax.numpy as jnp

    rs = RSModel.from_template(bands=("g", "r", "z"))
    back = Calibration.read(Calibration(rs=rs).write(tmp_path / "cal.fits")).rs
    for k in ("mean", "slope", "log_sigma", "corr", "pivot"):
        a, b = np.asarray(getattr(rs, k)), np.asarray(getattr(back, k))
        assert a.shape == b.shape, k
        np.testing.assert_allclose(b, a, rtol=1e-12)
    assert back.bands == ("g", "r", "z") and back.z_corr == rs.z_corr
    assert np.asarray(back.at(jnp.asarray(0.3)).cint).shape == (2, 2)
    zz = jnp.asarray([0.2, 0.5, 0.7])
    np.testing.assert_allclose(np.asarray(back.at(zz).cint), np.asarray(rs.at(zz).cint), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(back.at(zz).mean), np.asarray(rs.at(zz).mean), rtol=1e-6)


# --------------------------------------------------------------------------- zred, background, scan
@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    """A 0.36 deg^2 mock field with one cluster (spectroscopic redshifts for its 12 brightest
    members), its galaxy table, a template calibration and a footprint."""
    d = tmp_path_factory.mktemp("mock_run")
    rng = np.random.default_rng(53)
    cfg = RemaConfig()
    rs = RSModel.from_template()
    ms, cosmo = MStar(cfg.model.mstar), CosmoTable.create()
    field = mocks.mock_field(rng, rs, BOX, density=4000, depth5=DEPTH5, mag_range=(14.0, 22.0))
    ra, dec, z, lam = CLUSTER
    cl = mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, ra, dec, z, lam, DEPTH5, poisson=False,
                            central_dmag=-1.5)
    gal = mocks.concat(field, cl)
    n = gal["RA"].size
    gal["ID"] = np.arange(n, dtype=np.int64)
    gal["ZSPEC"] = np.full(n, -1.0, np.float32)
    nf = field["RA"].size
    rows = nf + np.argsort(cl["REFMAG"])[:12]
    gal["ZSPEC"][rows] = z + (1 + z) * rng.normal(0.0, 600.0, rows.size) / 299792.458
    write_table(d / "gal.fits", gal, header={"SURVHASH": survey_hash(cfg)}, extname="GALAXIES")
    Calibration(rs=rs, config=cfg).write(d / "cal0.fits")
    write_randoms(d / "randoms.fits", rng, Box(9.8, 10.8, -0.5, 0.5))
    build_footprint(read_randoms(d / "randoms.fits", cfg, BOX), cfg, box=BOX).write(d / "fp.fits")
    return {"dir": d, "ngal": n, "nfield": nf, "cluster_rows": nf + np.arange(cl["RA"].size)}


def test_cli_zred(mock_run):
    d = mock_run["dir"]
    run("zred", "--galaxies", d / "gal.fits", "--calib", d / "cal0.fits", "--out", d / "zred.fits")
    t = read_table(d / "zred.fits", hdu="ZRED")
    assert set(t) == {"ID", "ZRED", "ZRED_E", "ZRED_UNCORR", "ZRED_UNCORR_E", "ZRED_CHISQ"}
    assert t["ID"].size == mock_run["ngal"]
    # Red-sequence members at z = 0.3 have zred close to it (median of the cluster).
    zr = t["ZRED"][mock_run["cluster_rows"]]
    assert abs(np.median(zr) - CLUSTER[2]) < 0.03
    assert np.all(t["ZRED_E"][t["ZRED"] > 0] > 0)


def test_cli_calib_import_grz_then_zred(mock_run, tmp_path):
    """A g, r, z calibration imported from pars, read back and used on a griz galaxy table."""
    d = mock_run["dir"]
    pars = write_pars(tmp_path / "grz_pars.fit", bands=("g", "r", "z"))
    run("calib-import", pars, "--out", tmp_path / "grz.fits")
    cal = Calibration.read(tmp_path / "grz.fits")
    assert cal.rs.bands == ("g", "r", "z") and np.asarray(cal.rs.corr).shape == (4, 1)
    run("zred", "--galaxies", d / "gal.fits", "--calib", tmp_path / "grz.fits", "--out",
        tmp_path / "zred.fits")
    t = read_table(tmp_path / "zred.fits", hdu="ZRED")
    assert t["ID"].size == mock_run["ngal"]
    ok = t["ZRED"] > 0
    assert ok.mean() > 0.5 and np.all(np.isfinite(t["ZRED_CHISQ"][ok]))
    assert np.all((t["ZRED"][ok] > 0.0) & (t["ZRED"][ok] < 1.2))


@pytest.fixture(scope="module")
def calibrated(mock_run):
    """`rema background` of the mock run: the template calibration with both backgrounds."""
    d = mock_run["dir"]
    run("background", "--galaxies", d / "gal.fits", "--calib", d / "cal0.fits", "--footprint",
        d / "fp.fits", "--box", *BOX.as_tuple(), "--out", d / "cal.fits")
    return d / "cal.fits"


def test_cli_background(calibrated):
    cal = Calibration.read(calibrated)
    assert cal.bkg is not None and cal.zbkg is not None and cal.config == RemaConfig()
    sg = np.asarray(cal.bkg.sigma_g)
    assert np.isfinite(sg).any() and np.nanmax(sg[np.isfinite(sg)]) > 0
    with fits.open(calibrated) as h:
        assert {"BKG_CHISQ", "BKG_ZRED", "CONFIG"} <= {x.name for x in h}


def test_cli_scan_and_specpost(mock_run, calibrated, caplog):
    d = mock_run["dir"]
    # Scan at the cluster and at a field position; --box keeps only the first two.
    pos = {"RA": np.array([CLUSTER[0], 10.15, 20.0]), "DEC": np.array([CLUSTER[1], 0.2, 0.0]),
           "NAME_ID": np.array([7, 8, 9])}
    write_table(d / "pos.fits", pos)
    cfg = d / "scan.yaml"
    RemaConfig().replace(scan={"zrange": (0.15, 1.0), "zstep": 0.01}).to_yaml(cfg)
    run("scan", "--galaxies", d / "gal.fits", "--calib", d / "cal.fits", "--footprint", d / "fp.fits",
        "--positions", d / "pos.fits", "--id-col", "name_id", "--box", *BOX.as_tuple(),
        "--config", cfg, "--centering", "auto", "--batch", 2, "--specpost", "--out", d / "scan.fits")
    assert "scan.zrange, scan.zstep" in caplog.text     # differs from the stored configuration
    cat, mem, hdr = read_catalog(d / "scan.fits")
    assert hdr["MODE"] == "scan" and hdr["CENTRING"] == "bcg" and hdr["NCLUSTER"] == 2
    assert list(cat["MEM_MATCH_ID"]) == [7, 8]
    assert abs(cat["Z_LAMBDA_OPT"][0] - CLUSTER[2]) < 0.02
    assert 0.6 * CLUSTER[3] < cat["LAMBDA_OPT"][0] < 1.5 * CLUSTER[3]
    assert cat["LAMBDA_OPT"][1] < 0.5 * cat["LAMBDA_OPT"][0]
    assert 0.0 <= cat["MASKFRAC"][0] < 0.05 and cat["MASKFRAC"][1] > 0.05   # 10.15 is near the mask
    assert abs(cat["SPEC_Z_BOOT"][0] - CLUSTER[2]) < 0.005
    assert set(np.unique(mem["MEM_MATCH_ID"])) <= {7, 8}
    assert cli._catalog_cfg(d / "scan.fits").scan.zstep == 0.01

    # specpost of the scan catalogue: its stored configuration, same spectroscopic redshift.
    run("specpost", d / "scan.fits", "--out", d / "spec.fits")
    cat2, mem2, hdr2 = read_catalog(d / "spec.fits")
    assert hdr2["MODE"] == "specpost"
    np.testing.assert_allclose(cat2["SPEC_Z_BOOT"], cat["SPEC_Z_BOOT"])
    assert "ISMEMBER_SPEC" in mem2 and mem2["ID"].size == mem["ID"].size
    # An empty catalogue stays empty.
    write_catalog(d / "empty.fits", {}, {}, RemaConfig())
    run("specpost", d / "empty.fits", "--out", d / "empty_spec.fits")
    cat3, mem3, hdr3 = read_catalog(d / "empty_spec.fits")
    assert cat3 == {} and hdr3["NCLUSTER"] == 0


@pytest.mark.slow
def test_cli_blind(mock_run, caplog):
    d = mock_run["dir"]
    run("blind", "--galaxies", d / "gal.fits", "--calib", d / "cal0.fits", "--box", *BOX.as_tuple(),
        "--own", 10.0, 10.6, -0.3, 0.3, "--region-id", 3, "--specpost", "--checkpoint", d / "ck",
        "--out", d / "blind.fits")
    cat, mem, hdr = read_catalog(d / "blind.fits")
    assert hdr["MODE"] == "blind" and hdr["REGION"] == 3 and hdr["CALSHA1"] == file_sha1(d / "cal0.fits")
    assert hdr["OWNRA0"] == 10.0 and hdr["NGAL"] == mock_run["ngal"] and hdr["NSEED"] > 0
    assert {p.name for p in (d / "ck").iterdir()} == {"firstpass.npz", "likelihood.npz"}
    assert np.all((cat["MEM_MATCH_ID"] >> 32) == 3)
    i = int(np.argmax(cat["LAMBDA"]))
    assert np.hypot(cat["RA"][i] - CLUSTER[0], cat["DEC"][i] - CLUSTER[1]) < 0.02
    assert abs(cat["Z_LAMBDA"][i] - CLUSTER[2]) < 0.02 and "SPEC_Z_BOOT" in cat
    # The same region from a plan resumes from the checkpoint and gives the same catalogue.
    caplog.clear()
    caplog.set_level(logging.INFO)
    run("blind", "--galaxies", d / "gal.fits", "--calib", d / "cal0.fits", "--regions",
        mock_plan(d / "plan.fits"), "--region-id", 3, "--checkpoint", d / "ck", "--out", d / "blind2.fits")
    assert "likelihood: read from" in caplog.text
    cat2, _, hdr2 = read_catalog(d / "blind2.fits")
    assert hdr2["PLANHASH"] == "testplan0001" and hdr2["OWNDEC1"] == BOX.dec_max
    np.testing.assert_array_equal(cat2["MEM_MATCH_ID"], cat["MEM_MATCH_ID"])
    np.testing.assert_array_equal(cat2["ID_CENT"], cat["ID_CENT"])
    np.testing.assert_allclose(cat2["LAMBDA"], cat["LAMBDA"])
    np.testing.assert_allclose(cat2["Z_LAMBDA"], cat["Z_LAMBDA"])


def test_cli_remeasure(mock_run, calibrated):
    """Blind clusters re-measured at their centres: with the catalogue's free fractions the
    fiducial cosmology gives the catalogue back; a larger Omega_m a larger lambda."""
    from rema.modes import remeasure as R

    d = mock_run["dir"]
    common = ["--galaxies", d / "gal.fits", "--calib", calibrated, "--footprint", d / "fp.fits"]
    run("blind", *common, "--box", *BOX.as_tuple(), "--out", d / "blind_fp.fits")
    run("remeasure", *common, "--box", *BOX.as_tuple(), "--catalog", d / "blind_fp.fits",
        "--members", d / "blind_fp.fits", "--vary", "Omega_m=0.35", "--jvp", "Omega_m", "--fd",
        "--lambda-min", 10, "--out", d / "rem.fits")
    cat, labels, hdr = R.read_remeasure(d / "rem.fits")
    assert labels == ["fiducial", "Omega_m=0.35"] and hdr["PFREE"] == "members" and hdr["OMEGAM"] == 0.3
    assert cat["LAMBDA"].shape == (cat["LAMBDA_CAT"].size, 2) and cat["LAMBDA_CAT"].size >= 1
    np.testing.assert_allclose(cat["LAMBDA"][:, 0], cat["LAMBDA_CAT"], rtol=0.01)
    np.testing.assert_allclose(cat["Z_LAMBDA"][:, 0], cat["Z_LAMBDA_CAT"], atol=2e-3)
    np.testing.assert_allclose(cat["SCALEVAL"][:, 0], cat["SCALEVAL_CAT"], rtol=0.01)
    assert np.all(cat["LAMBDA"][:, 1] > cat["LAMBDA"][:, 0])
    assert np.all(cat["DLNLAMBDA_DOMEGA_M"] > 0)
    np.testing.assert_allclose(cat["DLNLAMBDA_DOMEGA_M"], cat["DLNLAMBDA_DOMEGA_M_FD"], rtol=0.3, atol=0.05)
    # Without members every neighbour is free: lambda can only grow.
    run("remeasure", *common, "--catalog", d / "blind_fp.fits", "--fixed-z", "--lambda-min", 10,
        "--max-clusters", 1, "--mstar-follows-cosmology", "--vary", "w0=-0.8", "--out", d / "rem1.fits")
    cat1, labels1, hdr1 = R.read_remeasure(d / "rem1.fits")
    assert hdr1["PFREE"] == "one" and hdr1["FIXEDZ"] and labels1 == ["fiducial", "w0=-0.8"]
    assert cat1["LAMBDA"].shape[0] == 1
    with pytest.raises(ValueError, match="KEY=V1"):
        run("remeasure", *common, "--catalog", d / "blind_fp.fits", "--vary", "s8=1", "--out", d / "x.fits")


# --------------------------------------------------------------------------- large runs
SWEEPS = ["sweep-000m005-005p000.fits", "sweep-005m005-010p000.fits", "sweep-000m010-005m005.fits",
          "sweep-005m010-010m005.fits"]


def per_sweep_tables(outdir, rng, n=200):
    """Per-sweep galaxy tables with the headers `rema ingest --outdir` writes."""
    for i, s in enumerate(SWEEPS):
        b = sweep_box(s)
        t = {"ID": np.arange(n, dtype=np.int64) + 1000 * i, "RA": rng.uniform(b.ra_min, b.ra_max, n),
             "DEC": rng.uniform(b.dec_min, b.dec_max, n), "ZSPEC": np.full(n, -1.0, np.float32)}
        write_table(outdir / s, t, header={"NZSPEC": 10 * (i + 1), "EBVMEAN": 0.02, **sky_header(b)},
                    extname="GALAXIES")


def fake_index(index_dir, tiles, nside=16):
    """A randoms index (offsets and json only) with randoms in the given 5 deg tiles."""
    import healpy as hp

    index_dir.mkdir(parents=True, exist_ok=True)
    ra, dec = hp.pix2ang(nside, np.arange(12 * nside**2), nest=True, lonlat=True)
    has = np.zeros(ra.size, bool)
    for r0, d0 in tiles:
        has |= Box(r0, r0 + 5, d0, d0 + 5).contains(ra, dec)
    off = np.concatenate([[0], np.cumsum(has)])
    np.save(index_dir / "randoms-south-1-0.offsets.npy", off)
    (index_dir / "randoms-south-1-0.json").write_text(json.dumps({"nside_index": nside}))


def test_cli_regions_status_merge_todo(tmp_path, capsys, caplog):
    rng = np.random.default_rng(54)
    gal = tmp_path / "galaxies"
    per_sweep_tables(gal, rng)
    plan_path = tmp_path / "plan.fits"
    run("regions", "--galaxies", gal, "--target-area", 25, "--band-height", 5, "--buffer", 1.0,
        "--out", plan_path)
    plan, meta = read_plan(plan_path)
    assert meta["NREGION"] == len(plan["REGION_ID"]) == 4
    assert np.all(plan["NGAL_DATA"] >= 200)
    # A box restricts the plan to the sweeps it overlaps.
    run("regions", "--galaxies", gal, "--box", 0, 5, -10, 0, "--target-area", 25, "--band-height", 5,
        "--out", tmp_path / "plan_box.fits")
    assert read_plan(tmp_path / "plan_box.fits")[1]["NREGION"] == 2
    # Calibration areas instead of a plan.
    capsys.readouterr()
    run("regions", "--galaxies", gal, "--calib-suggest", 100)
    out = capsys.readouterr().out
    assert out.startswith("calib_box=0 10 -10 0 nzspec=100 ngal=800")
    # Tiles with randoms but no galaxy table: an error unless allowed.
    fake_index(tmp_path / "idx", [(0, -5), (5, -5), (0, -10), (5, -10), (10, -5)])
    with pytest.raises(SystemExit, match="sweep-010m005-015p000.fits"):
        cli.main(["regions", "--galaxies", str(gal), "--index", str(tmp_path / "idx"),
                  "--out", str(tmp_path / "p2.fits")])
    run("regions", "--galaxies", gal, "--index", tmp_path / "idx", "--allow-uncovered",
        "--target-area", 25, "--band-height", 5, "--buffer", 1.0, "--out", tmp_path / "p2.fits")
    # --sweeps: a tile with no sweep in the survey is outside its data; a listed one is missing.
    sdir = tmp_path / "survey"
    sdir.mkdir()
    (sdir / "sweep-000m005-005p000.fits").touch()
    (sdir / "sweep-000m005-005p000-pz.fits").touch()
    run("regions", "--galaxies", gal, "--index", tmp_path / "idx", "--sweeps", sdir,
        "--target-area", 25, "--band-height", 5, "--buffer", 1.0, "--out", tmp_path / "p3.fits")
    (sdir / "sums.sha256sum").write_text("abc  sweep-010m005-015p000.fits\nxyz  sweep-010m005-015p000-pz.fits\n"
                                         "nothing to see\n")
    with pytest.raises(SystemExit, match="sweep-010m005-015p000.fits"):
        cli.main(["regions", "--galaxies", str(gal), "--index", str(tmp_path / "idx"), "--sweeps", str(sdir),
                  "--out", str(tmp_path / "p4.fits")])

    # maps of a planned region needs its id.
    with pytest.raises(SystemExit, match="region-id"):
        cli.main(["maps", "--regions", str(plan_path), "--out", str(tmp_path / "fp.fits")])

    # Region catalogues: all but the last region done, the last one missing.
    runs = tmp_path / "runs"
    calib = tmp_path / "cal.fits"
    calib.write_bytes(b"calibration")
    rids = [int(r) for r in plan["REGION_ID"]]
    rcfg = RemaConfig().replace(cosmology={"Omega_m": 0.28})
    for k, r in enumerate(rids[:-1]):
        own, _ = plan_boxes(plan, r, meta)
        m = 3
        cat = {"MEM_MATCH_ID": (np.int64(r) << 32) | np.arange(m, dtype=np.int64),
               "RA": np.full(m, 0.5 * (own.ra_min + own.ra_max)) + np.arange(m) * 0.5,
               "DEC": np.full(m, 0.5 * (own.dec_min + own.dec_max)), "Z_LAMBDA": np.full(m, 0.3),
               "LAMBDA": np.full(m, 20.0)}
        mem = {"MEM_MATCH_ID": np.repeat(cat["MEM_MATCH_ID"], 2), "ID": np.arange(2 * m) + 100 * k,
               "PMEM": np.full(2 * m, 0.7)}
        write_catalog(runs / f"{r:04d}" / "clusters.fits", cat, mem, rcfg,
                      {"PLANHASH": meta["PLANHASH"], "CALSHA1": file_sha1(calib),
                       "CFGSHA1": "a" if k == 0 else "b"})
    capsys.readouterr()
    run("status", "--plan", plan_path, "--runs", runs, "--calib", calib, "--check-version")
    lines = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines() if "=" in line
                 and not line.startswith("done"))
    assert lines["todo_ids"] == str(rids[-1]) and lines["prime"] == str(rids[-1])
    assert lines["todo_array_noprime"] == "" and lines["stale_ids"] == ""

    out = tmp_path / "merged" / "clusters.fits"
    with pytest.raises(RuntimeError, match="missing"):
        cli.main(["merge", "--plan", str(plan_path), "--runs", str(runs), "--out", str(out)])
    caplog.clear()
    run("merge", "--plan", plan_path, "--runs", runs, "--out", out, "--calib", calib, "--allow-missing")
    cat, mem, hdr = read_catalog(out)
    assert hdr["MODE"] == "merged" and hdr["NREGION"] == 3 and len(cat["RA"]) == 9 and mem == {}
    # The regions' configuration and cosmology are kept (with a warning when they differ).
    assert hdr["OMEGAM"] == 0.28 and cli._catalog_cfg(out) == rcfg
    assert "made with 2 configurations" in caplog.text
    members = read_table(out.with_name("clusters_members.fits"), hdu="MEMBERS")
    assert members["ID"].size == 18
    regions = read_table(out.with_name("clusters_regions.fits"), hdu="REGIONS")
    assert sorted(regions["REGION_ID"]) == sorted(rids[:-1])
    qa = json.loads(out.with_name("clusters_qa.json").read_text())
    assert qa["n_clusters"] == 9 and qa["missing"] == [rids[-1]] and qa["duplicate_ids"] == 0
    assert qa["members_without_cluster"] == 0 and len(qa["edge_profile"]["n"]) > 0
    # --glat-min: these mock clusters lie at |b| of about 60 deg.
    out2 = tmp_path / "merged" / "clusters_b70.fits"
    run("merge", "--plan", plan_path, "--runs", runs, "--out", out2, "--calib", calib, "--allow-missing",
        "--glat-min", 70)
    qa2 = json.loads(out2.with_name("clusters_b70_qa.json").read_text())
    assert qa2["glat_min"] == 70 and qa2["n_clusters_low_glat"] == 9 and qa2["n_clusters"] == 0
    assert qa2["n_members"] == 0                            # counted after the cut, like the file
    assert read_catalog(out2, members=False)[2]["GLATMIN"] == 70
    assert read_table(out2.with_name("clusters_b70_members.fits"), hdu="MEMBERS")["ID"].size == 0

    # todo: chunks of sweeps without a galaxy table, randoms files without an index.
    sweeps = tmp_path / "sweeps"
    sweeps.mkdir()
    for s in SWEEPS + ["sweep-010m005-015p000.fits", "sweep-010m005-015p000-pz.fits"]:
        (sweeps / s).touch()
    capsys.readouterr()
    run("todo", "--sweeps", sweeps, "--galaxies", gal, "--index", tmp_path / "idx", "--chunk", 2,
        "--nrand", 3)
    out = capsys.readouterr().out.splitlines()
    assert out == ["nsweeps=5 nchunks=3", "ingest_array=2", "randoms_array=1-2", "ingest_done=4/5",
                   "randoms_done=1/3"]


def test_cli_warnings(mock_run, calibrated, tmp_path, caplog):
    from types import SimpleNamespace

    caplog.set_level(logging.INFO)
    # A calibration with a wcen model but BCG centring selected.
    reg = SimpleNamespace(centering_method=lambda: "bcg", wcen=object())
    assert cli._centring(reg) == "bcg" and "selects BCG centring" in caplog.text
    # A footprint built for another box than the data box.
    d = mock_run["dir"]
    run("background", "--galaxies", d / "gal.fits", "--calib", calibrated, "--footprint", d / "fp.fits",
        "--box", 10.0, 10.5, -0.3, 0.3, "--cosmology", "Omega_m=0.35,w0=-0.9", "--out", tmp_path / "cal.fits")
    assert "the footprint was built for" in caplog.text
    # --cosmology: the run's configuration, and the calibration's header.
    assert "cosmology: Omega_m=0.35,w0=-0.9" in caplog.text
    assert Calibration.read(tmp_path / "cal.fits").config.cosmology.w0 == -0.9
    assert fits.getheader(tmp_path / "cal.fits")["OMEGAM"] == 0.35
    # --config for a catalogue that stores none.
    write_catalog(tmp_path / "nocfg.fits", {}, {}, None)
    RemaConfig().replace(spec={"nboot": 4}).to_yaml(tmp_path / "c.yaml")
    run("specpost", tmp_path / "nocfg.fits", "--config", tmp_path / "c.yaml", "--out", tmp_path / "s.fits")
    assert cli._catalog_cfg(tmp_path / "s.fits").spec.nboot == 4
