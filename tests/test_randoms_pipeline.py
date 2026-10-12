"""Cluster randoms end to end on a synthetic survey: two parts, a masked disc and a shallow strip.

The mock catalogue is drawn from the selection the randoms model (clusters uniform on the sky,
kept where LAMBDA / SCALEVAL >= 3 and MASKFRAC < 0.2 at their position), so a volume-limited
sample and its weighted randoms must have the same angular distribution: across the depth step,
around the masked disc and in every redshift bin.
"""

import json

import numpy as np
import pytest

from rema import cli
from rema.abundance.area import ZvlimMap
from rema.config import RemaConfig
from rema.io.tables import read_table, write_catalog, write_table
from rema.pipeline import write_plan
from rema.randoms.aperture import aperture_values, footprint_map
from rema.randoms.domain import RunDomain
from rema.sky.maps import Footprint, depth_ivar_from_mag, good_randoms

DEPTH5 = (24.9, 24.7, 24.2, 23.6)
DISC = (11.0, -1.0, 0.35)                 # masked disc: RA, Dec, radius [deg]
SHALLOW_DEC = 1.2                         # z-band 5 sigma depth 21.3 mag above this Dec


@pytest.fixture(autouse=True)
def no_jax_setup(monkeypatch):
    monkeypatch.setattr(cli, "_setup_jax", lambda: None)


def run(*argv):
    assert cli.main([str(a) for a in argv]) == 0


def _randoms(rng, n):
    ra = rng.uniform(10.0, 14.0, n)
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-2.0)), np.sin(np.radians(2.0)), n)))
    sep = np.hypot((ra - DISC[0]) * np.cos(np.radians(dec)), dec - DISC[1])
    cols = {"RA": ra, "DEC": dec, "MASKBITS": np.where(sep < DISC[2], 2, 0).astype(np.int16),
            "EBV": np.full(n, 0.02, np.float32)}
    for b, d in zip("GRIZ", DEPTH5):
        cols[f"NOBS_{b}"] = np.full(n, 3, np.int16)
        cols[f"GALDEPTH_{b}"] = np.full(n, depth_ivar_from_mag(d), np.float32)
    cols["GALDEPTH_Z"] = np.where(dec > SHALLOW_DEC, depth_ivar_from_mag(21.3),
                                  cols["GALDEPTH_Z"]).astype(np.float32)
    return cols


def _run_dir(path, boxes, ids, missing=()):
    path.mkdir()
    plan = {"REGION_ID": np.asarray(ids, np.int32)}
    for i, a in enumerate(("RA0", "RA1", "DEC0", "DEC1")):
        plan[f"OWN_{a}"] = np.array([b[i] for b in boxes], np.float64)
        plan[f"DATA_{a}"] = plan[f"OWN_{a}"] + (-0.5 if i % 2 == 0 else 0.5)
    write_plan(path / "regions.fits", plan, {"NREGION": len(ids)})
    (path / "clusters_dr11_qa.json").write_text(json.dumps({"missing": list(missing), "glat_min": 15.0}))
    return path


@pytest.fixture(scope="module")
def survey(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cr")
    rng = np.random.default_rng(2026)
    rnd = _randoms(rng, 400_000)
    write_table(tmp / "randoms-south-1-0.fits", rnd)
    # Part 0: RA 10-12 in two regions; part 1: RA 12-14 in two regions, one of which failed.
    r0 = _run_dir(tmp / "part0", [(10, 12, -2, 0), (10, 12, 0, 2)], [0, 1])
    r1 = _run_dir(tmp / "part1", [(12, 14, -2, 0), (12, 14, 0, 2)], [0, 1], missing=[1])
    # Mock catalogue: clusters at uniform positions of the run area, cut as the finder cuts.
    cfg = RemaConfig()
    pos = _randoms(rng, 200_000)
    ok = good_randoms(pos, cfg) & RunDomain.from_runs([r0, r1]).contains(pos["RA"], pos["DEC"])
    ra, dec = pos["RA"][ok][:12_000], pos["DEC"][ok][:12_000]
    n = ra.size
    zraw = rng.uniform(0.08, 0.72, n)
    lam = 10.0 * (1.0 - rng.uniform(size=n)) ** (-1 / 1.6)              # dN/dlam ~ lam^-2.6
    return dict(tmp=tmp, cfg=cfg, ra=ra, dec=dec, zraw=zraw, lam=lam, runs=(r0, r1))


@pytest.fixture(scope="module")
def products(survey):
    s, tmp, cfg = survey, survey["tmp"], survey["cfg"]
    mp = pytest.MonkeyPatch()
    mp.setattr(cli, "_setup_jax", lambda: None)
    rnd = tmp / "randoms-south-1-0.fits"
    run("maps", rnd, "--out", tmp / "fp.fits")
    fp = Footprint.read(tmp / "fp.fits")
    ap = aperture_values(s["ra"], s["dec"], s["zraw"], s["lam"], footprint_map(fp, "z"), cfg)
    keep = (s["lam"] / ap["SCALEVAL"] >= 3) & (ap["MASKFRAC"] < 0.2)
    cat = {"RA": s["ra"][keep], "DEC": s["dec"][keep], "LAMBDA": s["lam"][keep],
           "Z_LAMBDA_RAW": s["zraw"][keep], "Z_LAMBDA": s["zraw"][keep] + 0.002,
           "SCALEVAL": ap["SCALEVAL"][keep], "MASKFRAC": ap["MASKFRAC"][keep],
           "MEM_MATCH_ID": np.arange(keep.sum(), dtype=np.int64)}
    write_catalog(tmp / "clusters.fits", cat, None, cfg)
    (tmp / "spec.yaml").write_text("name: test\nz_edges: [0.1, 0.3, 0.5, 0.7]\nobs_edges: [10, 20, 1000]\n"
                                   "n_samp: 40\n")
    run("cluster-randoms", "domain", "--runs", *s["runs"], "--out", tmp / "domain.fits")
    run("cluster-randoms", "vlim-map", "--footprint", tmp / "fp.fits", "--domain", tmp / "domain.fits",
        "--catalog", tmp / "clusters.fits", rnd, "--density", 25_000, "--out", tmp / "zvlim.fits")
    common = ["--clusters", tmp / "clusters.fits", "--spec", tmp / "spec.yaml", "--zvlim",
              tmp / "zvlim.fits", "--domain", tmp / "domain.fits"]
    run("cluster-randoms", "copies", *common, rnd, "--with-id", "--out", tmp / "copies.fits")
    run("cluster-randoms", "place", "--copies", tmp / "copies.fits", "--footprint", tmp / "fp.fits")
    run("cluster-randoms", "sample", *common, "--copies", tmp / "copies.fits", "--keep", "MEM_MATCH_ID",
        "--out", tmp / "sample.fits")
    mp.undo()
    return dict(data=read_table(tmp / "sample.fits", hdu="CLUSTERS"),
                rand=read_table(tmp / "sample.fits", hdu="RANDOMS"),
                src=read_table(tmp / "sample.fits", hdu="SOURCES"),
                zmap=ZvlimMap.read(tmp / "zvlim.fits"), cat=cat, tmp=tmp)


def test_zvlim_map(products):
    zm = products["zmap"]
    assert zm.meta["LSTAR"] == 0.2 and zm.meta["NSIGMA"] == 5.0 and zm.meta["BAND"] == "Z"
    # Area: 12 deg2 of finished regions minus the masked disc (and the empty strip edges).
    disc = np.pi * DISC[2] ** 2
    assert zm.area.sum() == pytest.approx(12.0 - disc, rel=0.02)
    ra, dec = zm.radec()
    deep, shallow = zm.zvlim[dec < SHALLOW_DEC - 0.1], zm.zvlim[dec > SHALLOW_DEC + 0.1]
    assert np.ptp(deep) < 1e-6 and np.ptp(shallow) < 1e-6 and shallow[0] < 0.5 < deep[0]
    # Nothing in the failed region (RA 12-14, Dec > 0).
    assert not np.any((ra > 12.05) & (dec > 0.05))


def test_sample_weights(products):
    d, r, s = products["data"], products["rand"], products["src"]
    assert np.all(d["Z_LAMBDA"] < d["ZVLIM"]) and np.all(r["Z"] < r["ZVLIM"])
    # Every cluster's copies carry its n_samp; the weighted randoms have the data's z and lambda.
    sumw = np.bincount(np.searchsorted(s["ROW"], r["ROW"]), weights=r["WEIGHT"], minlength=s["ROW"].size)
    np.testing.assert_allclose(sumw, s["N_SAMP"], rtol=1e-10)
    assert np.all(s["N_SAMP"] == 40) and np.all(s["SUMQ"] > 0)
    for col, bins in (("Z_LAMBDA", np.linspace(0.1, 0.7, 13)), ("LAMBDA", np.geomspace(10, 200, 9))):
        hd = np.histogram(d[col], bins)[0]
        hr = np.histogram(r["Z" if col == "Z_LAMBDA" else col], bins, weights=r["WEIGHT"])[0]
        np.testing.assert_allclose(hr, 40 * hd, rtol=1e-9)
    # Copies: a random point near the disc can fail MASKFRAC; Q = 0 copies are dropped.
    assert np.all(r["Q"] == 1) and np.all(r["MASKFRAC"] < 0.2)


def _frac(sel_d, sel_r, d, r):
    fd = sel_d.sum() / d.size
    fr = np.sum(r["WEIGHT"][sel_r]) / np.sum(r["WEIGHT"])
    err = np.sqrt(fd * (1 - fd) / d.size)
    return fd, fr, err


def test_angular_distribution_follows_the_data(products):
    """Data and weighted randoms in the shallow strip and around the disc, per z bin."""
    d, r = products["data"], products["rand"]
    for k in range(3):
        dk, rk = d["BIN_Z"] == k, r["BIN_Z"] == k
        dd, rr = {c: v[dk] for c, v in d.items()}, {c: v[rk] for c, v in r.items()}
        strip_d, strip_r = dd["DEC"] > SHALLOW_DEC, rr["DEC"] > SHALLOW_DEC
        fd, fr, err = _frac(strip_d, strip_r, dd["DEC"], rr)
        assert abs(fd - fr) < 4 * err + 0.005, (k, fd, fr, err)
        near = lambda t: np.hypot((t["RA"] - DISC[0]) * np.cos(np.radians(t["DEC"])), t["DEC"] - DISC[1]) < 0.7
        fd, fr, err = _frac(near(dd), near(rr), dd["DEC"], rr)
        assert abs(fd - fr) < 4 * err + 0.005, (k, fd, fr, err)
    # Above the strip's z_vlim neither the data nor the randoms are there (but in the pixels on
    # its edge, whose median depth is the deep one).
    edge = SHALLOW_DEC + 0.06
    assert not np.any(d["DEC"][d["BIN_Z"] == 2] > edge) and not np.any(r["DEC"][r["BIN_Z"] == 2] > edge)


def test_a_sub_sample_reuses_the_copies(products, tmp_path, monkeypatch):
    """A higher threshold with lambda / S >= its lower edge, from the same placed copies."""
    monkeypatch.setattr(cli, "_setup_jax", lambda: None)
    tmp = products["tmp"]
    (tmp_path / "hi.yaml").write_text("name: hi\nz_edges: [0.1, 0.5]\nobs_edges: [20, 1000]\n"
                                      "detection_from_bin: true\n")
    run("cluster-randoms", "sample", "--clusters", tmp / "clusters.fits", "--spec", tmp_path / "hi.yaml",
        "--zvlim", tmp / "zvlim.fits", "--domain", tmp / "domain.fits", "--copies", tmp / "copies.fits",
        "--out", tmp_path / "hi.fits")
    d, r = read_table(tmp_path / "hi.fits", hdu="CLUSTERS"), read_table(tmp_path / "hi.fits", hdu="RANDOMS")
    assert np.all(d["LAMBDA"] / d["SCALEVAL"] >= 20) and np.all(r["LAMBDA"] / r["SCALEVAL"] >= 20)
    assert r["WEIGHT"].sum() == pytest.approx(40 * d["LAMBDA"].size)
    # Clusters outside the copies' sources are an error.
    (tmp_path / "lo.yaml").write_text("name: lo\nz_edges: [0.08, 0.5]\nobs_edges: [10, 1000]\n")
    with pytest.raises(ValueError, match="no copies"):
        run("cluster-randoms", "sample", "--clusters", tmp / "clusters.fits", "--spec", tmp_path / "lo.yaml",
            "--zvlim", tmp / "zvlim.fits", "--domain", tmp / "domain.fits", "--copies",
            tmp / "copies.fits", "--out", tmp_path / "lo.fits")


# --------------------------------------------------------------------------- external selection
from rema.randoms.selection import ExternalSelection, register  # noqa: E402


def p_ra(ra):
    """Toy external survey: detection probability rising across the field in RA."""
    return 0.1 + 0.8 * (np.asarray(ra) - 10.0) / 4.0


@register("toy_ra")
class ToyRA(ExternalSelection):
    columns = ("SNR",)

    def __init__(self, snr_min=0.0):
        self.snr_min = snr_min

    def in_footprint(self, ra, dec):
        return np.asarray(dec) > -1.9                    # the survey misses the southern edge

    def own_cut(self, cat):
        return np.asarray(cat["SNR"]) >= self.snr_min

    def p_detect(self, src, ra, dec):
        assert set(src) >= {"SNR", "Z", "LAMBDA"}
        return p_ra(ra)


def test_external_selection(products, tmp_path, monkeypatch):
    """Clusters kept with the toy survey's probability: the weighted randoms follow it, from a
    randoms index this time."""
    monkeypatch.setattr(cli, "_setup_jax", lambda: None)
    tmp = products["tmp"]
    cat = dict(products["cat"])
    rng = np.random.default_rng(11)
    det = (rng.uniform(size=cat["RA"].size) < p_ra(cat["RA"])) & (cat["DEC"] > -1.9)
    cat = {k: v[det] for k, v in cat.items()}
    cat["SNR"] = np.where(rng.uniform(size=cat["RA"].size) < 0.1, 3.0, 8.0)
    write_catalog(tmp_path / "ext.fits", cat, None, RemaConfig())
    (tmp_path / "ext.yaml").write_text("name: ext\nz_edges: [0.1, 0.45]\nobs_edges: [10, 1000]\nn_samp: 30\n"
                                       "selection: {name: toy_ra, snr_min: 5}\n")
    run("randoms-index", tmp / "randoms-south-1-0.fits", "--outdir", tmp_path / "idx")
    common = ["--clusters", tmp_path / "ext.fits", "--spec", tmp_path / "ext.yaml", "--zvlim",
              tmp / "zvlim.fits", "--domain", tmp / "domain.fits"]
    run("cluster-randoms", "copies", *common, "--index", tmp_path / "idx", "--nrand", 1,
        "--out", tmp_path / "c.fits")
    run("cluster-randoms", "place", "--copies", tmp_path / "c.fits", "--footprint", tmp / "fp.fits",
        "--out", tmp_path / "cp.fits")
    run("cluster-randoms", "sample", *common, "--copies", tmp_path / "cp.fits", "--out", tmp_path / "s.fits")
    d, r = read_table(tmp_path / "s.fits", hdu="CLUSTERS"), read_table(tmp_path / "s.fits", hdu="RANDOMS")
    assert np.all(d["SNR"] >= 5) and np.all(r["DEC"] > -1.9) and np.all(d["DEC"] > -1.9)
    np.testing.assert_allclose(r["P_EXT"], p_ra(r["RA"]), rtol=1e-6)
    edges = np.linspace(10, 14, 5)
    fd = np.histogram(d["RA"], edges)[0] / d["RA"].size
    fr = np.histogram(r["RA"], edges, weights=r["WEIGHT"])[0] / r["WEIGHT"].sum()
    err = np.sqrt(fd * (1 - fd) / d["RA"].size)
    assert np.all(np.abs(fd - fr) < 4 * err + 0.005), (fd, fr, err)
    # Without the external selection the randoms would follow the area alone (half of it at
    # RA > 12, where a region failed): the data are far from that.
    assert fd[-1] / fd[0] > 1.6 and fr[-1] / fr[0] > 1.6
    from astropy.io import fits

    assert fits.getheader(tmp_path / "s.fits", "RANDOMS")["SELNAME"] == "toy_ra"


def test_cli_errors(products, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_setup_jax", lambda: None)
    tmp = products["tmp"]
    common = ["--clusters", tmp / "clusters.fits", "--zvlim", tmp / "zvlim.fits", "--domain", tmp / "domain.fits"]
    with pytest.raises(SystemExit, match="randoms"):
        run("cluster-randoms", "copies", *common, "--out", tmp_path / "x.fits")
    (tmp_path / "bad.yaml").write_text("name: bad\ncuts: ['NOPE >= 1']\n")
    with pytest.raises(SystemExit, match="missing columns"):
        run("cluster-randoms", "copies", *common, "--spec", tmp_path / "bad.yaml", tmp / "randoms-south-1-0.fits",
            "--out", tmp_path / "x.fits")
    from rema.randoms.products import write_copies

    write_copies(tmp_path / "unplaced.fits", {"SRC": np.zeros(1, np.int64)}, {"ROW": np.zeros(1, np.int64)},
                 {}, "name: x\n", RemaConfig().to_yaml())
    with pytest.raises(SystemExit, match="place"):
        run("cluster-randoms", "sample", *common, "--copies", tmp_path / "unplaced.fits", "--out",
            tmp_path / "x.fits")


def test_qa(products, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_setup_jax", lambda: None)
    tmp = products["tmp"]
    run("cluster-randoms", "qa", "--sample", tmp / "sample.fits", "--zvlim", tmp / "zvlim.fits",
        "--domain", tmp / "domain.fits", "--nbins", 4, "--nside", 32, "--out", tmp_path / "qa.json")
    q = json.loads((tmp_path / "qa.json").read_text())
    assert {"ZVLIM", "DEPTH_Z", "AREA_FRAC", "ABS_GLAT", "SEAM_DIST"} <= set(q)
    for name in ("DEPTH_Z", "SEAM_DIST"):
        a = q[name]["all"]
        assert a["CHI2"] < 3 * a["NDOF"] + 10, (name, a)
    assert set(q["DEPTH_Z"]) == {"all", "z0", "z1", "z2"}


def test_external_join_and_erass1_model(tmp_path, monkeypatch):
    """external-join on a toy eRASS1 scan, then the approximate model from toy HEALPix maps."""
    import healpy as hp
    import yaml
    from astropy.io import fits

    monkeypatch.setattr(cli, "_setup_jax", lambda: None)
    rng = np.random.default_rng(13)
    n = 3000
    ra, dec = rng.uniform(200, 230, n), rng.uniform(-40, -20, n)
    t4 = 1000.0
    cr = np.exp(rng.normal(-2.5, 1.0, n))
    theta_r500 = rng.uniform(500, 1200, n)
    z = rng.uniform(0.1, 0.6, n)
    from astropy.cosmology import FlatLambdaCDM

    from rema.randoms.selection.erass1 import theta500_arcmin

    th = theta500_arcmin(theta_r500, z, FlatLambdaCDM(H0=70.0, Om0=0.3))
    lnl = -1.4 + 1.0 * np.log(cr * 0.27 * t4) - 0.4 * np.log(th) + rng.normal(0, 0.6, n)
    det = lnl >= np.log(3.0)
    orig = {"NAME": np.array([f"C{i}" for i in range(n)]), "EXT_LIKE": np.exp(lnl), "DET_LIKE_0": np.ones(n),
            "EXP": np.full(n, 0.27 * t4), "CR500": cr, "CTS500": cr * 270, "F500": cr * 5.0, "L500": cr * 1e44,
            "R500": theta_r500, "BEST_Z": z, "M500": np.ones(n), "PCONT": np.zeros(n)}
    write_table(tmp_path / "orig.fits", orig)
    rows = np.flatnonzero(det)
    scan = {"INPUT_ROW": rows, "INPUT_NAME": orig["NAME"][rows], "INPUT_RA": ra[rows], "INPUT_DEC": dec[rows],
            "LAMBDA": np.full(rows.size, 20.0), "PZ": np.zeros((rows.size, 3))}
    write_catalog(tmp_path / "scan.fits", scan, None, RemaConfig())
    run("cluster-randoms", "external-join", "--catalogue", "erass1", "--scan", tmp_path / "scan.fits",
        "--original", tmp_path / "orig.fits", "--out", tmp_path / "joined.fits")
    j = read_table(tmp_path / "joined.fits", hdu="CLUSTERS")
    np.testing.assert_allclose(j["EXT_CR500"], cr[rows])
    assert "PZ" not in j and fits.getheader(tmp_path / "joined.fits", "CLUSTERS")["EXTCAT"] == "erass1"
    npix = hp.nside2npix(32)
    fits.BinTableHDU.from_columns([fits.Column("TEXP", "E", array=np.full(npix, t4))],
                                  header=fits.Header({"ORDERING": "NESTED", "COORDSYS": "C"})).writeto(tmp_path / "exp.fits")
    fits.BinTableHDU.from_columns([fits.Column("NHI", "E", array=np.full(npix, 3e20))],
                                  header=fits.Header({"ORDERING": "RING", "COORDSYS": "GALACTIC"})).writeto(tmp_path / "nh.fits")
    run("cluster-randoms", "erass1-model", "--joined", tmp_path / "joined.fits", "--expmap", tmp_path / "exp.fits",
        "--exp-column", "TEXP", "--nhmap", tmp_path / "nh.fits", "--out-model", tmp_path / "m.yaml",
        "--out", tmp_path / "joined_nh.fits")
    m = yaml.safe_load((tmp_path / "m.yaml").read_text())
    assert m["exp_scale"] == pytest.approx(0.27, rel=1e-3) and m["c1"] == pytest.approx(1.0, abs=0.08)
    jn = read_table(tmp_path / "joined_nh.fits", hdu="CLUSTERS")
    np.testing.assert_allclose(jn["EXT_NH"], 3e20, rtol=1e-6)
    np.testing.assert_allclose(jn["EXT_THETA500"], th[rows], rtol=1e-6)
