"""rema grid-points / grid-lambda: the nested HEALPix grid of the scan-mode randoms, and lambda(z) at
fixed positions and redshifts (mock field and cluster, synthetic DR11 files)."""

import healpy as hp
import numpy as np
import pytest

from rema import cli
from rema.config import RemaConfig
from rema.core.richness import RadialQuad, Stage
from rema.io.tables import read_table
from rema.model.cosmo import CosmoTable
from rema.model.profiles import MStar
from rema.model.redsequence import RSModel
from rema.modes.common import Region
from rema.modes.grid_lambda import grid_points, grid_zsteps, level_offsets, level_pixels, run_grid
from rema.modes.scan import _refine
from rema.randoms.domain import RunDomain
from rema.sky.regions import Box
from rema.validate import mocks

from test_scan_positions import _dr11


@pytest.fixture(autouse=True)
def no_jax_setup(monkeypatch):
    monkeypatch.setattr(cli, "_setup_jax", lambda: None)


def run(*argv):
    assert cli.main([str(a) for a in argv]) == 0


# --------------------------------------------------------------------------- grid
@pytest.mark.parametrize("nmax,base", [(64, 4), (4096, 64)])
def test_levels_nest_and_tile(nmax, base):
    levels, n = [], base
    while n <= min(nmax, base * 8):
        levels.append(n)
        n *= 2
    parents = np.arange(12 * base * base) if nmax <= 64 else np.array([0, 5, 12 * base * base - 1])
    done = np.zeros(0, np.int64)
    for nside in levels:
        new = level_pixels(nside, nmax, base, parents)
        assert np.intersect1d(new, done).size == 0                   # a level adds new points only
        done = np.sort(np.r_[done, new])
        step = (nmax // nside) ** 2
        # the union so far is one point per pixel of nside: the first sub-pixel of every pixel
        assert np.array_equal(done, np.sort(np.concatenate(
            [np.arange(p * (nmax // base) ** 2, (p + 1) * (nmax // base) ** 2, step) for p in parents])))
        assert np.array_equal(np.unique(done // step), np.unique(done // step)) and np.all(done % step == 0)
        # each point lies in the nside pixel it represents
        r, d = hp.pix2ang(nmax, done, nest=True, lonlat=True)
        assert np.array_equal(hp.ang2pix(nside, r, d, nest=True, lonlat=True), done // step)
    assert level_offsets(base, nmax, base).tolist() == [0]


def test_level_errors():
    with pytest.raises(ValueError, match="power of two"):
        level_pixels(100, 4096, 64)
    with pytest.raises(ValueError, match="base"):
        level_pixels(32, 4096, 64)


def test_grid_points_full_sky_and_domain():
    pts = grid_points(4, None, nside_max=64, base=4)
    assert pts["PIX"].size == 12 * 16 and np.all(pts["LEVEL"] == 4)
    dom = RunDomain(np.array([[0.0, 90.0, 20.0, 60.0]]), parts=np.array([0]), regions=np.array([0]),
                    glat_min=0.0)
    p64 = grid_points(64, dom, nside_max=4096, base=64)
    assert np.all(dom.contains(p64["RA"], p64["DEC"]))
    area = np.radians(90.0) * (np.sin(np.radians(60)) - np.sin(np.radians(20))) * (180 / np.pi) ** 2
    assert p64["PIX"].size == pytest.approx(area / hp.nside2pixarea(64, degrees=True), rel=0.03)
    p128 = grid_points(128, dom, nside_max=4096, base=64)
    assert p128["PIX"].size == pytest.approx(3 * p64["PIX"].size, rel=0.03)
    assert np.intersect1d(p64["PIX"], p128["PIX"]).size == 0


# --------------------------------------------------------------------------- measurement
@pytest.fixture(scope="module")
def mock_region():
    rng = np.random.default_rng(5)
    cfg = RemaConfig()
    rs = RSModel.from_template()
    ms, cosmo = MStar(cfg.model.mstar), CosmoTable.create()
    depth5 = np.array([24.9, 24.7, 24.2, 23.6])
    box = Box(10.0, 11.0, -0.5, 0.5)
    field = mocks.mock_field(rng, rs, box, density=12000, depth5=depth5, mag_range=(12.0, 22.5))
    cl = mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, 10.5, 0.0, 0.3, 40.0, depth5, poisson=False,
                            central_dmag=-1.5)
    gal = mocks.concat(field, cl)
    n = gal["RA"].size
    gal["ID"] = np.arange(n, dtype=np.int64)
    gal["ZSPEC"] = np.full(n, -1.0, np.float32)
    return Region.build(gal, rs, cfg, area_deg2=box.area_deg2())


def test_run_grid_fixed_z_matches_the_richness_solve(mock_region):
    reg = mock_region
    z = grid_zsteps(reg.cfg, (0.2, 0.4), 0.1)
    ra, dec = np.array([10.5, 10.2]), np.array([0.0, 0.3])           # the cluster, an empty field point
    cat, mem = run_grid(reg, ra, dec, ids=np.array([7, 9]), zsteps=z, batch=1)
    assert mem == {} and cat["MEM_MATCH_ID"].tolist() == [7, 9]
    lam = cat["LAMBDA_STEPS"]
    assert lam.shape == (2, 3) and cat["SCALEVAL_STEPS"].shape == (2, 3)
    i3 = 1                                                           # z = 0.3
    assert 25 < lam[0, i3] < 60                                      # the cluster at its redshift
    assert lam[0, i3] > lam[0, [0, 2]].max()                         # peaks at its redshift
    assert lam[1, i3] < 0.3 * lam[0, i3]                             # field point: background only
    assert np.all(cat["MASKFRAC_STEPS"] == 0)                       # no footprint: nothing masked
    # the same lambda as the percolation solve at fixed z (the scan's refinement at z0, 0 iterations)
    cfg = reg.cfg
    st = Stage.make(cfg.richness.percolation.r0, cfg.richness.percolation.beta, cfg.richness.maxrad_factor)
    q = RadialQuad.make(rmax=float(st.maxrad) + 5 * cfg.model.rsig)
    from rema.core.richness import richness
    zz = np.array([0.3])
    rad = reg.radius_deg(float(st.maxrad), zz)
    pad = reg.query(ra[:1], dec[:1], rad, mag_max=float(reg.maxmag(zz.max())) + 1.0)
    fr, fg = reg.completeness(ra[:1], dec[:1], zz, q)
    ref = richness(reg.neighbors(pad), zz.astype(np.float32), fr, fg, q, reg.model, st)
    assert lam[0, i3] == pytest.approx(float(ref.lam[0]), rel=1e-4)


def test_run_grid_empty(mock_region):
    cat, _ = run_grid(mock_region, np.zeros(0), np.zeros(0), zsteps=np.array([0.3]))
    assert cat["LAMBDA_STEPS"].shape == (0, 1)


# --------------------------------------------------------------------------- command line
def test_cli_grid_points_and_lambda(tmp_path):
    rng = np.random.default_rng(11)
    dr11 = _dr11(tmp_path, rng)                                      # one sweep: RA 10-15, Dec -5-0
    dom = RunDomain(np.array([[10.5, 14.5, -4.5, -0.5]]), parts=np.array([0]), regions=np.array([0]),
                    glat_min=0.0)
    dom.write(tmp_path / "domain.fits")
    run("grid-points", "--nside", 64, "--domain", tmp_path / "domain.fits", "--out", tmp_path / "p64.fits")
    pts = read_table(tmp_path / "p64.fits", hdu="POINTS")
    assert 15 < pts["PIX"].size < 25 and np.all(pts["LEVEL"] == 64)  # 16 deg2 / 0.84 deg2
    run("grid-lambda", tmp_path / "p64.fits", "--dr11", dr11, "--calib", dr11 / "cal.fits",
        "--work", tmp_path / "work", "--nrand", 1, "--zrange", 0.1, 0.3, "--zstep", 0.1,
        "--task", 1, "--ntask", 2, "--out", tmp_path / "g1.fits")
    g = read_table(tmp_path / "g1.fits", hdu="GRID")
    zs = read_table(tmp_path / "g1.fits", hdu="ZSTEPS")
    np.testing.assert_allclose(zs["Z"], [0.1, 0.2, 0.3], atol=1e-6)
    half = np.array_split(np.arange(pts["PIX"].size), 2)[1]
    np.testing.assert_array_equal(g["PIX"], pts["PIX"][half])
    assert g["SCANNED"].all()
    assert set(c for c in g if c.endswith("_STEPS")) == {"LAMBDA_STEPS", "LAMBDA_E_STEPS", "SCALEVAL_STEPS",
                                                          "MASKFRAC_STEPS"}
    assert g["LAMBDA_STEPS"].shape == (half.size, 3)
    assert np.all((g["MASKFRAC_STEPS"] >= 0) & (g["MASKFRAC_STEPS"] <= 1))
    with pytest.raises(SystemExit, match="--columns"):
        cli.main(["grid-lambda", str(tmp_path / "p64.fits"), "--dr11", str(dr11), "--calib",
                  str(dr11 / "cal.fits"), "--columns", "FOO", "--out", str(tmp_path / "x.fits")])
    with pytest.raises(SystemExit, match="--task"):
        cli.main(["grid-lambda", str(tmp_path / "p64.fits"), "--dr11", str(dr11), "--calib",
                  str(dr11 / "cal.fits"), "--task", "3", "--ntask", "2", "--out", str(tmp_path / "x.fits")])
