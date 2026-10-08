"""PSCD (photo-z space cluster detection, the AMICO matched filter): grid, template, normalisation
integrals, unbiased amplitudes on mocks, detection and cleaning, masks, and the command line."""

import numpy as np
import pytest

from rema.config import RemaConfig, apply_overrides
from rema.model.background import build_photoz_bkg
from rema.model.cosmo import CosmoTable
from rema.model.photoz import photoz_sigma
from rema.model.profiles import MStar
from rema.model.redsequence import RSModel
from rema.pscd import detect as D
from rema.pscd.grid import Grid, convolve, gnomonic, inverse_gnomonic, paint
from rema.pscd.model import MagIntegrals, Template, WidthTable, noise_np
from rema.pscd.run import run_pscd
from rema.sky.regions import Box
from rema.validate import mocks

DEPTH5 = np.array([24.9, 24.7, 24.2, 23.6])


@pytest.fixture(scope="module")
def tpl():
    cfg = RemaConfig()
    return Template.create(cfg.pscd, MStar(cfg.model.mstar), CosmoTable.create())


# --------------------------------------------------------------------------- grid
def test_grid_projection_and_painting():
    rng = np.random.default_rng(0)
    ra, dec = rng.uniform(350, 370, 500) % 360, rng.uniform(-60, -45, 500)
    g = Grid.covering(ra, dec, 0.05, margin=0.1)
    px, py = g.to_pix(ra, dec)
    assert px.min() > 0 and py.min() > 0 and px.max() < g.nx - 1 and py.max() < g.ny - 1
    r2, d2 = g.to_sky(px, py)
    np.testing.assert_allclose(((r2 - ra + 180) % 360) - 180, 0, atol=1e-9)
    np.testing.assert_allclose(d2, dec, atol=1e-9)
    rc, dc = g.centres()
    assert rc.shape == g.shape and g.area == pytest.approx(0.0025)
    xi, eta = gnomonic(g.ra0 + 1.0, g.dec0, g.ra0, g.dec0)
    assert float(eta) == pytest.approx(0.0, abs=0.01) and 0.5 < float(xi) < 1.0
    np.testing.assert_allclose(inverse_gnomonic(0.0, 0.0, 10.0, 20.0), (10.0, 20.0))
    with pytest.raises(ValueError):
        Grid.covering([], [], 0.1)
    img = paint(px, py, np.ones(px.size), g.shape)
    assert img.sum() == pytest.approx(500.0)
    assert paint([-5.0], [1.0], [1.0], (3, 3)).sum() == 0


def test_convolution_matches_direct_sum():
    rng = np.random.default_rng(1)
    img = rng.uniform(size=(40, 50))
    ker = rng.uniform(size=(7, 7))
    ker = ker + ker[::-1, ::-1]                       # symmetric, as the profile kernels
    out = convolve(img, ker)
    y, x = 20, 31
    direct = np.sum(img[y - 3:y + 4, x - 3:x + 4] * ker[::-1, ::-1])
    assert out[y, x] == pytest.approx(direct)
    assert np.all(convolve(np.zeros((5, 5)), ker) == 0)


# --------------------------------------------------------------------------- template
def test_template_normalisation(tpl):
    pc = tpl.pc
    for z in (0.1, 0.5, 0.9):
        r200 = float(tpl.r200_deg(z))
        th = np.linspace(1e-7, r200, 20001)
        inside = 2 * np.pi * np.trapezoid(th * tpl.psi(th, z), th)
        assert inside == pytest.approx(1.0, rel=2e-3)
        rmax = float(tpl.rmax_deg(z))
        th2 = np.linspace(1e-7, rmax, 20001)
        total = 2 * np.pi * np.trapezoid(th2 * tpl.psi(th2, z), th2)
        pix = 0.2 * r200
        k = tpl.kernel(z, pix)
        assert k.shape[0] % 2 == 1 and k.sum() * pix**2 == pytest.approx(total, rel=0.03)
        assert np.all(tpl.psi(np.array([1.01 * rmax]), z) == 0)
        ms = float(tpl.mstar(z))
        m = np.linspace(ms - 8, ms + pc.dmag_n200, 4001)
        assert np.trapezoid(tpl.phi(m, z), m) == pytest.approx(pc.n200, rel=1e-3)
    assert float(tpl.r200_deg(0.2)) > float(tpl.r200_deg(0.6))


def test_width_table_and_integrals(tpl):
    rng = np.random.default_rng(2)
    n = 50000
    zp, m = rng.uniform(0.1, 1.0, n), rng.uniform(16, 23, n)
    s = np.where(m < 20, 0.02, 0.08)
    w = WidthTable.build(zp, s, m)
    a1, a2 = w.at(0.5, np.array([18.0, 22.0]))
    np.testing.assert_allclose(a1, [50.0, 12.5], rtol=1e-6)
    np.testing.assert_allclose(a2, [2500.0, 156.25], rtol=1e-6)
    b1, _ = w.at(1.5, np.array([18.0]))               # empty bins take the nearest filled one
    assert b1[0] == pytest.approx(50.0)
    with pytest.raises(ValueError):
        WidthTable.build(zp[:2], np.full(2, -1.0), m[:2])
    bkg = build_photoz_bkg(zp, s, m, lambda mm: 10.0)
    assert noise_np(bkg, 0.5, np.array([20.1]))[0] == pytest.approx(float(bkg.lookup(0.5, 20.1)), rel=1e-5)
    assert np.isinf(noise_np(bkg, 0.5, np.array([5.0]))[0])
    mi = MagIntegrals.build(tpl, bkg, w, 0.5, 22.0)
    mm = np.linspace(float(tpl.mstar(0.5)) - 6, 21.0, 20001)
    N = noise_np(bkg, 0.5, mm)
    inv1, inv2 = w.at(0.5, mm)
    phi = tpl.phi(mm, 0.5)
    i1, i2, i3 = mi.at(21.0)
    assert i1 == pytest.approx(np.trapezoid(phi, mm), rel=1e-3)
    assert i2 == pytest.approx(np.trapezoid(phi**2 * inv1 / (2 * np.sqrt(np.pi)) / N, mm), rel=1e-2)
    assert i3 == pytest.approx(np.trapezoid(phi**3 * inv2 / (2 * np.sqrt(3) * np.pi) / N**2, mm), rel=1e-2)
    assert mi.at(np.array([30.0]))[0][0] == pytest.approx(mi.i1[-1])


def test_amplitude_formula():
    A, sig, snr, L = D.amplitude(np.array([5.0, 1.0, 1.0]), np.array([2.0, 2.0, 0.0]),
                                 np.array([1.0, 1.0, 1.0]), np.array([4.0, 4.0, 4.0]))
    np.testing.assert_allclose(A[:2], [2.0, 0.0])
    np.testing.assert_allclose(sig[:2], [np.sqrt(0.5 + 2.0 * 4.0 / 4.0), np.sqrt(0.5)])
    assert snr[2] == 0 and np.isinf(sig[2]) and L[0] == pytest.approx(8.0)
    assert D._vertex(1.0, 2.0, 1.0) == 0.0 and D._vertex(0.0, 2.0, 1.0) == pytest.approx(1.0 / 6.0)
    assert D._vertex(np.nan, 2.0, 1.0) == 0.0 and D._vertex(3.0, 2.0, 3.0) == 0.0
    w = D._window(np.arange(16.0).reshape(4, 4), 0, 0, 1)
    assert w.tolist() == [[0, 0, 0], [0, 0, 1], [0, 4, 5]]
    cfg = apply_overrides(RemaConfig(), ["pscd.zrange=[0.1, 0.3]", "pscd.dz=0.05"])
    np.testing.assert_allclose(D.z_grid(cfg), [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    assert D.magnitude_limit(cfg) == 99.0
    assert D.magnitude_limit(apply_overrides(cfg, ["photoz.mag_max=22", "survey.mag_max=23"])) == 22.0


# --------------------------------------------------------------------------- mocks
MLIM = 22.0         # measured-magnitude limit of the mocks (drawn to 22.5 in true magnitude)


def _mock(tpl, seed=3, A=2.0, density=12000, box=Box(10.0, 12.0, -1.0, 1.0), specs=None, cut=None):
    rng = np.random.default_rng(seed)
    rs = RSModel.from_template()
    pz = mocks.GaussianPhotoz()
    field = mocks.mock_field(rng, rs, box, density=density, depth5=DEPTH5, mag_range=(12.0, MLIM + 0.5),
                             photoz=pz)
    specs = specs or [(10.5, -0.5, 0.3), (11.4, 0.4, 0.5), (10.6, 0.5, 0.2), (11.5, -0.5, 0.7)]
    cl = [mocks.mock_template_cluster(rng, rs, tpl, ra, dec, z, A, DEPTH5, MLIM + 0.5, pz) for ra, dec, z in specs]
    gal = mocks.concat(field, *cl)
    keep = gal["REFMAG"] < MLIM
    if cut is not None:
        keep &= cut.contains(gal["RA"], gal["DEC"])
    gal = {k: v[keep] for k, v in gal.items()}
    n = gal["RA"].size
    gal["ID"] = np.arange(n, dtype=np.int64)
    gal["ZSPEC"] = np.where(gal["REFMAG"] < 18.0, gal["ZTRUE"], -1.0).astype(np.float32)
    return gal, rs, box, specs


def _cube(tpl, gal, box, cfg):
    g = dict(gal)
    g["ZPHOT_E"] = photoz_sigma(g["ZPHOT"], g["ZPHOT_STD"], g["REFMAG"], cfg.photoz)
    bkg = build_photoz_bkg(g["ZPHOT"], g["ZPHOT_E"], g["REFMAG"], lambda m: box.area_deg2())
    widths = WidthTable.build(g["ZPHOT"], g["ZPHOT_E"], g["REFMAG"])
    grid = Grid.covering(g["RA"], g["DEC"], cfg.pscd.pixel)
    px, py = grid.to_pix(g["RA"], g["DEC"])
    gx = D.Galaxies(rows=np.arange(px.size), ra=g["RA"], dec=g["DEC"], refmag=g["REFMAG"].astype(float),
                    zphot=g["ZPHOT"].astype(float), s=g["ZPHOT_E"].astype(float), px=px, py=py,
                    pfield=np.ones(px.size))
    rc, dc = grid.centres()
    cube = D.build_cube(gx, grid, tpl, bkg, widths, cfg, box.contains(rc, dc).astype(float), mag_limit=MLIM)
    return cube, gx


@pytest.mark.slow
def test_amplitude_is_unbiased(tpl):
    """At the injected clusters the amplitude is A_in; at random points it is 0 with the scatter
    sigma_A = 1/sqrt(alpha) predicted by the filter."""
    cfg = RemaConfig()
    zs = (0.2, 0.35, 0.5, 0.65)
    specs = [(10.3 + 0.45 * i, -0.6 + 0.6 * j, zs[(i + j) % 4]) for i in range(4) for j in range(3)]
    gal, rs, box, specs = _mock(tpl, A=3.0, specs=specs)
    cube, _ = _cube(tpl, gal, box, cfg)
    Ain = []
    for ra, dec, z in specs:
        px, py = cube.grid.to_pix(ra, dec)
        k = int(np.argmin(np.abs(cube.zs - z)))
        A, sig, snr, _ = D.amplitude(*(c[k, int(round(float(py))), int(round(float(px)))]
                                       for c in (cube.S, cube.alpha, cube.beta, cube.gamma)))
        Ain.append((float(A), float(sig)))
    Ain = np.array(Ain)
    err = np.std(Ain[:, 0]) / np.sqrt(len(Ain))
    assert abs(np.mean(Ain[:, 0]) - 3.0) < 3 * err, Ain
    assert np.std(Ain[:, 0]) == pytest.approx(np.mean(Ain[:, 1]), rel=0.5)
    assert np.all(Ain[:, 0] / Ain[:, 1] > 3)          # AMICO's S/N: the clusters' own shot noise
    # field: inner pixels away from the clusters
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(3000):
        k = rng.integers(5, cube.zs.size - 5)
        y, x = rng.integers(15, cube.grid.ny - 15), rng.integers(15, cube.grid.nx - 15)
        ra, dec = cube.grid.to_sky(x, y)
        if min(np.hypot((ra - s[0]) * np.cos(np.radians(dec)), dec - s[1]) for s in specs) < 0.25:
            continue
        A, sig, _, _ = D.amplitude(cube.S[k, y, x], cube.alpha[k, y, x], cube.beta[k, y, x], cube.gamma[k, y, x])
        vals.append((float(A), float(1 / np.sqrt(cube.alpha[k, y, x]))))
    vals = np.array(vals)
    pull = vals[:, 0] / vals[:, 1]
    # slightly negative: the noise is measured on all the galaxies, the clusters' included
    assert abs(np.mean(pull)) < 0.25 and 0.7 < np.std(pull) < 1.3, (np.mean(pull), np.std(pull))


@pytest.mark.slow
def test_run_pscd_detects_and_cleans(tpl):
    cfg = apply_overrides(RemaConfig(), [f"photoz.mag_max={MLIM}"])
    gal, rs, box, specs = _mock(tpl, A=2.0)
    info = {}
    cat, mem = run_pscd(gal, cfg, sky=box, own=box, rs=rs, area_deg2=box.area_deg2(), region_id=2, info=info)
    assert info["n_detections"] >= len(specs) and info["n_used"] <= info["n_galaxies"]
    for ra, dec, z in specs:
        d = np.hypot((cat["RA"] - ra) * np.cos(np.radians(dec)), cat["DEC"] - dec)
        near = np.flatnonzero(d < 0.05)
        assert near.size, (ra, dec, z)
        i = near[np.argmax(cat["SNR"][near])]
        assert abs(cat["Z"][i] - z) < 0.03 and cat["SNR"][i] > 3 and cat["SNR_NOCL"][i] > 15
        assert cat["A"][i] == pytest.approx(2.0, rel=0.5) and 0 < cat["Z_E"][i] < 0.02
        assert cat["LAMBDA"][i] > 0.5 * cat["N_EXP"][i] and cat["LAMBDA_STAR"][i] <= cat["LAMBDA"][i]
        assert cat["ID_BCG"][i] >= 0
    # one strong detection per cluster: the cleaning removed the cluster's imprint
    assert np.sum(cat["SNR_NOCL"] > 15) <= len(specs) + 2
    assert np.all(np.diff(cat["SNR"]) <= 0)
    assert np.all(cat["MEM_MATCH_ID"] >> 32 == 2)
    assert set(np.unique(mem["MEM_MATCH_ID"])) <= set(cat["MEM_MATCH_ID"])
    tot = np.bincount(mem["ID"], weights=mem["P"])
    assert tot.max() <= 1.0 + 1e-5 and np.all(mem["PFIELD"] >= 0)
    assert {"CHISQ_RS", "ZPHOT_E", "ZSPEC", "R"} <= set(mem)
    # blue members are found as well as red ones
    top = cat["MEM_MATCH_ID"][np.argmax(cat["SNR"])]
    sel = (mem["MEM_MATCH_ID"] == top) & (mem["P"] > 0.5)
    assert np.mean(mem["CHISQ_RS"][sel] > 20) > 0.2


@pytest.mark.slow
def test_pscd_mask_and_null(tpl):
    """A cluster cut by the data box keeps an unbiased amplitude and has MASKFRAC ~ 0.5; the
    photo-z null test finds no strong detection."""
    cfg = apply_overrides(RemaConfig(), ["pscd.max_maskfrac=1.0", f"photoz.mag_max={MLIM}"])
    box = Box(10.0, 11.0, -0.5, 0.5)
    gal, rs, _, specs = _mock(tpl, A=3.0, density=8000, box=Box(9.5, 11.5, -1.0, 1.0),
                              specs=[(11.0, 0.0, 0.4), (10.4, -0.1, 0.3)], cut=box)
    cat, mem = run_pscd(gal, cfg, sky=box, area_deg2=box.area_deg2())
    d = np.hypot((cat["RA"] - 11.0) * np.cos(0.0), cat["DEC"])
    i = np.flatnonzero(d < 0.05)
    i = i[np.argmax(cat["SNR"][i])]
    assert cat["MASKFRAC"][i] == pytest.approx(0.5, abs=0.15)
    assert cat["A"][i] == pytest.approx(3.0, rel=0.5)
    null = apply_overrides(cfg, ["null.shuffle=photoz"])
    cn, _ = run_pscd(gal, null, sky=box, area_deg2=box.area_deg2())
    assert np.sum(cn.get("SNR_NOCL", np.zeros(0)) > 20) == 0


def test_run_pscd_edge_cases(tpl):
    cfg = RemaConfig()
    with pytest.raises(KeyError, match="ZPHOT"):
        run_pscd({"REFMAG": np.zeros(3)}, cfg)
    info = {}
    gal = {"ID": np.arange(3), "RA": np.zeros(3), "DEC": np.zeros(3), "REFMAG": np.full(3, 20.0),
           "ZPHOT": np.full(3, -1.0), "ZPHOT_STD": np.full(3, 0.1)}
    assert run_pscd(gal, cfg, info=info) == ({}, {}) and info["n_detections"] == 0


@pytest.mark.slow
def test_cli_pscd(tpl, tmp_path, monkeypatch):
    from rema import cli
    from rema.calibration import Calibration
    from rema.io.legacy import survey_hash
    from rema.io.tables import read_catalog, write_table

    monkeypatch.setattr(cli, "_setup_jax", lambda: None)
    box = Box(10.0, 11.0, -0.5, 0.5)
    gal, rs, _, _ = _mock(tpl, A=3.0, density=6000, box=box, specs=[(10.5, 0.0, 0.35)])
    cfg = RemaConfig()
    write_table(tmp_path / "gal.fits", gal, header={"SURVHASH": survey_hash(cfg)}, extname="GALAXIES")
    Calibration(rs=rs, config=cfg).write(tmp_path / "cal.fits")
    assert cli.main(["pscd", "--galaxies", str(tmp_path / "gal.fits"), "--calib", str(tmp_path / "cal.fits"),
                     "--box", *map(str, box.as_tuple()), "--own", *map(str, box.as_tuple()),
                     "--set", "pscd.snr_kind=background", "--set", "pscd.snr_min=5", "--set",
                     f"photoz.mag_max={MLIM}", "--specpost", "--out", str(tmp_path / "p.fits")]) == 0
    cat, mem, hdr = read_catalog(tmp_path / "p.fits")
    assert hdr["FILTER"] == "pscd" and hdr["MODE"] == "pscd" and hdr["NDET"] >= 1
    i = int(np.argmax(cat["SNR"]))
    assert abs(cat["RA"][i] - 10.5) < 0.03 and abs(cat["Z"][i] - 0.35) < 0.03
    assert "NSPEC" in cat and "BEST_Z" in cat and "CHISQ_RS" in mem


def test_uncovered_cells_are_not_searched(tpl):
    """Cells whose profile is mostly off the footprint get no amplitude (no overflow at edges)."""
    rng = np.random.default_rng(4)
    n = 3000
    box = Box(10.0, 10.5, 0.0, 0.5)
    gal = {"RA": rng.uniform(10.0, 10.5, n), "DEC": rng.uniform(0.0, 0.5, n),
           "REFMAG": rng.uniform(17, 21, n).astype(np.float32), "ZPHOT": rng.uniform(0.1, 0.8, n).astype(np.float32),
           "ZPHOT_STD": np.full(n, 0.03, np.float32)}
    cfg = apply_overrides(RemaConfig(), ["pscd.zrange=[0.3, 0.4]", "pscd.snr_min=0"])
    cube, gx = _cube(tpl, gal, box, cfg)
    assert np.isfinite(cube.tmax).any()
    fmap = np.zeros(cube.grid.shape)
    fmap[:2, :2] = 1e-30                              # almost nothing observed
    bkg = build_photoz_bkg(gal["ZPHOT"], gx.s, gal["REFMAG"], lambda m: box.area_deg2())
    widths = WidthTable.build(gal["ZPHOT"], gx.s, gal["REFMAG"])
    empty = D.build_cube(gx, cube.grid, tpl, bkg, widths, cfg, fmap, mag_limit=21.0)
    assert np.all(empty.alpha == 0) and not np.isfinite(empty.tmax).any()
    assert D.extract(empty, gx) == ([], [])
    with pytest.raises(ValueError, match="snr_kind"):
        D.build_cube(gx, cube.grid, tpl, bkg, widths, apply_overrides(cfg, ["pscd.snr_kind=both"]), fmap)


def test_finder_comparison_helpers():
    from rema.validate.compare import match_physical, null_threshold

    d = CosmoTable.create().mpc_per_deg
    i1, i2, r = match_physical([10.0], [0.0], [-1.0], [10.0], [0.0], [0.3], d)
    assert i1.size == 0 and r.size == 0
    i1, i2, r = match_physical([10.0, 10.0], [0.0, 0.0], [0.3, 0.3], [10.001, 10.002], [0.0, 0.0],
                               [0.3, 0.31], d, rank2=[1.0, 5.0])
    # the first object takes the best-ranked match, the second the one left
    assert i2.tolist() == [1, 0] and np.all(r < 1.0)
    assert match_physical([10.0], [0.0], [0.3], [12.0], [0.0], [0.3], d)[0].size == 0
    t, pur = null_threshold([1.0, 2.0], [0.5], 1.0, 5.0)
    assert t == -np.inf and pur == pytest.approx(0.5)
    t, pur = null_threshold([1.0], [3.0, 2.0], 1.0, 0.5)
    assert t == 3.0 and np.isnan(pur)
