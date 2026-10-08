"""Photo-z filter (model.filter: photoz): configuration overrides, photo-z widths, the stacked
photo-z background, null shuffles, the membership terms, and recovery of mock clusters with blue
members."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from rema.config import PhotozConfig, RemaConfig, apply_overrides
from rema.core import richness as R
from rema.core.context import FilterModel
from rema.core.zlambda import _member_lnl_one, _weighted_lnl, zlambda
from rema.io.tables import read_catalog, write_catalog
from rema.model.background import ZredBkg, build_photoz_bkg
from rema.model.cosmo import CosmoTable
from rema.model.photoz import err_scale, photoz_sigma
from rema.model.profiles import MStar, selection_fraction
from rema.model.redsequence import RSModel
from rema.modes import blind as B
from rema.modes.common import Region
from rema.sky.neighbors import NeighborIndex
from rema.sky.regions import Box
from rema.validate import mocks
from rema.validate.null import _permutation, null_shuffle
from rema.validate.photoz import fit_err_scale, nmad

PZ = ["model.filter=photoz"]


# --------------------------------------------------------------------------- configuration
def test_overrides(tmp_path):
    cfg = RemaConfig()
    c = apply_overrides(cfg, ["richness.percolation.beta=0.3", "photoz.mag_max=22",
                              "photoz.err_scale_mag=[19, 21]", "photoz.err_scale=[0.8, 0.6]"])
    assert c.richness.percolation == dataclasses.replace(cfg.richness.percolation, beta=0.3)
    assert c.richness.firstpass == cfg.richness.firstpass
    assert c.photoz.mag_max == 22.0 and isinstance(c.photoz.mag_max, float)
    assert c.photoz.err_scale_mag == (19, 21)
    assert apply_overrides(cfg, None) == cfg
    over = tmp_path / "pz.yaml"
    over.write_text("model: {filter: photoz}\nseeds: {dmag_max: 1}\n")
    c = apply_overrides(cfg, [f"@{over}", "null.shuffle=photoz"])
    assert (c.model.filter, c.seeds.dmag_max, c.null.shuffle) == ("photoz", 1.0, "photoz")
    assert c.model.chisq_max == cfg.model.chisq_max
    for bad, err in [("model.filtr=photoz", KeyError), ("filter=photoz", ValueError),
                     ("model.zrange.lo=1", KeyError), ("model", ValueError)]:
        with pytest.raises(err):
            apply_overrides(cfg, [bad])
    over.write_text("model: {filtr: photoz}\n")
    with pytest.raises(KeyError, match="filtr"):
        apply_overrides(cfg, [f"@{over}"])
    # A configuration written before the photo-z sections reads with their defaults.
    d = cfg.to_dict()
    for k in ("photoz", "null"):
        d.pop(k)
    d["model"].pop("filter")
    assert RemaConfig.from_dict(d) == cfg
    assert RemaConfig.from_yaml(_yaml(tmp_path, apply_overrides(cfg, PZ))).model.filter == "photoz"


def _yaml(tmp_path, cfg):
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    return p


def test_filter_model_checks():
    rs = RSModel.from_template()
    with pytest.raises(ValueError, match="expected one of"):
        FilterModel.create(rs, None, apply_overrides(RemaConfig(), ["model.filter=colour"]))
    with pytest.raises(ValueError, match="pzbkg"):
        FilterModel.create(rs, None, apply_overrides(RemaConfig(), PZ))


# --------------------------------------------------------------------------- photo-z widths
def test_photoz_sigma_and_scale():
    pc = PhotozConfig(err_scale=(0.9, 0.5), err_scale_mag=(19.0, 22.0), err_floor=0.02)
    np.testing.assert_allclose(err_scale([18.0, 20.5, 25.0], pc), [0.9, 0.7, 0.5])
    s = photoz_sigma([0.5, 0.5, 0.5, -1.0, 0.5, np.nan], [0.2, 0.2, 0.001, 0.2, 0.0, 0.1],
                     [18.0, 25.0, 20.0, 20.0, 20.0, 20.0], pc)
    np.testing.assert_allclose(s, [0.18, 0.1, 0.03, -1.0, -1.0, -1.0], rtol=1e-6)
    for bad in (PhotozConfig(err_scale=(1.0, 0.5)), PhotozConfig(err_scale=(1.0, 0.5), err_scale_mag=(21, 19)),
                PhotozConfig(err_scale=(1.0,), err_scale_mag=(19, 21))):
        with pytest.raises(ValueError):
            err_scale([20.0], bad)


def test_fit_err_scale():
    rng = np.random.default_rng(3)
    m = rng.uniform(17, 23, 40000)
    zs = rng.uniform(0.05, 1.0, m.size)
    sig = 0.01 * (1 + zs) * 10 ** (0.2 * (m - 18))
    zp = zs + sig * rng.normal(size=m.size)
    out = rng.uniform(size=m.size) < 0.05
    zp[out] = rng.uniform(0, 1.5, out.sum())
    fit = fit_err_scale(zp, 2.0 * sig, zs, m, edges=(17, 19, 21, 23))
    np.testing.assert_allclose(fit["err_scale"], 0.5, atol=0.05)     # 5 % outliers widen the NMAD a little
    assert all(0.03 < b["outliers"] < 0.07 and abs(b["bias"]) < 0.01 for b in fit["bins"])
    assert np.all(np.diff(fit["err_scale_mag"]) > 0)
    assert fit_err_scale(zp[:10], sig[:10], zs[:10], m[:10])["bins"] == []
    assert np.isnan(nmad([]))


# --------------------------------------------------------------------------- background
def test_photoz_background_units():
    rng = np.random.default_rng(0)
    n, area = 200000, 10.0
    zp = rng.uniform(0.3, 0.7, n)
    m = rng.uniform(18, 22, n)
    s = np.full(n, 0.02)
    bkg = build_photoz_bkg(zp, s, m, lambda mm: area, zrange=(0.0, 1.6), zbinsize=0.005,
                           magbinsize=0.2, mag_min=12.0, mag_max=24.0)
    # inside: N / (A dm dz)
    assert float(bkg.lookup(0.5, 20.0)) == pytest.approx(n / (area * 4.0 * 0.4), rel=0.03)
    # the redshift integral is N(m), the counts per deg^2 per mag
    zz = np.asarray(bkg.zred)
    col = np.asarray(bkg.lookup(jnp.asarray(zz), jnp.full(zz.size, 20.1)))
    assert np.trapezoid(col, zz) == pytest.approx(n / area / 4.0, rel=0.03)
    # outside the redshift range: the Gaussian tails, then nothing
    assert float(bkg.lookup(0.3, 20.0)) == pytest.approx(0.5 * float(bkg.lookup(0.5, 20.0)), rel=0.05)
    assert float(bkg.lookup(1.0, 20.0)) < 1e-6 * float(bkg.lookup(0.5, 20.0))
    # below the tables: +inf (no background, never a member)
    assert np.isinf(float(bkg.lookup(-0.1, 20.0))) and np.isinf(float(bkg.lookup(0.5, 11.0)))
    # invalid photo-z are ignored; sparse bright bins borrow from their neighbours
    bk2 = build_photoz_bkg(np.r_[zp, 0.5, 0.5], np.r_[s, -1.0, 0.02], np.r_[m, 20.0, 14.0],
                           lambda mm: area, chunk=50000)
    assert float(bk2.lookup(0.5, 20.0)) == pytest.approx(float(bkg.lookup(0.5, 20.0)), rel=1e-3)
    assert np.isfinite(float(bk2.lookup(0.5, 14.0)))
    # no area: no background
    bk3 = build_photoz_bkg(zp[:100], s[:100], m[:100], lambda mm: 0.0)
    assert not np.isfinite(float(bk3.lookup(0.5, 20.0)))


# --------------------------------------------------------------------------- null tests
def test_null_shuffles():
    rng = np.random.default_rng(1)
    n = 5000
    m = rng.uniform(17, 23, n).astype(np.float32)
    flux = (10 ** (-0.4 * (m[:, None] - 22.5)) * rng.uniform(0.5, 2, (n, 4))).astype(np.float32)
    flux[:, 3] = 10 ** (-0.4 * (m - 22.5))
    gal = {"REFMAG": m, "ZPHOT": rng.uniform(0, 1, n).astype(np.float32),
           "ZPHOT_STD": rng.uniform(0.01, 0.1, n).astype(np.float32), "FLUX": flux,
           "FLUX_IVAR": np.full((n, 4), 100.0, np.float32), "RA": rng.uniform(size=n)}
    cfg = RemaConfig()
    assert null_shuffle(gal, cfg.null, 3) is gal
    c = apply_overrides(cfg, ["null.shuffle=photoz", "null.magbin=0.5"])
    out = null_shuffle(gal, c.null, 3)
    assert out["RA"] is gal["RA"] and out["REFMAG"] is gal["REFMAG"]
    assert not np.array_equal(out["ZPHOT"], gal["ZPHOT"])
    key = np.floor(m / 0.5)
    for k in np.unique(key)[:5]:
        sel = key == k
        assert sorted(out["ZPHOT"][sel]) == sorted(gal["ZPHOT"][sel])
    # pairs move together
    pairs = set(zip(gal["ZPHOT"].tolist(), gal["ZPHOT_STD"].tolist()))
    assert set(zip(out["ZPHOT"].tolist(), out["ZPHOT_STD"].tolist())) == pairs
    assert np.array_equal(null_shuffle(gal, c.null, 3)["ZPHOT"], out["ZPHOT"])     # seeded
    col = null_shuffle(gal, apply_overrides(cfg, ["null.shuffle=colour"]).null, 3)
    np.testing.assert_array_equal(col["FLUX"][:, 3], flux[:, 3])
    ratio = col["FLUX"][:, 0] / col["FLUX"][:, 3]
    np.testing.assert_allclose(np.sort(ratio), np.sort(flux[:, 0] / flux[:, 3]), rtol=1e-5)
    assert not np.allclose(ratio, flux[:, 0] / flux[:, 3])
    with pytest.raises(ValueError):
        null_shuffle(gal, apply_overrides(cfg, ["null.shuffle=positions"]).null, 3)
    with pytest.raises(KeyError):
        null_shuffle({"REFMAG": m}, c.null, 3)
    p = _permutation(np.array([1.0, np.nan, 1.01, 5.0]), 0.1, np.random.default_rng(0))
    assert p[3] == 3 and sorted(p) == [0, 1, 2, 3]


# --------------------------------------------------------------------------- membership terms
def _model(cfg, sigma_g=50.0):
    rs = RSModel.from_template()
    pzbkg = ZredBkg(jnp.linspace(0.0, 1.6, 321), jnp.linspace(12.1, 23.9, 60), jnp.full((321, 60), sigma_g))
    return FilterModel.create(rs, None, cfg, pzbkg=pzbkg), rs


def _neighbors(k=6, seed=0):
    rng = np.random.default_rng(seed)
    z = jnp.asarray
    return R.Neighbors(theta=z(rng.uniform(0.001, 0.05, k), jnp.float32),
                       refmag=z(rng.uniform(18, 21, k), jnp.float32), refmag_err=jnp.full(k, 0.02),
                       flux=jnp.ones((k, 4)), ivar=jnp.ones((k, 4)), zred=jnp.zeros(k), zred_e=jnp.ones(k),
                       pfree=jnp.ones(k), valid=jnp.ones(k, bool), is_center=jnp.zeros(k, bool),
                       zphot=z([0.40, 0.45, 0.30, 0.41, -1.0, 0.40], jnp.float32),
                       zphot_e=z([0.02, 0.02, 0.02, 0.01, 0.02, -1.0], jnp.float32))


def test_photoz_terms_against_numpy():
    cfg = apply_overrides(RemaConfig(), PZ + ["photoz.mag_max=20.5"])
    model, _ = _model(cfg)
    nb = _neighbors()
    st = R.Stage.make(1.0, 0.2)
    z = jnp.float32(0.4)
    t = R._filter_terms(nb, z, model, st)
    zp, s = np.asarray(nb.zphot), np.asarray(nb.zphot_e)
    x = (zp - 0.4) / np.where(s > 0, s, 1)
    rho = np.exp(-0.5 * x**2) / (np.sqrt(2 * np.pi) * s)
    D = float(model.mpc_per_deg(0.4))
    r = np.asarray(nb.theta) * D
    ok = (zp >= 0) & (s > 0) & (x**2 < 16) & (np.asarray(nb.refmag) < 20.5) & (r < float(st.maxrad))
    np.testing.assert_array_equal(np.asarray(t["ok"]), ok)
    np.testing.assert_allclose(np.asarray(t["rho"])[ok], rho[ok], rtol=1e-5)
    assert np.all(np.asarray(t["rho"])[~ok] == 0) and np.all(np.asarray(t["u0"])[~ok] == 0)
    np.testing.assert_allclose(np.asarray(t["b"]), 2 * np.pi * r * np.where(ok, 50.0, 1.0) / D**2, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(t["chi2"])[:4], x[:4] ** 2, rtol=1e-5)
    assert np.all(np.asarray(t["chi2"])[4:] == R.BAD_CHISQ)
    assert np.all(np.isfinite(np.asarray(t["lndet"])))
    # the richness solve runs on these terms
    ones = jnp.ones((R.RadialQuad.make().r.shape[0],))
    rich = R.richness_one(nb, z, ones, ones, R.RadialQuad.make(), model, st)
    assert np.all(np.isfinite(np.asarray(rich.p)))


def test_photoz_zlambda_likelihood():
    cfg = apply_overrides(RemaConfig(), PZ)
    model, _ = _model(cfg)
    nb = _neighbors()
    lnl = np.asarray(_member_lnl_one(nb, jnp.float32(0.42), model))
    zp, s = np.asarray(nb.zphot)[:4], np.asarray(nb.zphot_e)[:4]
    np.testing.assert_allclose(lnl[:4], -0.5 * ((zp - 0.42) / s) ** 2 - np.log(s), rtol=1e-5)
    # sum_i w_i ln p_i(z) peaks at the 1/s^2-weighted mean photo-z
    w = jnp.asarray([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    zg = jnp.linspace(0.38, 0.46, 81)
    L = np.asarray(_weighted_lnl(nb, w, zg, model))
    wm = np.sum(np.array([1, 1, 1]) * zp[[0, 1, 3]] / s[[0, 1, 3]] ** 2) / np.sum(1 / s[[0, 1, 3]] ** 2)
    assert float(zg[np.argmax(L)]) == pytest.approx(wm, abs=0.001)


def test_selection_fraction_without_footprint():
    """With a photo-z magnitude limit and no footprint, K is the fraction of the luminosity
    function brighter than the limit."""
    region = _light_pz_region(["photoz.mag_max=20"])
    assert region.mag_limit() == 20.0
    q = R.RadialQuad.make()
    frad, fgeo = region.completeness(np.array([10.5, 10.6]), np.array([0.5, 0.5]), np.array([0.3, 0.6]), q)
    ms = MStar("des_z03")
    for i, z in enumerate((0.3, 0.6)):
        mst = float(ms(z))
        want = float(selection_fraction(mst, mst + 2.5 * np.log10(5.0), -1.0, 20.0))
        np.testing.assert_allclose(np.asarray(frad)[i], want, rtol=1e-5)
    assert np.all(np.asarray(fgeo) == 1)
    assert float(selection_fraction(ms(0.6), ms(0.6) + 1.75, -1.0, 20.0)) < 0.5
    assert _light_pz_region([]).mag_limit() == 99.0


def _light_pz_region(extra, n=400, seed=8):
    rng = np.random.default_rng(seed)
    gal = {"ID": np.arange(n, dtype=np.int64), "RA": rng.uniform(10, 11, n), "DEC": rng.uniform(0, 1, n),
           "FLUX": rng.uniform(1, 10, (n, 4)).astype(np.float32),
           "FLUX_IVAR": np.full((n, 4), 100.0, np.float32),
           "REFMAG": rng.uniform(17, 21, n).astype(np.float32),
           "REFMAG_ERR": np.full(n, 0.02, np.float32),
           "ZPHOT": rng.uniform(0.05, 0.9, n).astype(np.float32),
           "ZPHOT_STD": np.full(n, 0.05, np.float32)}
    for k in ("ZRED", "ZRED_UNCORR"):
        gal[k] = np.full(n, 0.3, np.float32)
    for k in ("ZRED_E", "ZRED_UNCORR_E", "ZRED_CHISQ"):
        gal[k] = np.full(n, 1.0, np.float32)
    return Region.build(gal, RSModel.from_template(), apply_overrides(RemaConfig(), PZ + extra), area_deg2=1.0)


def test_photoz_region():
    reg = _light_pz_region([])
    assert reg.model.filter == "photoz" and reg.model.bkg is None and reg.model.pzbkg is not None
    assert reg.wcen is None and reg.zlcorr is None and reg.centering_method() == "bcg"
    with pytest.raises(ValueError, match="BCG"):
        reg.centering_method("wcen")
    np.testing.assert_allclose(reg.gal["ZPHOT_E"], np.maximum(0.05, 0.01 * (1 + reg.gal["ZPHOT"])), rtol=1e-6)
    pad = reg.query([10.5], [0.5], 0.2)
    nb = reg.neighbors(pad)
    np.testing.assert_array_equal(np.asarray(nb.zphot), reg.gal["ZPHOT"][pad.idx])
    np.testing.assert_array_equal(reg.seed_z([0, 1]), reg.gal["ZPHOT"][[0, 1]])
    assert set(reg.photoz_columns([0, 1])) == {"ZPHOT", "ZPHOT_STD", "ZPHOT_E"}
    seeds = B.select_seeds(reg)
    g = reg.gal
    assert np.all(g["ZPHOT_E"][seeds] < 0.1 * (1 + g["ZPHOT"][seeds]))
    assert np.all(g["REFMAG"][seeds] < reg.mstar_np(g["ZPHOT"][seeds]) + 1.75)
    chi = reg.rs_chisq(g["FLUX"][:5], g["FLUX_IVAR"][:5], np.full(5, 0.3), chunk=2)
    assert chi.shape == (5,) and np.all(chi > 0)
    gal = {k: v for k, v in reg.gal.items() if k not in ("ZPHOT", "ZPHOT_E")}
    with pytest.raises(KeyError, match="ZPHOT"):
        Region.build(gal, reg.rs, reg.cfg, area_deg2=1.0)
    # the cosmology is changed with the photo-z background kept
    other = reg.with_cosmology({"Omega_m": 0.35}, recompute_zred=False)
    assert other.model.pzbkg is reg.model.pzbkg and other.model.filter == "photoz"


def test_catalog_header(tmp_path):
    cfg = apply_overrides(RemaConfig(), PZ + ["null.shuffle=photoz"])
    write_catalog(tmp_path / "c.fits", {"RA": np.zeros(2)}, {"ID": np.zeros(3)}, cfg)
    _, _, hdr = read_catalog(tmp_path / "c.fits")
    assert hdr["FILTER"] == "photoz" and hdr["NULL"] == "photoz"
    write_catalog(tmp_path / "d.fits", {"RA": np.zeros(2)}, None, RemaConfig())
    _, _, hdr = read_catalog(tmp_path / "d.fits")
    assert hdr["FILTER"] == "redsequence" and "NULL" not in hdr


# --------------------------------------------------------------------------- mock clusters
def _blue_mock(seed=4, blue=0.5, density=12000, std_factor=1.0):
    rng = np.random.default_rng(seed)
    cfg = RemaConfig()
    rs = RSModel.from_template()
    ms, cosmo = MStar(cfg.model.mstar), CosmoTable.create()
    depth5 = np.array([24.9, 24.7, 24.2, 23.6])
    box = Box(10.0, 12.0, -1.0, 1.0)
    pz = mocks.GaussianPhotoz(std_factor=std_factor)
    field = mocks.mock_field(rng, rs, box, density=density, depth5=depth5, mag_range=(12.0, 22.5), photoz=pz)
    specs = [(10.5, -0.5, 0.3, 40.0), (11.4, 0.4, 0.5, 30.0), (10.6, 0.5, 0.25, 25.0), (11.5, -0.5, 0.6, 35.0)]
    cl = [mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, ra, dec, z, lam, depth5, poisson=False,
                             blue_fraction=blue, photoz=pz, central_dmag=-1.0)
          for ra, dec, z, lam in specs]
    gal = mocks.concat(field, *cl)
    n = gal["RA"].size
    gal["ID"] = np.arange(n, dtype=np.int64)
    gal["ZSPEC"] = np.full(n, -1.0, np.float32)
    return gal, rs, box, specs, cl


def test_mock_photoz_and_blue_members():
    gal, rs, box, specs, cl = _blue_mock(density=2000)
    assert {"ZTRUE", "ZPHOT", "ZPHOT_STD"} <= set(gal)
    blue = np.concatenate([c["IS_BLUE"] for c in cl])
    assert 0.35 < blue.mean() < 0.65 and not any(c["IS_BLUE"][0] for c in cl)
    x = (gal["ZPHOT"] - gal["ZTRUE"]) / gal["ZPHOT_STD"]
    assert nmad(x[gal["ZTRUE"] > 0.1]) == pytest.approx(1.0, abs=0.05)
    out = mocks.GaussianPhotoz(outlier_frac=0.5)(np.random.default_rng(0), np.full(1000, 0.5), np.full(1000, 20.0))
    assert np.mean(np.abs(out["ZPHOT"] - 0.5) > 0.2) > 0.3


@pytest.mark.slow
def test_photoz_lambda_counts_blue_members():
    """The photo-z filter counts the blue members that the red-sequence filter misses."""
    gal, rs, box, specs, _ = _blue_mock()
    cfg = RemaConfig()
    cosmo = CosmoTable.create()
    S = np.array(specs)
    res = {}
    for name, over in (("rs", []), ("pz", PZ)):
        reg = Region.build(gal, rs, apply_overrides(cfg, over), area_deg2=box.area_deg2())
        st, quad = R.Stage.make(1.0, 0.2), R.RadialQuad.make()
        rad = float(st.maxrad) / np.asarray([float(cosmo.mpc_per_deg(z)) for z in S[:, 2]])
        pad = reg.query(S[:, 0], S[:, 1], rad)
        nb = reg.neighbors(pad)
        ones = jnp.ones((len(S), quad.r.shape[0]))
        zl = zlambda(nb, jnp.asarray(S[:, 2] + 0.01, jnp.float32), ones, ones, quad, reg.model, st)
        res[name] = (np.asarray(zl.rich.lam), np.asarray(zl.z), np.asarray(zl.z_e))
    lam_pz, z_pz, ze_pz = res["pz"]
    lam_rs = res["rs"][0]
    ratio = lam_pz / S[:, 3]
    assert np.all((ratio > 0.75) & (ratio < 1.3)), ratio
    assert np.all(lam_rs < 0.8 * lam_pz), (lam_rs, lam_pz)
    assert np.all(np.abs(z_pz - S[:, 2]) < 0.02) and np.all(ze_pz > 0)


@pytest.mark.slow
def test_blind_photoz_recovers_blue_clusters(tmp_path):
    gal, rs, box, specs, _ = _blue_mock()
    cfg = apply_overrides(RemaConfig(), PZ + ["seeds.dmag_max=1.0"])
    reg = Region.build(gal, rs, cfg, area_deg2=box.area_deg2())
    info = {}
    cat, mem = B.run_blind(reg, own=box, info=info)
    for ra, dec, z, lam in specs:
        # BCG centring: the brightest member can be a few hundred kpc from the injected centre
        d = np.hypot((cat["RA"] - ra) * np.cos(np.radians(dec)), cat["DEC"] - dec)
        near = np.flatnonzero(d < 0.1)
        assert near.size
        i = near[np.argmax(cat["LAMBDA"][near])]
        assert abs(cat["Z_LAMBDA"][i] - z) < 0.02
        assert 0.6 * lam < cat["LAMBDA"][i] < 1.5 * lam
        assert cat["SNR"][i] > 5
    np.testing.assert_array_equal(cat["Z_LAMBDA"], cat["Z_LAMBDA_RAW"])
    assert np.all(cat["LNCGLIKE"] == 0)
    assert {"ZPHOT", "ZPHOT_E", "SNR"} <= set(cat)
    assert {"ZPHOT", "ZPHOT_STD", "ZPHOT_E", "CHISQ_RS"} <= set(mem)
    # Z_INIT is the seed's photo-z
    seed_rows = np.searchsorted(reg.gal["ID"], cat["SEED_ID"])
    np.testing.assert_allclose(cat["Z_INIT"], reg.gal["ZPHOT"][seed_rows], rtol=1e-6)
    # the checkpoint key depends on the filter and on the photo-z
    seeds = B.select_seeds(reg)
    k_pz = B._Checkpoint.for_run(tmp_path / "a", reg, seeds).key
    reg_rs = Region.build(gal, rs, RemaConfig(), area_deg2=box.area_deg2())
    assert B._Checkpoint.for_run(tmp_path / "b", reg_rs, seeds).key != k_pz
    gal2 = dict(gal, ZPHOT=gal["ZPHOT"] + np.float32(0.001))
    reg2 = Region.build(gal2, rs, cfg, area_deg2=box.area_deg2(), pzbkg=reg.model.pzbkg)
    assert B._Checkpoint.for_run(tmp_path / "c", reg2, seeds).key != k_pz


@pytest.mark.slow
def test_scan_photoz():
    from rema.modes.scan import run_scan

    gal, rs, box, specs, _ = _blue_mock()
    cfg = apply_overrides(RemaConfig(), PZ + ["scan.zrange=[0.1, 0.8]", "scan.zstep=0.01"])
    reg = Region.build(gal, rs, cfg, area_deg2=box.area_deg2())
    S = np.array(specs)
    cat, mem = run_scan(reg, S[:, 0], S[:, 1])
    assert np.all(np.abs(cat["Z_LAMBDA"] - S[:, 2]) < 0.03)
    assert np.all(cat["SNR"] > 5) and {"ZPHOT", "ZPHOT_E"} <= set(mem)


# --------------------------------------------------------------------------- command line
@pytest.fixture(scope="module")
def pz_run(tmp_path_factory):
    """A 0.36 deg^2 mock field with photo-z and one cluster with blue members, its galaxy table
    and a template calibration."""
    from rema.calibration import Calibration
    from rema.io.legacy import survey_hash
    from rema.io.tables import write_table

    d = tmp_path_factory.mktemp("pz_run")
    rng = np.random.default_rng(53)
    cfg = RemaConfig()
    rs = RSModel.from_template()
    ms, cosmo = MStar(cfg.model.mstar), CosmoTable.create()
    depth5 = np.array([24.9, 24.7, 24.2, 23.6])
    box = Box(10.0, 10.6, -0.3, 0.3)
    pz = mocks.GaussianPhotoz()
    field = mocks.mock_field(rng, rs, box, density=4000, depth5=depth5, mag_range=(14.0, 22.0), photoz=pz)
    cl = mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, 10.3, 0.0, 0.3, 30.0, depth5, poisson=False,
                            central_dmag=-1.5, blue_fraction=0.5, photoz=pz)
    gal = mocks.concat(field, cl)
    n = gal["RA"].size
    gal["ID"] = np.arange(n, dtype=np.int64)
    gal["ZSPEC"] = np.full(n, -1.0, np.float32)
    write_table(d / "gal.fits", gal, header={"SURVHASH": survey_hash(cfg)}, extname="GALAXIES")
    Calibration(rs=rs, config=cfg).write(d / "cal0.fits")
    return d, box


def _cli(*argv):
    from rema import cli

    return cli.main([str(a) for a in argv])


def test_cli_refuses_photoz(pz_run, monkeypatch):
    from rema import cli

    monkeypatch.setattr(cli, "_setup_jax", lambda: None)
    d, box = pz_run
    common = ["--galaxies", d / "gal.fits", "--set", "model.filter=photoz"]
    with pytest.raises(SystemExit, match="red sequence"):
        _cli("calibrate", *common, "--out", d / "x.fits")
    with pytest.raises(SystemExit, match="red-sequence filter only"):
        _cli("background", *common, "--calib", d / "cal0.fits", "--out", d / "x.fits")
    with pytest.raises(SystemExit, match="red-sequence filter only"):
        _cli("remeasure", *common, "--calib", d / "cal0.fits", "--catalog", d / "x.fits", "--out", d / "y.fits")


@pytest.mark.slow
def test_cli_blind_photoz(pz_run, monkeypatch):
    from rema import cli
    from rema.cli import _catalog_cfg

    monkeypatch.setattr(cli, "_setup_jax", lambda: None)
    d, box = pz_run
    over = d / "pz.yaml"
    over.write_text("model: {filter: photoz}\nseeds: {dmag_max: 1.0}\n")
    assert _cli("blind", "--galaxies", d / "gal.fits", "--calib", d / "cal0.fits", "--box", *box.as_tuple(),
                "--set", f"@{over}", "--set", "photoz.mag_max=21.5", "--out", d / "pz.fits") == 0
    cat, mem, hdr = read_catalog(d / "pz.fits")
    assert hdr["FILTER"] == "photoz" and hdr["CENTRING"] == "bcg"
    cfg = _catalog_cfg(d / "pz.fits")
    assert cfg.model.filter == "photoz" and cfg.photoz.mag_max == 21.5 and cfg.seeds.dmag_max == 1.0
    i = int(np.argmax(cat["LAMBDA"]))
    assert abs(cat["RA"][i] - 10.3) < 0.02 and abs(cat["Z_LAMBDA"][i] - 0.3) < 0.02
    assert cat["SNR"][i] > 4 and np.all(mem["REFMAG"] < 21.5)


def test_effective_area_over_the_data_box():
    """Backgrounds built from galaxies of a sub-box count the footprint's area in that box only."""
    import healpy as hp

    from rema.modes.common import _area_function
    from rema.sky.maps import Footprint, SparseMap

    nside = 1024
    big = Box(10.0, 14.0, -2.0, 2.0)
    ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=True, lonlat=True)
    pix = np.flatnonzero(big.contains(ra, dec))
    fp = Footprint(SparseMap(nside, pix, {"FRACGOOD": np.ones(pix.size), "SIGF_Z": np.full(pix.size, 0.01)}),
                   ("g", "r", "i", "z"), 2500.0, big)
    full = float(fp.effective_area(20.0, "z")[0])
    assert full == pytest.approx(big.area_deg2(), rel=0.02)
    assert float(fp.effective_area(20.0, "z", sky=big)[0]) == full
    sub = Box(11.0, 12.0, -1.0, 0.0)
    assert float(fp.effective_area(20.0, "z", sky=sub)[0]) == pytest.approx(sub.area_deg2(), rel=0.05)
    assert fp.pixels_in(sub).sum() < pix.size
    gal = {"RA": np.array([11.5]), "DEC": np.array([-0.5])}
    area = _area_function(gal, RSModel.from_template(), RemaConfig(), fp, None, sub)
    assert area(20.0) == pytest.approx(sub.area_deg2(), rel=0.05)
    assert apply_overrides(RemaConfig(), ["richness.min_lnlamlike=1e6"]).richness.min_lnlamlike == 1e6


def test_min_lnlamlike_cuts_candidates():
    """With richness.min_lnlamlike the first-pass and likelihood candidates need that LNLAMLIKE."""
    reg = _light_pz_region(["richness.min_lnlamlike=1e6"])
    info = {}
    cat, mem = B.run_blind(reg, info=info)
    assert info["n_seeds"] > 0 and info["n_firstpass"] == 0 and cat == {}
