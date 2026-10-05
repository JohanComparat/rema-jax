"""Calibration file round trip and the calib-import command."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits

from rema.calibration import Calibration, ZlambdaCorrection
from rema.cli import main
from rema.config import RemaConfig
from rema.core.centering import ZRMOD_IDENTITY, WcenModel
from rema.model.background import ChisqBkg
from rema.model.redsequence import RSModel, ZredCorrection

from conftest import DR10_PARS, require


def test_calibration_roundtrip(tmp_path):
    rs = RSModel.from_template()
    zc = ZredCorrection(jnp.linspace(0, 0.01, 5), jnp.zeros(5), jnp.ones(5) * 1.2,
                        jnp.linspace(0.1, 0.9, 5), jnp.linspace(0.1, 0.9, 5))
    bkg = ChisqBkg(jnp.linspace(0.05, 0.9, 5), jnp.linspace(0.25, 19.75, 40), jnp.linspace(12.1, 23.9, 60),
                   jnp.ones((5, 40, 60)), jnp.ones(60))
    zl = ZlambdaCorrection(jnp.linspace(0.05, 0.9, 6), jnp.full(6, 0.002), jnp.zeros(6), jnp.full(6, 0.004))
    cfg = RemaConfig().replace(model={"chisq_mode": "mag"})
    cal = Calibration(rs=rs, zredcorr=zc, bkg=bkg, zlcorr=zl, wcen={"DELTA0": -1.2, "SIGMA_M": 0.3},
                      config=cfg, meta={"NCLUSTER": 10})
    p = cal.write(tmp_path / "cal.fits")
    back = Calibration.read(p)
    np.testing.assert_allclose(np.asarray(back.rs.mean), np.asarray(rs.mean), rtol=1e-12)
    assert back.rs.z_mean == rs.z_mean and back.rs.bands == rs.bands
    np.testing.assert_allclose(np.asarray(back.zredcorr.corr_r), 1.2)
    np.testing.assert_allclose(np.asarray(back.bkg.sigma_g), 1.0)
    np.testing.assert_allclose(np.asarray(back.zlcorr.scatter), 0.004)
    assert back.wcen["DELTA0"] == pytest.approx(-1.2)
    assert back.config == cfg
    assert back.meta["NCLUSTER"] == 10
    z, ze = back.zlcorr.apply(jnp.asarray([0.3]), jnp.asarray([0.003]), jnp.asarray([30.0]))
    assert float(z[0]) == pytest.approx(0.302) and float(ze[0]) == pytest.approx(0.005)
    assert back.zlcorr.zred_uncorr is None and back.zlcorr.zrmod_table() is None


def test_calibration_zred_uncorr_roundtrip(tmp_path):
    """The z -> zred_uncorr mapping is the ZRED_UNCORR column of ZLAMBDACORR. A file without it
    (as every file written before it) reads as no mapping: the wcen model uses zrmod(z) = z."""
    rs = RSModel.from_template()
    nodes = np.arange(0.05, 0.94, 0.08)
    zru = nodes + 0.01 + 0.02 * np.sin(6.0 * nodes)
    zl = ZlambdaCorrection(jnp.asarray(nodes), jnp.full(nodes.size, 0.002), jnp.zeros(nodes.size),
                           jnp.full(nodes.size, 0.004), zred_uncorr=jnp.asarray(zru))
    params = {"DELTA0": -1.5, "DELTA1": 0.0, "SIGMA_M": 0.3, "LNW_CEN_MEAN": 0.3,
              "LNW_CEN_SIGMA": 0.25, "LNW_SAT_MEAN": -0.2, "LNW_SAT_SIGMA": 0.35,
              "LNW_FG_MEAN": -0.6, "LNW_FG_SIGMA": 0.45}
    cfg = RemaConfig()
    # With the mapping.
    p = Calibration(rs=rs, zlcorr=zl, wcen=params).write(tmp_path / "with.fits")
    with fits.open(p) as h:
        assert h["ZLAMBDACORR"].columns.names == ["Z", "OFFSET", "SLOPE", "SCATTER", "ZRED_UNCORR"]
    back = Calibration.read(p).zlcorr
    np.testing.assert_allclose(np.asarray(back.zred_uncorr), zru, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(back.offset), 0.002)
    (zz, zr), (zz0, zr0) = back.zrmod_table(), zl.zrmod_table()
    np.testing.assert_array_equal(zz, zz0)
    np.testing.assert_allclose(zr, zr0, rtol=0, atol=1e-6)
    z, ze = back.apply(jnp.asarray([0.3]), jnp.asarray([0.003]), jnp.asarray([30.0]))
    assert float(z[0]) == pytest.approx(0.302) and float(ze[0]) == pytest.approx(0.005)
    wc = WcenModel.create(params, None, cfg, zrmod=back.zrmod_table())
    np.testing.assert_allclose(np.asarray(wc.zrmod), zr0, rtol=1e-6)
    # Without it: the four columns of earlier files, no table, the identity in the wcen model.
    p0 = Calibration(rs=rs, zlcorr=dataclasses.replace(zl, zred_uncorr=None),
                     wcen=params).write(tmp_path / "without.fits")
    with fits.open(p0) as h:
        assert h["ZLAMBDACORR"].columns.names == ["Z", "OFFSET", "SLOPE", "SCATTER"]
    back0 = Calibration.read(p0).zlcorr
    assert back0.zred_uncorr is None and back0.zrmod_table() is None
    np.testing.assert_allclose(np.asarray(back0.offset), 0.002)
    wc0 = WcenModel.create(params, None, cfg, zrmod=back0.zrmod_table())
    np.testing.assert_array_equal(np.asarray(wc0.zrmod_z), ZRMOD_IDENTITY[0])
    np.testing.assert_array_equal(np.asarray(wc0.zrmod), ZRMOD_IDENTITY[1])


@pytest.mark.data
def test_cli_calib_import(tmp_path):
    require(DR10_PARS)
    out = tmp_path / "dr10.fits"
    assert main(["calib-import", str(DR10_PARS), "--out", str(out)]) == 0
    cal = Calibration.read(out)
    assert cal.rs.bands == ("g", "r", "z") and cal.zredcorr is not None


def test_compare_catalogs():
    from rema.validate.compare import match_catalogs, summary

    rng = np.random.default_rng(4)
    n = 200
    ref = {"RA": rng.uniform(0, 5, n), "DEC": rng.uniform(-5, 0, n), "Z_LAMBDA": rng.uniform(0.1, 0.8, n),
           "LAMBDA": rng.uniform(5, 80, n)}
    test = {"RA": ref["RA"] + rng.normal(0, 0.002, n), "DEC": ref["DEC"] + rng.normal(0, 0.002, n),
            "Z_LAMBDA": ref["Z_LAMBDA"] + 0.005, "LAMBDA": ref["LAMBDA"] * 1.1}
    i1, i2, sep = match_catalogs(test["RA"], test["DEC"], test["Z_LAMBDA"], ref["RA"], ref["DEC"], ref["Z_LAMBDA"])
    assert i1.size == n and np.all(i1 == i2)
    s = summary(test, ref, lam_min=20)
    assert s["recovered"] == 1.0 and s["confirmed"] == 1.0
    assert s["median_lnlam_ratio"] == pytest.approx(np.log(1.1), abs=1e-6)
