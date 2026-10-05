"""Small model pieces: splines, one-colour and two-colour red sequences, the zred correction, the
chi^2 modes, cosmology and m* tables, redshift corrections, catalogue matching, the wcen model
and configuration updates."""

import jax.numpy as jnp
import numpy as np
import pytest

from rema.calib.corrections import _binned, _clipped, fit_zred_correction
from rema.config import RemaConfig
from rema.core.centering import WcenModel
from rema.model.cosmo import CosmoTable
from rema.model.likelihood import _mm, chisq
from rema.model.profiles import MStar, _table_path
from rema.model.redsequence import RSModel, ZredCorrection
from rema.model.splines import NaturalSpline
from rema.validate.compare import match_catalogs
from rema.validate import mocks


def test_spline_degenerate_nodes():
    with pytest.raises(ValueError, match="non-empty"):
        NaturalSpline([])
    with pytest.raises(ValueError, match="increasing"):
        NaturalSpline([0.1, 0.1, 0.3])
    one = NaturalSpline([0.3])
    np.testing.assert_array_equal(one.basis(np.array([0.0, 1.0])), [[1.0], [1.0]])
    np.testing.assert_allclose(np.asarray(one.eval_jax(jnp.asarray([0.1, 0.9]), jnp.asarray([[2.0, 3.0]]))),
                               [[2.0, 3.0], [2.0, 3.0]])
    two = NaturalSpline([0.0, 1.0])                  # a straight line
    assert np.all(two._G == 0)
    np.testing.assert_allclose(two(np.array([0.25, 2.0]), np.array([1.0, 3.0])), [1.5, 5.0])


def test_one_colour_model():
    rs = RSModel.from_template(bands=("r", "z"))
    assert rs.ncol == 1 and rs.corr.shape[1] == 0
    at = rs.at(jnp.asarray([0.2, 0.4]))
    assert at.cint.shape == (2, 1, 1)
    np.testing.assert_allclose(np.asarray(at.cint)[:, 0, 0], 0.05**2, rtol=1e-5)
    T, others = rs.band_offsets_matrix()
    assert others == [0] and T.tolist() == [[1.0]]
    rng = np.random.default_rng(1)
    mags = mocks.red_sequence_mags(rng, rs, 0.3, np.array([19.0, 20.0]), scatter=False)
    np.testing.assert_allclose(mags[:, 0] - mags[:, 1], float(rs.at(jnp.asarray(0.3)).mean[0]), atol=1e-5)


def test_zred_correction_identity_and_apply():
    ident = ZredCorrection.identity()
    zu, ze, m = jnp.asarray([0.2, 0.5]), jnp.asarray([0.01, 0.02]), jnp.asarray([19.0, 20.0])
    z, e = ident.apply(zu, ze, m, lambda z: jnp.full_like(z, 19.5))
    np.testing.assert_allclose(np.asarray(z), [0.2, 0.5], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(e), [0.01, 0.02], rtol=1e-6)
    nodes = jnp.linspace(0.0, 1.0, 5)
    zc = ZredCorrection(jnp.full(5, 0.01), jnp.full(5, 0.002), jnp.full(5, 1.5), nodes, nodes)
    z, e = zc.apply(zu, ze, m, lambda z: jnp.full_like(z, 19.5))
    np.testing.assert_allclose(np.asarray(z), np.asarray(zu) + 0.01 + 0.002 * (np.asarray(m) - 19.5), atol=1e-6)
    np.testing.assert_allclose(np.asarray(e), 1.5 * np.asarray(ze), rtol=1e-6)


def test_chisq_modes():
    rng = np.random.default_rng(4)
    rs = RSModel.from_template()
    z = np.full(200, 0.3)
    refmag = rng.uniform(17, 19, 200)
    flux, ivar = mocks.noisy_fluxes(rng, mocks.red_sequence_mags(rng, rs, z, refmag),
                                    np.array([24.9, 24.7, 24.2, 23.6]))
    at = rs.at(jnp.asarray(z, jnp.float32))
    c_lupt, _ = chisq(jnp.asarray(flux), jnp.asarray(ivar), at, rs.iref, "lupt")
    c_mag, _ = chisq(jnp.asarray(flux), jnp.asarray(ivar), at, rs.iref, "mag")
    # Bright red-sequence galaxies: chi^2 ~ 3 (three colours) in both modes.
    assert 2.0 < np.median(np.asarray(c_lupt)) < 4.5 and 2.0 < np.median(np.asarray(c_mag)) < 4.5
    with pytest.raises(ValueError, match="unknown chisq mode"):
        chisq(jnp.asarray(flux), jnp.asarray(ivar), at, rs.iref, "flux")
    a = jnp.arange(6.0).reshape(2, 3)
    np.testing.assert_allclose(np.asarray(_mm(a, jnp.ones(3))), [3.0, 12.0])


def test_cosmology_and_mstar_tables(tmp_path):
    from astropy.io import fits

    c = CosmoTable.create()
    z = jnp.asarray([0.1, 0.5])
    np.testing.assert_allclose(np.asarray(c.da(z)), np.asarray(c.dc(z)) / (1 + np.asarray(z)), rtol=1e-6)
    assert float(c.dc(jnp.asarray(0.5))) == pytest.approx(1322.0, rel=0.01)    # Mpc/h, Omega_m 0.3
    # m* from an explicit FITS table path.
    zz = np.linspace(0.0, 2.0, 41)
    path = str(tmp_path / "mstar.fits")
    fits.BinTableHDU.from_columns([fits.Column("Z", "D", array=zz),
                                   fits.Column("MSTAR", "D", array=17.0 + 3.0 * zz)]).writeto(path)
    assert _table_path(path, "mstar") == path
    assert float(MStar(path)(0.5)) == pytest.approx(18.5, abs=1e-3)


def test_correction_helpers():
    keep = _clipped(np.array([1.0, np.nan]))
    assert keep.tolist() == [True, False]                  # too few points: no clipping
    x = np.linspace(0.1, 0.3, 30)
    y = np.where(np.arange(30) < 25, 0.0, 10.0)            # outliers clipped below min_n
    assert np.isnan(_binned(x, y, np.array([0.2]), 0.2, min_n=26))[0]
    # No usable galaxies: zero offset, unit error scaling.
    zc = fit_zred_correction(np.full(5, -1.0), np.ones(5), np.full(5, 0.3), zrange=(0.1, 0.3))
    np.testing.assert_array_equal(np.asarray(zc.corr), 0.0)
    np.testing.assert_array_equal(np.asarray(zc.corr_r), 1.0)


def test_match_catalogs_one_to_one():
    ra1, dec1, z1 = np.array([10.0, 10.0001]), np.array([0.0, 0.0]), np.array([0.3, 0.3])
    i1, i2, sep = match_catalogs(ra1, dec1, z1, np.array([10.0]), np.array([0.0]), np.array([0.3]))
    assert i1.tolist() == [0] and i2.tolist() == [0] and sep[0] == pytest.approx(0.0, abs=1e-6)
    i1, i2, sep = match_catalogs(ra1, dec1, z1, np.array([20.0]), np.array([0.0]), np.array([0.3]))
    assert i1.size == i2.size == sep.size == 0


def test_wcen_model_pivot_and_config_update():
    params = dict.fromkeys(("DELTA0", "DELTA1", "SIGMA_M", "LNW_CEN_MEAN", "LNW_CEN_SIGMA",
                            "LNW_SAT_MEAN", "LNW_SAT_SIGMA", "LNW_FG_MEAN", "LNW_FG_SIGMA"), 0.1)
    cfg = RemaConfig()
    assert WcenModel.create({**params, "PIVOT": np.nan}, None, cfg).pivot == cfg.centering.wcen_pivot
    assert WcenModel.create({**params, "pivot": 20.0}, None, cfg).pivot == 20.0
    # replace() with a whole section object rather than a dict of changes.
    spec = cfg.spec.__class__(nboot=4)
    assert cfg.replace(spec=spec).spec.nboot == 4
