"""Red-sequence model and photometric likelihoods."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import chi2 as chi2_dist, kstest

from rema.model.likelihood import chisq_lupt, chisq_mag, small_cholesky_solve
from rema.model.redsequence import (RSModel, ZredCorrection, band_offsets_matrix,
                                    corr_from_partial, partial_from_corr)

from conftest import DR10_PARS, require


def test_partial_correlation_roundtrip(rng):
    for n in (2, 3, 4):
        A = rng.normal(size=(n, n + 2))
        C = A @ A.T
        d = np.sqrt(np.diag(C))
        R = C / d[:, None] / d[None, :]
        z = partial_from_corr(R)
        np.testing.assert_allclose(np.asarray(corr_from_partial(jnp.asarray(z, jnp.float32))), R, atol=1e-5)
    # Any parameter values give a valid correlation matrix.
    R = np.asarray(corr_from_partial(jnp.tanh(jnp.asarray(rng.normal(size=(50, 6)) * 3))))
    assert np.all(np.linalg.eigvalsh(R) > -1e-6)
    np.testing.assert_allclose(np.diagonal(R, axis1=1, axis2=2), 1.0, atol=1e-5)


def test_small_cholesky(rng):
    A = rng.normal(size=(100, 3, 5))
    S = A @ np.swapaxes(A, 1, 2) + 0.1 * np.eye(3)
    r = rng.normal(size=(100, 3))
    chi2, logdet = small_cholesky_solve(jnp.asarray(S), jnp.asarray(r))
    np.testing.assert_allclose(np.asarray(chi2), np.einsum("ni,nij,nj->n", r, np.linalg.inv(S), r), rtol=1e-4)
    np.testing.assert_allclose(np.asarray(logdet), np.linalg.slogdet(S)[1], rtol=1e-4, atol=1e-4)


def test_band_offsets_other_reference():
    T, others = band_offsets_matrix(4, 1)            # reference r
    assert others == [0, 2, 3]
    np.testing.assert_array_equal(T, [[1, 0, 0], [0, -1, 0], [0, -1, -1]])


def _simulate(rs, z, n, rng, refmag, depth5=None, noise=True):
    """Fluxes of red-sequence galaxies at redshift z (bands of rs), optional Gaussian noise."""
    at = rs.at(jnp.full(n, z))
    mean = np.asarray(at.colours(jnp.asarray(refmag)))
    cint = np.asarray(at.cint)
    col = mean + np.einsum("nij,nj->ni", np.linalg.cholesky(cint), rng.normal(size=mean.shape))
    T, others = band_offsets_matrix(rs.nband, rs.iref)
    mags = np.zeros((n, rs.nband))
    mags[:, rs.iref] = refmag
    mags[:, others] = refmag[:, None] + col @ T.T
    flux = 10 ** (-0.4 * (mags - 22.5))
    if depth5 is None:
        sig = 1e-4 * flux
    else:
        sig = np.broadcast_to(10 ** (-0.4 * (np.asarray(depth5) - 22.5)) / 5.0, flux.shape)
    if noise:
        flux = flux + rng.normal(size=flux.shape) * sig
    return flux.astype(np.float32), (1.0 / sig**2).astype(np.float32)


@pytest.fixture(scope="module")
def template_rs():
    rs = RSModel.from_template()
    # Give it non-trivial slopes and correlations.
    return RSModel.from_arrays(
        {"mean": rs.z_mean, "slope": rs.z_slope, "sigma": rs.z_sigma, "corr": rs.z_corr, "pivot": rs.z_pivot},
        {"mean": np.asarray(rs.mean), "slope": np.full(np.shape(rs.slope), -0.02),
         "log_sigma": np.asarray(rs.log_sigma), "corr": np.full(np.shape(rs.corr), 0.5),
         "pivot": np.asarray(rs.pivot)}, rs.bands, rs.ref_band)


def test_lupt_and_mag_chisq_agree_at_high_snr(template_rs, rng):
    n, z = 4000, 0.4
    refmag = rng.uniform(17.5, 20.0, n)
    flux, ivar = _simulate(template_rs, z, n, rng, refmag)
    at = template_rs.at(jnp.full(n, z))
    cf, _ = chisq_lupt(flux, ivar, at, template_rs.iref, eps=0.0)
    cm, _ = chisq_mag(flux, ivar, at, template_rs.iref, eps=0.0)
    cf, cm = np.asarray(cf), np.asarray(cm)
    assert np.all(np.abs(cf - cm) < 0.02 * cm + 0.05)
    # Under the model the chi^2 follows chi^2_3.
    assert kstest(cf, chi2_dist(3).cdf).pvalue > 1e-3


def test_lupt_chisq_noisy_faint_galaxies(template_rs, rng):
    """Faint galaxies with noisy (even negative) g fluxes still give chi^2_3 under the model."""
    n, z = 20000, 0.8
    refmag = rng.uniform(21.5, 22.8, n)
    flux, ivar = _simulate(template_rs, z, n, rng, refmag, depth5=[24.9, 24.7, 24.2, 23.6])
    cf, _ = chisq_lupt(flux, ivar, template_rs.at(jnp.full(n, z)), template_rs.iref, eps=0.0)
    cf = np.asarray(cf)
    assert np.all(np.isfinite(cf))
    # The linearised ref-noise term makes the tail slightly heavy; the bulk is chi^2_3.
    assert abs(np.median(cf) - chi2_dist(3).median()) < 0.15
    assert np.mean(cf > chi2_dist(3).ppf(0.99)) < 0.03


def test_chisq_gradients(template_rs, rng):
    flux, ivar = _simulate(template_rs, 0.5, 8, rng, rng.uniform(19, 21, 8),
                           depth5=[24.9, 24.7, 24.2, 23.6])

    def f(mean):
        rs = RSModel.from_arrays(
            {"mean": template_rs.z_mean, "slope": template_rs.z_slope, "sigma": template_rs.z_sigma,
             "corr": template_rs.z_corr, "pivot": template_rs.z_pivot},
            {"mean": mean, "slope": template_rs.slope, "log_sigma": template_rs.log_sigma,
             "corr": template_rs.corr, "pivot": template_rs.pivot}, template_rs.bands, template_rs.ref_band)
        return jnp.sum(chisq_lupt(flux, ivar, rs.at(jnp.full(8, 0.5)), rs.iref)[0])

    g = jax.grad(f)(template_rs.mean)
    assert np.all(np.isfinite(np.asarray(g))) and np.any(np.asarray(g) != 0)


@pytest.mark.data
def test_read_dr10_pars():
    require(DR10_PARS)
    rs = RSModel.from_redmapper_pars(str(DR10_PARS))
    assert rs.bands == ("g", "r", "z") and rs.ref_band == "z" and rs.ncol == 2
    at = rs.at(jnp.asarray([0.05, 0.3, 0.6]))
    np.testing.assert_allclose(np.asarray(at.mean[0]), [0.858, 0.620], atol=2e-3)
    R = np.asarray(at.cint[1]) / np.sqrt(np.outer(np.diag(at.cint[1]), np.diag(at.cint[1])))
    assert R[0, 1] == pytest.approx(0.9, abs=1e-3)
    corr = ZredCorrection.from_redmapper_pars(str(DR10_PARS))
    z, ze = corr.apply(jnp.asarray([0.3]), jnp.asarray([0.02]), jnp.asarray([19.0]),
                       lambda zz: rs.at(zz).pivot)
    assert abs(float(z[0]) - 0.3) < 0.02 and float(ze[0]) > 0
