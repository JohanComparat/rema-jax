"""Calibration pieces: red-sequence M-step, initial model, corrections."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from rema.calib.corrections import fit_zlambda_correction, fit_zred_correction, fit_zred_uncorr
from rema.calib.fit import fit_redsequence, update_pivot
from rema.calib.init import initial_model
from rema.model.profiles import MStar
from rema.model.redsequence import RSModel
from rema.validate import mocks


@pytest.fixture(scope="module")
def truth():
    rs = RSModel.from_template(zrange=(0.1, 0.8), dz_mean=0.1, dz_other=0.2)
    n_s = len(rs.z_slope)
    return dataclasses.replace(
        rs, slope=jnp.full((n_s, 3), -0.03), log_sigma=jnp.log(jnp.full((len(rs.z_sigma), 3), 0.06)),
        corr=jnp.full((len(rs.z_corr), 3), 0.4))


def _members(rs, rng, n=6000):
    ms = MStar("des_z03")
    z = rng.uniform(0.1, 0.8, n)
    refmag = np.asarray(ms(z)) + rng.uniform(-1.5, 1.5, n)
    mags = mocks.red_sequence_mags(rng, rs, z, refmag)
    flux, ivar = mocks.noisy_fluxes(rng, mags, np.array([24.9, 24.7, 24.2, 23.6]))
    return flux, ivar, z, refmag


def test_fit_recovers_red_sequence(truth):
    rng = np.random.default_rng(11)
    flux, ivar, z, refmag = _members(truth, rng)
    start = dataclasses.replace(truth, mean=truth.mean + 0.08, slope=jnp.zeros_like(truth.slope),
                                log_sigma=jnp.log(jnp.full_like(truth.log_sigma, 0.12)),
                                corr=jnp.zeros_like(truth.corr))
    fit, info = fit_redsequence(start, flux, ivar, z, np.ones(z.size), smooth=0.1, maxiter=400)
    zz = jnp.linspace(0.15, 0.75, 7)
    a, b = truth.at(zz), fit.at(zz)
    np.testing.assert_allclose(np.asarray(b.mean), np.asarray(a.mean), atol=0.025)   # g-r at z=0.75 is noise-limited
    np.testing.assert_allclose(np.asarray(b.slope), np.asarray(a.slope), atol=0.015)
    sig_a = np.sqrt(np.diagonal(np.asarray(a.cint), axis1=1, axis2=2))
    sig_b = np.sqrt(np.diagonal(np.asarray(b.cint), axis1=1, axis2=2))
    np.testing.assert_allclose(sig_b, sig_a, rtol=0.25)
    assert info["loss"][-1] < info["loss"][0]


def test_initial_model_finds_red_sequence(truth):
    rng = np.random.default_rng(5)
    flux, ivar, z, refmag = _members(truth, rng, 8000)
    # add blue galaxies: bluer by 0.3-1 mag in every colour
    fb, ib, zb, mb = _members(truth, rng, 8000)
    fb = fb * 10 ** (0.4 * rng.uniform(0.3, 1.0, (zb.size, 1)) * np.array([3, 2, 1, 0]))
    template = dataclasses.replace(truth, mean=truth.mean - 0.05, log_sigma=jnp.log(jnp.full_like(truth.log_sigma, 0.05)))
    rs, keep = initial_model(np.concatenate([flux, fb]), np.concatenate([ivar, ib]),
                             np.concatenate([refmag, mb]), np.concatenate([z, zb]), template,
                             MStar("des_z03"), zrange=(0.1, 0.8), min_gal=20)
    zz = jnp.linspace(0.15, 0.75, 7)
    np.testing.assert_allclose(np.asarray(rs.at(zz).mean), np.asarray(truth.at(zz).mean), atol=0.03)
    assert keep[: z.size].mean() > 0.8 and keep[z.size:].mean() < 0.2


def test_pivot_update(truth):
    rng = np.random.default_rng(2)
    z = rng.uniform(0.1, 0.8, 5000)
    m = 18 + 4 * z + rng.normal(0, 0.3, z.size)
    rs = update_pivot(truth, m, z, np.ones(z.size))
    # interior nodes (edge nodes only see one side of the window)
    np.testing.assert_allclose(np.asarray(rs.pivot)[1:-1], 18 + 4 * np.asarray(rs.z_pivot)[1:-1], atol=0.05)


def test_corrections():
    rng = np.random.default_rng(3)
    zs = rng.uniform(0.1, 0.8, 5000)
    zu = zs - 0.01 + rng.normal(0, 0.02, zs.size)
    zc = fit_zred_correction(zu, np.full(zs.size, 0.01), zs, zrange=(0.1, 0.8))
    assert np.allclose(np.asarray(zc.corr)[2:-2], 0.01, atol=0.004)
    assert np.allclose(np.asarray(zc.corr_r)[2:-2], 2.0, rtol=0.15)
    zl = zs + 0.005 + rng.normal(0, 0.01, zs.size)
    c = fit_zlambda_correction(zl, np.full(zs.size, 0.006), zs, np.full(zs.size, 30.0), zrange=(0.1, 0.8))
    assert np.allclose(np.asarray(c.offset)[1:-1], -0.005, atol=0.002)
    assert np.allclose(np.asarray(c.scatter)[1:-1], np.sqrt(0.01**2 - 0.006**2), atol=0.002)


def test_fit_zred_uncorr():
    """z -> zred_uncorr: the running median of the central zred against z_lambda, as redMaPPer's
    MedZFitter (tests/wcen_reference.py), robust to outliers; failed values are left out."""
    from scipy.interpolate import CubicSpline

    import wcen_reference as ref

    rng = np.random.default_rng(12)
    n = 3000
    zl = rng.uniform(0.05, 0.9, n)

    def truth(z):
        return z + 0.015 + 0.02 * np.sin(5.0 * z)

    zred = truth(zl) + rng.normal(0.0, 0.01, n)
    out = rng.uniform(size=n) < 0.15                       # outliers on both sides: median kept
    zred[out] = zl[out] + rng.choice([-1.0, 1.0], out.sum()) * rng.uniform(0.1, 0.4, out.sum())
    zred[rng.uniform(size=n) < 0.03] = -1.0                # failed zred
    zl_in = np.where(rng.uniform(size=n) < 0.02, -1.0, zl)  # failed z_lambda
    nodes = np.arange(0.05, 0.94, 0.08)                    # rema's z_lambda-correction nodes
    v = fit_zred_uncorr(zl_in, zred, nodes)
    zz = np.linspace(0.1, 0.85, 76)
    np.testing.assert_allclose(CubicSpline(nodes, v, bc_type="natural")(zz), truth(zz), atol=0.004)
    # redMaPPer's fitter on the same clusters reaches the same L1 cost and curve.
    ok = (zred > 0) & (zl_in > 0)
    vr = ref.medz_fit(nodes, zl[ok], zred[ok])

    def cost(p):
        return np.sum(np.abs(zred[ok] - ref.cubic_spline(nodes, p, zl[ok])))

    assert cost(v) == pytest.approx(cost(vr), rel=1e-4) and cost(v) < 0.8 * cost(nodes)
    np.testing.assert_allclose(ref.cubic_spline(nodes, v, zz), ref.cubic_spline(nodes, vr, zz),
                               atol=1e-3)
    # Fewer clusters than nodes: no mapping.
    assert fit_zred_uncorr(zl[:5], zred[:5], nodes) is None


def test_calibrate_wcen_empty_sample_gives_undefined_model():
    """No usable clusters: undefined wcen parameters (runs fall back to BCG), no exception."""
    from types import SimpleNamespace

    from rema.calib.driver import calibrate_wcen
    from rema.config import RemaConfig
    from rema.core.centering import wcen_calibrated

    reg = SimpleNamespace(gal={"ID": np.arange(5, dtype=np.int64)})
    cand = {"GI": np.zeros(0, np.int64), "Z_LAMBDA": np.zeros(0, np.float32),
            "LAMBDA": np.zeros(0, np.float32), "LNLIKE": np.zeros(0), "LNCGLIKE": np.zeros(0, np.float32)}
    params, info = calibrate_wcen(reg, np.zeros(0, np.int64), RemaConfig(), cand=cand, cat={})
    assert not wcen_calibrated(params)
    assert info["n_train"] == 0 and info["n_rand"] == 0 and info["n_randsat"] == 0
    assert np.isfinite(params["PHI1_MMSTAR_M"]) and np.isnan(params["DELTA0"])
