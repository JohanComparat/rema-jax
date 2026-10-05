"""Richness solve, implicit gradients, completeness and z_lambda on mocks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rema.config import RemaConfig
from rema.core import richness as R
from rema.core.context import FilterModel
from rema.core.zlambda import zlambda
from rema.model.background import build_chisq_bkg
from rema.model.cosmo import CosmoTable
from rema.model.profiles import MStar
from rema.model.redsequence import RSModel
from rema.sky.neighbors import NeighborIndex
from rema.sky.regions import Box
from rema.validate import mocks


def _terms(rng, n=600):
    r = rng.uniform(0.01, 1.5, n)
    u = rng.uniform(0.0, 3.0, n) * 2 * np.pi * r * 0.05
    b = rng.uniform(0.02, 0.5, n) * 2 * np.pi * r
    w = rng.uniform(0.5, 1.0, n)
    return r, u, b, w


@pytest.fixture(scope="module")
def model_defaults():
    cfg = RemaConfig()
    rs = RSModel.from_template()
    cosmo = CosmoTable.create()
    ms = MStar("des_z03")
    return cfg, rs, cosmo, ms


def _solve_with_terms(r, u, b, w, quad, model, stage, frad):
    t = dict(r=jnp.asarray(r), u0=jnp.asarray(u), b=jnp.asarray(b), w=jnp.asarray(w))
    lam, _ = R._solve_lambda(t, frad, quad, stage, model, 32, 4, 0.5, 2000.0)
    return lam


def test_lambda_solve_matches_brentq(rng, model_defaults):
    cfg, rs, cosmo, ms = model_defaults
    model = FilterModel(rs=rs, bkg=None, cosmo=cosmo, mstar_z=jnp.asarray(ms.z), mstar_m=jnp.asarray(ms.m))
    with jax.enable_x64(True):
        stage = R.Stage.make(1.0, 0.2)
        quad = R.RadialQuad.make()
        for _ in range(5):
            r, u, b, w = _terms(rng)
            lam = float(_solve_with_terms(r, u, b, w, quad, model, stage, jnp.ones(quad.r.shape)))
            ref = mocks.solve_lambda_numpy(u, b, w, r)
            assert lam == pytest.approx(ref, rel=1e-4)


def test_grid_chunk():
    assert [R.grid_chunk(21, k, 7 * 2048, 7) for k in (512, 2048, 4096, 8192)] == [7, 7, 3, 1]
    assert [R.grid_chunk(32, k, 32 * 2048) for k in (512, 2048, 8192, 32768)] == [32, 32, 8, 2]


def test_lambda_solve_large_k(rng, model_defaults):
    """Many neighbours (K = 8192): same solution as brentq."""
    cfg, rs, cosmo, ms = model_defaults
    model = FilterModel(rs=rs, bkg=None, cosmo=cosmo, mstar_z=jnp.asarray(ms.z), mstar_m=jnp.asarray(ms.m))
    with jax.enable_x64(True):
        stage = R.Stage.make(1.0, 0.2)
        quad = R.RadialQuad.make()
        r, u, b, w = _terms(rng, n=8192)
        w = w * 600 / 8192                     # keeps sum(w) and lambda as for 600 neighbours
        lam = float(_solve_with_terms(r, u, b, w, quad, model, stage, jnp.ones(quad.r.shape)))
        assert lam == pytest.approx(mocks.solve_lambda_numpy(u, b, w, r), rel=1e-4)


def test_lambda_implicit_gradient(rng, model_defaults):
    cfg, rs, cosmo, ms = model_defaults
    model = FilterModel(rs=rs, bkg=None, cosmo=cosmo, mstar_z=jnp.asarray(ms.z), mstar_m=jnp.asarray(ms.m))
    r, u, b, w = _terms(rng)
    with jax.enable_x64(True):
        stage = R.Stage.make(1.0, 0.2)
        quad = R.RadialQuad.make()
        f = lambda s: _solve_with_terms(r, u, jnp.asarray(b) * s, w, quad, model, stage,
                                        jnp.ones(quad.r.shape))
        g = float(jax.grad(f)(1.0))
        h = 1e-4
        fd = (float(f(1.0 + h)) - float(f(1.0 - h))) / (2 * h)
    assert g < 0 and g == pytest.approx(fd, rel=1e-4)


def test_completeness_scales_lambda(rng, model_defaults):
    """With a uniform completeness K the solution satisfies sum pmem = lambda K (SCALEVAL = 1/K)."""
    cfg, rs, cosmo, ms = model_defaults
    model = FilterModel(rs=rs, bkg=None, cosmo=cosmo, mstar_z=jnp.asarray(ms.z), mstar_m=jnp.asarray(ms.m))
    stage = R.Stage.make(1.0, 0.2)
    quad = R.RadialQuad.make()
    r, u, b, w = _terms(rng)
    lam1 = float(_solve_with_terms(r, u, b, w, quad, model, stage, jnp.ones(quad.r.shape)))
    lam2 = float(_solve_with_terms(r, u, b, w, quad, model, stage, 0.6 * jnp.ones(quad.r.shape)))
    assert lam2 > lam1
    assert lam2 == pytest.approx(mocks.solve_lambda_numpy(u, b, w, r, K=0.6), rel=1e-3)


@pytest.mark.slow
def test_mock_cluster_recovery(model_defaults):
    """Richness and z_lambda of mock clusters injected in a mock field (no mask)."""
    cfg, rs, cosmo, ms = model_defaults
    rng = np.random.default_rng(7)
    depth5 = np.array([24.9, 24.7, 24.2, 23.6])
    box = Box(0, 8, -4, 4)
    field = mocks.mock_field(rng, rs, box, density=30000, depth5=depth5)
    specs = [(ra, dec, 0.4, 40.0) for ra in np.arange(0.8, 7.5, 1.3) for dec in np.arange(-3.2, 3.5, 1.3)]
    clusters = [mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, ra, dec, z, lam, depth5, poisson=False)
                for ra, dec, z, lam in specs]
    gal = mocks.concat(field, *clusters)
    bkg = build_chisq_bkg(gal["FLUX"], gal["FLUX_IVAR"], gal["REFMAG"], rs, ms, lambda m: box.area_deg2(),
                          zrange=(0.05, 0.9), iref=rs.iref, mag_max=24.0)
    model = FilterModel.create(rs, bkg, cfg, cosmo=cosmo, mstar=ms)
    stage, quad = R.Stage.make(1.0, 0.2), R.RadialQuad.make()
    S = np.array(specs)
    pad = NeighborIndex(gal["RA"], gal["DEC"]).query(S[:, 0], S[:, 1], float(stage.maxrad) / float(cosmo.mpc_per_deg(0.4)))
    take = lambda a: jnp.asarray(a[pad.idx])
    nb = R.Neighbors(theta=jnp.asarray(pad.theta, jnp.float32), refmag=take(gal["REFMAG"]),
                     refmag_err=take(gal["REFMAG_ERR"]), flux=take(gal["FLUX"]), ivar=take(gal["FLUX_IVAR"]),
                     zred=jnp.zeros(pad.idx.shape), zred_e=jnp.ones(pad.idx.shape), pfree=jnp.ones(pad.idx.shape),
                     valid=jnp.asarray(pad.valid), is_center=jnp.zeros(pad.idx.shape, bool))
    ones = jnp.ones((len(S), quad.r.shape[0]))
    zl = zlambda(nb, jnp.full(len(S), 0.42, jnp.float32), ones, ones, quad, model, stage)
    lam = np.asarray(zl.rich.lam)
    assert np.median(lam / 40.0) == pytest.approx(1.0, abs=0.08)
    assert abs(np.median(np.asarray(zl.z)) - 0.4) < 0.005
    assert np.all(np.asarray(zl.z_e) > 0)
    # Without errors (first pass): same z_lambda and lambda, z_e = -1.
    zl2 = zlambda(nb, jnp.full(len(S), 0.42, jnp.float32), ones, ones, quad, model, stage, calc_err=False)
    np.testing.assert_array_equal(np.asarray(zl2.z), np.asarray(zl.z))
    np.testing.assert_array_equal(np.asarray(zl2.rich.lam), lam)
    assert np.all(np.asarray(zl2.z_e) == -1)
