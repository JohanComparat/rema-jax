"""The finder's response table (rema.abundance.response)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rema.abundance import response as RT
from rema.config import CosmologyConfig
from rema.modes import remeasure as R


def fake_remeasure(path, rng, n=4000, d1=lambda z, l: 0.4 + 0.5 * z, d2=0.8, dz=0.02):
    """A remeasure file: Omega_m 0.25 / 0.35 around 0.3, ln lambda shifted by d1 dOm + d2 dOm^2/2."""
    grid = R.cosmology_grid(CosmologyConfig(), {"Omega_m": [0.25, 0.35], "w0": [-0.9]})
    z = rng.uniform(0.1, 0.6, n)
    lam = np.exp(rng.uniform(np.log(10), np.log(150), n))
    dth = np.array([0.0, -0.05, 0.05, 0.0])
    lnl = (np.log(lam)[:, None] + d1(z, lam)[:, None] * dth + 0.5 * d2 * dth**2
           + rng.normal(0, 0.002, (n, 4)) * (dth != 0))
    lnl[:, 3] = np.log(lam) + 0.1 * 0.1                # w0 -0.9: d ln lambda / d w0 = 0.1
    res = {lab: {k: np.zeros(n, np.float32) for k in R.COLUMNS} for lab in grid}
    for j, lab in enumerate(grid):
        res[lab]["LAMBDA"] = np.exp(lnl[:, j]).astype(np.float32)
        res[lab]["Z_LAMBDA"] = (z + dz * dth[j]).astype(np.float32)
    cat = {"ID_CENT": np.arange(n), "MEM_MATCH_ID": np.arange(n), "LAMBDA": lam, "Z_LAMBDA": z}
    c = R.Centres(rows=np.arange(n), gi=np.arange(n), z0=z)
    R.write_remeasure(path, cat, c, res, grid, header={"PFREE": "one"})
    return path


def test_from_remeasure_recovers_coefficients(tmp_path):
    rng = np.random.default_rng(3)
    paths = [fake_remeasure(tmp_path / f"r{i}.fits", rng) for i in range(2)]
    tab, st = RT.from_remeasure(paths, [0.1, 0.2, 0.3, 0.45, 0.6], [10, 20, 40, 80, 150])
    assert tab.params == ("Omega_m", "w0") and tab.fiducial == (0.3, -1.0)
    k = tab.params.index("Omega_m")
    np.testing.assert_allclose(np.asarray(tab.d1[k]), 0.4 + 0.5 * st["z"][:, None] * np.ones((1, 4)), atol=0.01)
    np.testing.assert_allclose(np.asarray(tab.d2[k]), 0.8, atol=0.1)
    np.testing.assert_allclose(np.asarray(tab.dz[k]), 0.02, atol=1e-3)
    np.testing.assert_allclose(np.asarray(tab.d1[1]), 0.1, atol=1e-4)
    assert st["N"].sum() == 8000 and np.all(st["D1_NMAD"][k] < 0.06)       # noise 0.002 / 0.05
    # FITS round trip; evaluation (jitted) and its derivative in theta.
    tab.write(tmp_path / "resp.fits")
    back = RT.ResponseTable.read(tmp_path / "resp.fits")
    assert back.params == tab.params and back.fiducial == tab.fiducial
    np.testing.assert_allclose(np.asarray(back.d1), np.asarray(tab.d1))
    f = jax.jit(lambda t, om: RT.delta_lnlam(t, jnp.log(30.0), 0.3, t.dtheta({"Omega_m": om})))
    v = float(f(back, 0.35))
    assert v == pytest.approx((0.4 + 0.5 * 0.3) * 0.05 + 0.5 * 0.8 * 0.05**2, abs=2e-3)
    g = float(jax.grad(lambda om: f(back, om))(0.3))
    assert g == pytest.approx(0.4 + 0.5 * 0.3, abs=0.01)
    assert float(RT.delta_z(back, jnp.log(30.0), 0.3, back.dtheta({"Omega_m": 0.4}))) == pytest.approx(0.002, abs=2e-4)
    # Outside the nodes the table is clamped.
    assert np.isfinite(float(RT.delta_lnlam(back, jnp.log(1000.0), 1.5, back.dtheta({"Omega_m": 0.35}))))


def test_constant_and_zero_tables():
    c = RT.ResponseTable.constant({"Omega_m": 0.6}, dz={"Omega_m": 0.01})
    d = c.dtheta({"Omega_m": 0.4})
    np.testing.assert_allclose(np.asarray(RT.delta_lnlam(c, jnp.log(jnp.array([10.0, 100.0])), 0.5, d)), 0.06, rtol=1e-6)
    assert float(RT.delta_z(c, 3.0, 0.5, d)) == pytest.approx(0.001)
    z = RT.ResponseTable.zero()
    assert float(RT.delta_lnlam(z, 3.0, 0.5, z.dtheta({"Omega_m": 0.5}))) == 0.0


def test_errors(tmp_path):
    rng = np.random.default_rng(4)
    p = fake_remeasure(tmp_path / "a.fits", rng, n=50)
    with pytest.raises(ValueError, match="no \\(z, lambda\\) bin"):
        RT.from_remeasure([p], [0.1, 0.6], [10, 150], min_count=100)
    with pytest.raises(ValueError, match="no re-measurement"):
        RT.from_remeasure([], [0.1, 0.6], [10, 150])
    grid = {"fiducial": CosmologyConfig()}
    n = 3
    res = {"fiducial": {k: np.ones(n, np.float32) for k in R.COLUMNS}}
    cat = {"ID_CENT": np.arange(n), "LAMBDA": np.ones(n), "Z_LAMBDA": np.ones(n)}
    R.write_remeasure(tmp_path / "b.fits", cat, R.Centres(np.arange(n), np.arange(n), np.ones(n)), res, grid)
    with pytest.raises(ValueError, match="no one-at-a-time"):
        RT.from_remeasure(tmp_path / "b.fits", [0.1, 0.6], [10, 150])
    with pytest.raises(ValueError, match="differ"):
        RT.from_remeasure([p, tmp_path / "b.fits"], [0.1, 0.6], [10, 150])
