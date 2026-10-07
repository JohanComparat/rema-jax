"""Cluster abundance: area against z_vlim, data vector, mass-richness relation, counts model,
likelihood, Fisher matrix and fits (rema.abundance)."""

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from rema.abundance import fisher as FI
from rema.abundance import sampling as SA
from rema.abundance.area import ZvlimMap, zvlim_of
from rema.abundance.counts import CountsModel, CountsSetup
from rema.abundance.data import DataVector, build_data_vector
from rema.abundance.likelihood import DEFAULTS, Likelihood
from rema.abundance.mor import (LOG10_M0, MassRichness, mcclintock_cov, mcclintock_lnm, mean_lnlam,
                                p_bins, var_lnlam)
from rema.abundance.response import ResponseTable
from rema.model.cosmo import _float64
from rema.model.profiles import MStar

NSIDE = 64


@pytest.fixture(scope="module")
def zmap():
    """200 pixels of 0.84 deg2 with z_vlim from 0.3 to 0.9."""
    pix = np.arange(1000, 1200)
    area = np.full(pix.size, hp.nside2pixarea(NSIDE, degrees=True))
    return ZvlimMap(NSIDE, pix, area, np.linspace(0.3, 0.9, pix.size), {"DEPTH_Z10": np.linspace(22, 23, pix.size)})


@pytest.fixture(scope="module")
def data(zmap):
    rng = np.random.default_rng(8)
    n = 3000
    k = rng.integers(0, zmap.pix.size, n)
    ra, dec = hp.pix2ang(NSIDE, zmap.pix[k], nest=True, lonlat=True)
    cat = {"RA": ra, "DEC": dec, "LAMBDA": 20 * rng.pareto(2.0, n) + 20, "Z_LAMBDA": rng.uniform(0.1, 0.6, n),
           "Z_LAMBDA_E": rng.uniform(0.005, 0.02, n)}
    return cat, build_data_vector(cat, zmap, [20, 40, np.inf], [0.1, 0.3, 0.5])


def test_zvlim_map(zmap, tmp_path):
    a = zmap.area_above(np.array([0.2, 0.6, 1.0]))
    assert a[0] == pytest.approx(zmap.area.sum()) and a[2] == 0.0 and 0 < a[1] < a[0]
    assert zmap.area_above(0.6, mask=zmap.zvlim > 0.7) < a[1]
    ra, dec = zmap.radec()
    np.testing.assert_array_equal(zmap.index_of(ra[:5], dec[:5]), np.arange(5))
    np.testing.assert_allclose(zmap.value_at(ra[:3], dec[:3]), zmap.zvlim[:3])
    assert np.isnan(zmap.value_at([0.0], [-89.0])[0])
    np.testing.assert_allclose(zmap.value_at(ra[:2], dec[:2], "DEPTH_Z10"), zmap.extra["DEPTH_Z10"][:2])
    np.testing.assert_allclose(zmap.value_at(ra[:2], dec[:2], "AREA"), zmap.area[:2])
    zmap.write(tmp_path / "zv.fits")
    back = ZvlimMap.read(tmp_path / "zv.fits")
    assert back.nside == NSIDE and np.array_equal(back.pix, zmap.pix)
    d = tmp_path / "maps"
    d.mkdir()
    for k, v in {"PIX": zmap.pix, "AREA": zmap.area, "ZVLIM": zmap.zvlim}.items():
        np.save(d / f"{k}.npy", v)
    (tmp_path / "meta.json").write_text('{"nside": 64}')
    cols = ZvlimMap.from_columns(d)
    assert cols.nside == 64 and cols.area_above(0.6) == pytest.approx(a[1])
    sub = zmap.select(zmap.zvlim > 0.6)
    assert sub.pix.size < zmap.pix.size and sub.extra["DEPTH_Z10"].size == sub.pix.size
    full = zmap.full_mask()
    assert full.size == hp.nside2npix(NSIDE) and full.sum() == pytest.approx(zmap.pix.size)


def test_zvlim_of():
    ms = MStar("des_z03")
    depth = float(ms(0.5)) + 1.75
    assert float(zvlim_of(depth, ms)) == pytest.approx(0.5, abs=2e-3)
    assert np.isnan(zvlim_of(5.0, ms))


def test_data_vector(data, zmap, tmp_path):
    cat, dv = data
    zv = zmap.value_at(cat["RA"], cat["DEC"])
    hi = np.array([0.3, 0.5])
    for j in range(2):
        s = (cat["Z_LAMBDA"] >= [0.1, 0.3][j]) & (cat["Z_LAMBDA"] < hi[j]) & (zv >= hi[j])
        assert dv.counts[:, j].sum() == s.sum()
        assert dv.area[j] == pytest.approx(zmap.area_above(hi[j]))
    assert dv.shape == (2, 2) and np.all(dv.lam_mean[0] < 40) and np.all(dv.sigma_z > 0)
    # Pixels and clusters left out.
    dv2 = build_data_vector(cat, zmap, [20, 40, np.inf], [0.1, 0.3, 0.5], pix_keep=zmap.zvlim > 0.6,
                            cluster_keep=cat["LAMBDA"] < 30)
    assert dv2.counts[1].sum() == 0 and dv2.area[0] < dv.area[0]
    dv.write(tmp_path / "dv.fits")
    back = DataVector.read(tmp_path / "dv.fits")
    np.testing.assert_array_equal(back.counts, dv.counts)
    np.testing.assert_allclose(back.z_edges, dv.z_edges)
    assert back.meta["NCLUSTER"] == dv.meta["NCLUSTER"]


def test_mass_richness():
    mor = MassRichness(ln_s0=0.0, s1=0.0)
    # At the pivot, the inverted McClintock relation gives lambda = 40.
    assert float(mean_lnlam(mor, np.log(10**LOG10_M0), 0.35)) == pytest.approx(np.log(40.0))
    assert float(mcclintock_lnm(40.0, 0.35)) == pytest.approx(LOG10_M0 * np.log(10))
    p = p_bins(jnp.log(jnp.array([10.0, 30.0, 300.0])), jnp.array([0.04, 0.04, 0.04]),
               np.log([1e-3, 20, 40, 1e9]))
    np.testing.assert_allclose(np.asarray(p.sum(-1)), 1.0, atol=1e-6)
    assert float(var_lnlam(mor, jnp.log(1.0))) == pytest.approx(mor.sigma_int**2)
    c = mcclintock_cov([20.0, 40.0, 80.0, 30.0], [0.3, 0.35, 0.5, 0.6])
    w = np.linalg.eigvalsh(c)
    assert np.all(w > -1e-12) and np.sum(w > 1e-10) == 3


@pytest.fixture(scope="module")
def model(data):
    _, dv = data
    return CountsModel(CountsSetup.from_data(dv, nm=60, nk=256))


TH = {"Omega_m": 0.3, "ln10A_s": 3.044}


def test_counts_assembly(model):
    """All richness bins together count every halo: N_j = Omega_j int dV K_j int n dlnM."""
    s = model.setup
    with _float64():
        from ggah_mod.cosmology import comoving_volume_element

        from rema.abundance.counts import cosmology

        c = cosmology(TH)
        n, b = model.halo_tables(c)
        dv = np.asarray(comoving_volume_element(jnp.asarray(s.zt), c))
        one = CountsModel(CountsSetup(**{**s.__dict__, "lam_edges": np.array([1e-6, np.inf]),
                                         "sigma_z": s.sigma_z[:1], "lam_ref": s.lam_ref[:1]}), pk=model.pk)
        N = np.asarray(one.predict(TH)["N"])[0]
    inner = np.trapezoid(np.asarray(n), s.lnm, axis=1)
    for j in range(2):
        K = norm.cdf((s.z_edges[j + 1] - s.zt) / s.sigma_z[0]) - norm.cdf((s.z_edges[j] - s.zt) / s.sigma_z[0])
        ref = s.area_deg2[j] * (np.pi / 180) ** 2 * np.trapezoid(dv * K * inner, s.zt)
        assert N[j] == pytest.approx(ref, rel=1e-6)
    assert np.all(np.asarray(b) > 0.5)


def test_counts_response_equivalences(model):
    """A constant response is a shift of the richness normalisation (but for the Poisson part of
    the scatter, which stays that of the richness in the true cosmology); its Delta z a z bias."""
    resp = ResponseTable.constant({"Omega_m": 0.5}, dz={"Omega_m": 0.02})
    with_r = CountsModel(model.setup, response=resp, pk=model.pk)
    only_z = CountsModel(model.setup, response=ResponseTable.constant({"Omega_m": 0.0}, dz={"Omega_m": 0.02}),
                         pk=model.pk)
    th = {**TH, "Omega_m": 0.34}
    d = 0.04
    with _float64():
        a = with_r.predict(th)
        ref = model.predict({**th, "ln_s0": DEFAULTS["ln_s0"] - 0.5 * d, "dz_bias": -0.02 * d})
        fid = with_r.predict(TH)
        np.testing.assert_allclose(np.asarray(a["N"]), np.asarray(ref["N"]), rtol=1e-2)
        np.testing.assert_allclose(np.asarray(only_z.predict(th)["N"]),
                                   np.asarray(model.predict({**th, "dz_bias": -0.02 * d})["N"]), rtol=1e-8)
        np.testing.assert_allclose(np.asarray(fid["N"]), np.asarray(model.predict(TH)["N"]), rtol=1e-10)
        # Selection corrections multiply the counts.
        sel = CountsModel(model.setup, response=resp, pk=model.pk, selection=np.full((1, 2, 2), 2.0))
        np.testing.assert_allclose(np.asarray(sel.predict(th)["N"]), np.asarray(a["N"]) * np.exp(2.0 * d), rtol=1e-6)
        assert float(model.sigma8(TH)) == pytest.approx(0.817, abs=0.01)
    # More massive bins at higher richness.
    assert np.all(np.diff(np.asarray(a["lnM"]), axis=0) > 0)


@pytest.fixture(scope="module")
def lik(data, model):
    _, dv = data
    # Noise-free data: the model at a known point.
    truth = {**DEFAULTS, "Omega_m": 0.32, "mor_a": DEFAULTS["mor_a"] + 0.05}
    with _float64():
        pred = model.predict(truth)
    dv_true = DataVector(dv.lam_edges, dv.z_edges, np.asarray(pred["N"]), dv.area,
                         np.array([[28.0, 28.0], [60.0, 60.0]]), np.array([[0.25, 0.4], [0.25, 0.4]]), dv.sigma_z)
    return Likelihood(dv_true, model, ["Omega_m", "mor_a"], fixed={"ln10A_s": 3.044}, reference=truth), truth


def test_likelihood_and_fit(lik):
    L, truth = lik
    x_true = np.array([truth["Omega_m"], truth["mor_a"]])
    c2 = L.chi2(x_true)
    assert c2["counts"] == pytest.approx(0.0, abs=1e-6) and c2["n_wl"] == 4 and c2["wl"] > 0
    assert L.logpost(np.array([0.05, 3.7])) == -np.inf             # outside the prior / emulator box
    v, g = L.value_and_grad(L.x0())
    assert np.isfinite(v) and g.shape == (2,)
    # Without the weak-lensing term the noise-free counts are fitted exactly.
    L0 = Likelihood(L.data, L.model, ["Omega_m", "mor_a"], fixed={"ln10A_s": 3.044}, wl=False,
                    reference=truth)
    x, info = SA.map_fit(L0)
    np.testing.assert_allclose(x, x_true, atol=2e-3)
    assert info["success"] and info["at_bound"] == []
    cov = SA.laplace(L0, x)
    F = FI.fisher(L0, x)
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), np.sqrt(np.diag(np.linalg.inv(F))), rtol=0.05)


def test_fisher_and_shift(lik):
    L, truth = lik
    L0 = Likelihood(L.data, L.model, ["Omega_m", "mor_a"], fixed={"ln10A_s": 3.044}, wl=False,
                    reference=truth)
    x = np.array([truth["Omega_m"], truth["mor_a"]])
    jac = FI.jacobians(L0, x)
    h = 1e-4
    up = np.asarray(L0.predict(L0.theta(x + [h, 0]))["N"]).ravel()
    dn = np.asarray(L0.predict(L0.theta(x - [h, 0]))["N"]).ravel()
    np.testing.assert_allclose(jac["N"][:, 0], (up - dn) / (2 * h), rtol=1e-4)
    assert "WL" not in jac and jac["sigma8_value"] > 0
    F = FI.fisher(L0, x, jac)
    c = FI.constraints(F, L0.free)
    assert 0 < c["Omega_m"] < 0.2 and FI.sigma8_error(jac, c["_cov"]) > 0
    # A small change of the data moves the best fit as F^-1 J^T C^-1 delta.
    x2 = x + np.array([0.003, -0.01])
    delta = np.asarray(L0.predict(L0.theta(x2))["N"]).ravel() - np.asarray(L0.predict(L0.theta(x))["N"]).ravel()
    np.testing.assert_allclose(FI.shift(L0, F, jac, delta), [0.003, -0.01], rtol=0.05, atol=2e-4)
    # Poisson likelihood and the weak-lensing Fisher term.
    Lp = Likelihood(L.data, L.model, ["Omega_m", "mor_a"], fixed={"ln10A_s": 3.044}, kind="poisson")
    assert np.isfinite(Lp.logpost(x)) and np.isnan(Lp.chi2(x)["counts"])
    Fp = FI.fisher(Lp, x)
    assert np.all(np.linalg.eigvalsh(Fp) > 0)
    assert FI.shift(Lp, Fp, FI.jacobians(Lp, x), delta).shape == (2,)


def test_priors_and_errors(lik):
    L, _ = lik
    L2 = Likelihood(L.data, L.model, ["Omega_m", "h", "Omega_b", "ln_s0"], fixed={"ln10A_s": 3.044},
                    cov=np.eye(4) * 100.0)
    b = dict(zip(L2.free, L2.bounds()))
    x0 = dict(zip(L2.free, L2.x0()))
    # h starts at the prior mean (0.7 is 5 sigma away); Omega_b = 0.0493 is BBN's at that h.
    assert x0["h"] == 0.6736 and x0["Omega_b"] == 0.0493
    L3 = Likelihood(L.data, L.model, ["Omega_b"], fixed={"ln10A_s": 3.044}, cov=np.eye(4))
    assert L3.x0()[0] * 0.7**2 == pytest.approx(0.02237)            # at h = 0.7 it moves
    for p in L2.free:
        assert b[p][0] < x0[p] < b[p][1]
    F = FI.prior_fisher(L2, L2.x0())
    assert F[1, 1] == pytest.approx(1 / 0.0054**2) and F[2, 2] > 0 and F[0, 0] == 0
    with pytest.raises(ValueError, match="unknown parameter"):
        Likelihood(L.data, L.model, ["sigma8"])
    with pytest.raises(ValueError, match="kind"):
        Likelihood(L.data, L.model, ["Omega_m"], kind="chi")


@pytest.mark.slow
def test_nuts_smoke(lik):
    pytest.importorskip("blackjax")
    L, truth = lik
    out = SA.nuts(L, warmup=30, samples=20, seed=1)
    assert out["samples"].shape == (20, 2) and np.all(np.isfinite(out["logpost"]))
    lo, hi = zip(*L.bounds())
    assert np.all(out["samples"] > lo) and np.all(out["samples"] < hi)


def test_covariance(zmap, model):
    from rema.abundance.covariance import counts_covariance, sigma2_b_slabs

    s2 = sigma2_b_slabs(zmap, [0.1, 0.3, 0.5], TH, pk=model.pk, ell_max=96)
    assert np.all(s2 > 0)
    # A smaller footprint has a larger variance.
    half = sigma2_b_slabs(zmap, [0.1, 0.3, 0.5], TH, pk=model.pk, pix_keep=np.arange(zmap.pix.size) < 100, ell_max=96)
    assert np.all(half > s2)
    empty = sigma2_b_slabs(zmap, [0.1, 0.95], TH, pk=model.pk, ell_max=96)
    assert empty[0] == 0.0                              # no pixel reaches z_vlim = 0.95
    N, b = np.array([[100.0, 200.0], [10.0, 20.0]]), np.array([[2.0, 2.5], [3.0, 3.5]])
    C = counts_covariance(N, b, np.array([1e-3, 2e-3]))
    assert C.shape == (4, 4) and np.allclose(C, C.T) and np.all(np.linalg.eigvalsh(C) > 0)
    assert C[0, 0] == pytest.approx(100 + 1e-3 * 200**2) and C[0, 2] == pytest.approx(1e-3 * 200 * 30)
    assert C[0, 1] == 0.0                               # different redshift slabs
    np.testing.assert_array_equal(counts_covariance(N, b), np.diag(N.ravel()))
