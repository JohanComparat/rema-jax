"""Re-measurement at fixed centres in other cosmologies (tier A of the cosmology study)."""

import numpy as np
import pytest

from rema.config import CosmologyConfig, RemaConfig
from rema.model.cosmo import CosmoTable
from rema.model.profiles import MStar
from rema.model.redsequence import RSModel
from rema.modes import remeasure as R
from rema.modes.common import Region
from rema.sky.regions import Box
from rema.validate import mocks

SPECS = [(10.3, -0.25, 0.25, 40.0), (10.75, 0.2, 0.4, 35.0), (10.3, 0.25, 0.55, 50.0)]


@pytest.fixture(scope="module")
def mock():
    """Field + clusters whose members follow the NFW profile out to 2.5 r_lambda, each with a
    central galaxy (its first row); a catalogue of them as the blind mode would write it."""
    rng = np.random.default_rng(31)
    cfg = RemaConfig()
    rs = RSModel.from_template()
    ms, cosmo = MStar(cfg.model.mstar), CosmoTable.create()
    depth5 = np.array([24.9, 24.7, 24.2, 23.6])
    box = Box(10.0, 11.0, -0.5, 0.5)
    field = mocks.mock_field(rng, rs, box, density=12000, depth5=depth5, mag_range=(12.0, 22.5))
    cl = [mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, ra, dec, z, lam, depth5, poisson=False,
                             central_dmag=-1.5, extent=2.5) for ra, dec, z, lam in SPECS]
    gal = mocks.concat(field, *cl)
    n = gal["RA"].size
    gal["ID"] = np.arange(n, dtype=np.int64) + 1000
    gal["ZSPEC"] = np.full(n, -1.0, np.float32)
    central = field["RA"].size + np.cumsum([0] + [c["RA"].size for c in cl[:-1]])
    reg = Region.build(gal, rs, cfg, area_deg2=box.area_deg2())
    m = len(SPECS)
    cat = {"MEM_MATCH_ID": np.arange(m, dtype=np.int64), "SEED_ID": gal["ID"][central],
           "ID_CENT": np.stack([gal["ID"][central]] + [np.full(m, -1)] * 4, axis=1),
           "Z_LAMBDA_RAW": np.array([s[2] for s in SPECS]), "LAMBDA": np.array([s[3] for s in SPECS]),
           "LNLIKE": np.array([30.0, 20.0, 40.0])}
    return reg, cat


def test_centres_and_grid(mock, caplog):
    reg, cat = mock
    c = R.centres_from_catalog(reg, cat)
    np.testing.assert_array_equal(reg.gal["ID"][c.gi], cat["ID_CENT"][:, 0])
    np.testing.assert_allclose(c.z0, cat["Z_LAMBDA_RAW"])
    bad = dict(cat, ID_CENT=cat["ID_CENT"].copy())
    bad["ID_CENT"][1, 0] = 7
    c2 = R.centres_from_catalog(reg, bad)
    assert c2.size == 2 and list(c2.rows) == [0, 2] and "not in the galaxy table" in caplog.text
    grid = R.cosmology_grid(CosmologyConfig(), {"Omega_m": [0.35]})
    assert grid["Omega_m=0.35"].Omega_m == 0.35
    with pytest.raises(ValueError, match="unknown"):
        R.cosmology_grid(CosmologyConfig(), {"sigma8": [0.8]})


def test_response_to_distances(mock):
    """Fixed centres and redshifts: a smaller D_A puts the same galaxies at smaller radii, so lambda
    grows; the elasticity d ln(lambda)/d ln(D_A) is about -0.75 for these profiles."""
    reg, cat = mock
    c = R.centres_from_catalog(reg, cat)
    grid = R.cosmology_grid(reg.cfg.cosmology, {"Omega_m": [0.25, 0.35]})
    res = R.remeasure_cosmologies(reg, c, grid, fixed_z=True)
    fid = res["fiducial"]
    assert np.all((fid["LAMBDA"] > 0.6 * cat["LAMBDA"]) & (fid["LAMBDA"] < 1.5 * cat["LAMBDA"]))
    np.testing.assert_allclose(fid["R_LAMBDA"], (fid["LAMBDA"] / 100) ** 0.2, rtol=1e-5)
    lo, hi = res["Omega_m=0.25"]["LAMBDA"], res["Omega_m=0.35"]["LAMBDA"]
    assert np.all(hi > fid["LAMBDA"]) and np.all(lo < fid["LAMBDA"])
    z = c.z0
    dlnd = np.log(np.asarray(CosmoTable.create(Omega_m=0.35).da(z)) / np.asarray(CosmoTable.create(Omega_m=0.25).da(z)))
    # -gamma / (1 - beta gamma) for N(<R) ~ R^gamma; per cluster, the few members that cross the
    # aperture edge for a 2-5 % change of D_A make it noisy.
    elasticity = np.log(hi / lo) / dlnd
    assert np.all(elasticity < -0.1) and -1.0 < elasticity.mean() < -0.35, elasticity
    # z_lambda iterated from the true redshift stays close to it in every cosmology.
    zl = R.remeasure_cosmologies(reg, c, {"fiducial": reg.cfg.cosmology,
                                          "Omega_m=0.35": CosmologyConfig(Omega_m=0.35)})
    for v in zl.values():
        np.testing.assert_allclose(v["Z_LAMBDA_RAW"], z, atol=0.03)
        assert np.all(v["Z_LAMBDA_E"] > 0)


def test_jvp_matches_finite_differences(mock):
    reg, cat = mock
    c = R.centres_from_catalog(reg, cat)
    ad = R.response_jvp(reg, c, ("Omega_m", "w0", "Omega_b"))
    fd = R.response_fd(reg, c, {"Omega_m": 0.01, "w0": 0.03})
    for p in ("OMEGA_M", "W0"):
        np.testing.assert_allclose(ad[f"DLNLAMBDA_D{p}"], fd[f"DLNLAMBDA_D{p}_FD"], rtol=0.15, atol=0.02)
    assert np.all(ad["DLNLAMBDA_DOMEGA_M"] > 0) and np.all(ad["DLNLAMBDA_DW0"] > 0)
    # The baryons are not in the expansion history.
    np.testing.assert_array_equal(ad["DLNLAMBDA_DOMEGA_B"], 0.0)
    assert np.all(np.isfinite(ad["DLNLAMBDA_DZ"])) and np.all(np.isfinite(ad["LNLAMBDA"]))


def test_mstar_follows_cosmology(mock):
    """With m* following the luminosity distance, a larger Omega_m (smaller D_L) makes m* brighter:
    fewer galaxies above 0.2 L*, which pulls lambda down against the aperture effect."""
    reg, cat = mock
    c = R.centres_from_catalog(reg, cat)
    hi = {"Omega_m=0.35": CosmologyConfig(Omega_m=0.35)}
    plain = R.remeasure_cosmologies(reg, c, hi, fixed_z=True)["Omega_m=0.35"]
    follow = R.remeasure_cosmologies(reg, c, hi, fixed_z=True, mstar_follows=True)["Omega_m=0.35"]
    # At z = 0.25, m* moves by -0.013 mag only: less than one member crosses the limit.
    assert np.all(follow["LAMBDA"][1:] < plain["LAMBDA"][1:])
    assert abs(follow["LAMBDA"][0] / plain["LAMBDA"][0] - 1) < 0.01
    shifted = reg.with_cosmology(CosmologyConfig(Omega_m=0.35), mstar_follows=True, recompute_zred=False)
    assert shifted.mstar_np(0.3) < reg.mstar_np(0.3) - 0.01


def test_member_pfree(mock):
    """Members keep their PFREE; other neighbours lose the p of higher-ranked clusters' claims."""
    reg, cat = mock
    c = R.centres_from_catalog(reg, cat)
    ids = reg.gal["ID"]
    mem = {"MEM_MATCH_ID": np.array([0, 0, 2, 2, 1]), "ID": ids[[5, 6, 5, 7, 8]],
           "P": np.array([0.4, 0.9, 0.3, 0.8, 0.5]), "PFREE": np.array([0.7, 1.0, 1.0, 1.0, 1.0])}
    pf = R.MemberPfree(reg, cat, mem, c)
    idx = np.array([[5, 6, 7, 9], [5, 6, 7, 9], [5, 6, 7, 9]])
    valid = np.ones(idx.shape, bool)
    valid[0, 3] = False
    out = pf(np.array([0, 1, 2]), idx, valid)
    # Ranks by LNLIKE: cluster 2 (40), cluster 0 (30), cluster 1 (20).
    np.testing.assert_allclose(out[0], [0.7, 1.0, 1 - 0.8, 1.0])      # members of 0; 7 claimed by 2
    np.testing.assert_allclose(out[1], [1 - 0.4 - 0.3, 1 - 0.9, 1 - 0.8, 1.0])
    np.testing.assert_allclose(out[2], [1.0, 1.0, 1.0, 1.0])            # nobody ranks above 2
    lam1 = R.remeasure(reg, c, fixed_z=True)["LAMBDA"]
    lam2 = R.remeasure(reg, c, fixed_z=True, pfree=pf)["LAMBDA"]
    assert np.all(lam2 <= lam1 + 1e-4)


def test_galaxies_near(mock):
    reg, _ = mock
    g = R.galaxies_near(reg.gal, [10.3], [-0.25], [0.1])
    d = np.hypot((reg.gal["RA"] - 10.3) * np.cos(np.radians(-0.25)), reg.gal["DEC"] + 0.25)
    assert set(reg.gal["ID"][d < 0.1]) <= set(g["ID"])          # all of them, and not many more
    assert g["RA"].size < 2 * np.sum(d < 0.15)
    assert R.galaxies_near(reg.gal, [], [], [])["RA"].size == 0
