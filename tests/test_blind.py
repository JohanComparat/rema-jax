"""Blind mode: exactness of batched percolation, recovery of mock clusters."""

import numpy as np
import pytest

from rema.config import RemaConfig
from rema.core.richness import RadialQuad, Stage
from rema.model.profiles import MStar
from rema.model.redsequence import RSModel
from rema.modes import blind as B
from rema.modes.common import Region
from rema.model.cosmo import CosmoTable
from rema.sky.regions import Box
from rema.validate import mocks


@pytest.fixture(scope="module")
def mock_region():
    rng = np.random.default_rng(21)
    cfg = RemaConfig()
    rs = RSModel.from_template()
    ms, cosmo = MStar(cfg.model.mstar), CosmoTable.create()
    depth5 = np.array([24.9, 24.7, 24.2, 23.6])
    box = Box(10.0, 12.0, -1.0, 1.0)
    field = mocks.mock_field(rng, rs, box, density=12000, depth5=depth5, mag_range=(12.0, 22.5))
    specs = [(10.5, -0.5, 0.3, 40.0), (11.4, 0.4, 0.5, 30.0), (10.6, 0.5, 0.25, 25.0), (11.5, -0.5, 0.6, 35.0)]
    cl = [mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, ra, dec, z, lam, depth5, poisson=False)
          for ra, dec, z, lam in specs]
    gal = mocks.concat(field, *cl)
    n = gal["RA"].size
    gal["ID"] = np.arange(n, dtype=np.int64)
    gal["ZSPEC"] = np.full(n, -1.0, np.float32)
    reg = Region.build(gal, rs, cfg, area_deg2=box.area_deg2())
    return reg, specs


def test_region_with_cosmology(mock_region):
    """Another cosmology: new distances and zred, same galaxies; the fiducial gives the same lambda."""
    reg, specs = mock_region
    d0 = reg.mpc_per_deg_np(0.5)                       # fills the host-table cache
    gi = np.array([int(np.argmin(np.hypot(reg.gal["RA"] - ra, reg.gal["DEC"] - dec)))
                   for ra, dec, _, _ in specs])
    z0 = np.array([z for _, _, z, _ in specs])
    st, q = Stage.make(1.0, 0.2), RadialQuad.make(rmax=1.0 * 20**0.2 + 0.25)
    same = reg.with_cosmology(recompute_zred=False)
    assert same.gal is reg.gal and same.cfg == reg.cfg
    a = B._run_batched(reg, gi, z0, st, q, "richness", log_every=0)
    b = B._run_batched(same, gi, z0, st, q, "richness", log_every=0)
    np.testing.assert_array_equal(a["LAMBDA"], b["LAMBDA"])
    hi = reg.with_cosmology({"Omega_m": 0.35})
    assert hi.cfg.cosmology.Omega_m == 0.35 and reg.cfg.cosmology.Omega_m == 0.3
    assert hi.mpc_per_deg_np(0.5) < d0 == reg.mpc_per_deg_np(0.5)
    # zred sees the cosmology through E(z), weakly.
    dz = np.abs(hi.gal["ZRED"] - reg.gal["ZRED"])
    assert dz.max() > 0 and np.median(dz) < 0.01
    c = B._run_batched(hi, gi, z0, st, q, "richness", log_every=0)
    assert np.all(c["LAMBDA"] != a["LAMBDA"]) and np.all(np.abs(np.log(c["LAMBDA"] / a["LAMBDA"])) < 0.2)
    # R_lambda in Mpc follows lambda; the aperture in degrees grows as D_A falls.
    np.testing.assert_allclose(c["R_LAMBDA"], (c["LAMBDA"] / 100) ** 0.2, rtol=1e-5)


@pytest.mark.slow
def test_batched_percolation_is_exact(mock_region):
    reg, _ = mock_region
    seeds = B.select_seeds(reg)
    rng = np.random.default_rng(0)
    pick = rng.choice(seeds, min(150, seeds.size), replace=False)
    cand = {"GI": pick, "Z_LAMBDA": reg.gal["ZRED"][pick].astype(np.float32),
            "LAMBDA": np.full(pick.size, 10.0), "LNLIKE": rng.normal(size=pick.size)}
    st, q = Stage.make(1.0, 0.2), RadialQuad.make(rmax=1.0 * 20**0.2 + 0.25)
    a, _ = B.percolate(reg, cand, st, q, batch=1, log_every=0)
    b, _ = B.percolate(reg, cand, st, q, batch=8, log_every=0)
    # Same clusters in the same order; values agree to float32 precision (the two batch shapes
    # compile to different reduction orders, which moves lambda ~ 3 by up to ~1e-4).
    assert np.array_equal(a["SEED_ID"], b["SEED_ID"])
    np.testing.assert_allclose(a["LAMBDA"], b["LAMBDA"], rtol=2e-4)
    np.testing.assert_allclose(a["Z_LAMBDA"], b["Z_LAMBDA"], atol=1e-4)


@pytest.mark.slow
def test_blind_recovers_mock_clusters(mock_region):
    reg, specs = mock_region
    cat, mem = B.run_blind(reg, own=Box(10.0, 12.0, -1.0, 1.0))
    for ra, dec, z, lam in specs:
        d = np.hypot((cat["RA"] - ra) * np.cos(np.radians(dec)), cat["DEC"] - dec)
        near = np.flatnonzero(d < 0.05)
        assert near.size
        # one detection per cluster (no fragments re-centred on members of the cluster)
        assert np.sum(cat["LAMBDA"][near] > 10) == 1
        i = near[np.argmax(cat["LAMBDA"][near])]
        assert abs(cat["Z_LAMBDA"][i] - z) < 0.02
        assert 0.6 * lam < cat["LAMBDA"][i] < 1.5 * lam
    assert set(np.unique(mem["MEM_MATCH_ID"])) <= set(cat["MEM_MATCH_ID"])


@pytest.mark.slow
def test_blind_wcen_centres_on_mock_centrals():
    """With a wcen model, blind clusters are centred on the injected central galaxies."""
    rng = np.random.default_rng(5)
    cfg = RemaConfig().replace(centering={"method": "wcen"})
    rs = RSModel.from_template()
    ms, cosmo = MStar(cfg.model.mstar), CosmoTable.create()
    depth5 = np.array([24.9, 24.7, 24.2, 23.6])
    box = Box(10.0, 12.0, -1.0, 1.0)
    field = mocks.mock_field(rng, rs, box, density=12000, depth5=depth5, mag_range=(12.0, 22.5))
    specs = [(10.5, -0.5, 0.3, 40.0), (11.4, 0.4, 0.5, 30.0)]
    cl = [mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, ra, dec, z, lam, depth5, poisson=False,
                             central_dmag=-1.5) for ra, dec, z, lam in specs]
    gal = mocks.concat(field, *cl)
    n = gal["RA"].size
    gal["ID"] = np.arange(n, dtype=np.int64)
    gal["ZSPEC"] = np.full(n, -1.0, np.float32)
    # The first row of every mock cluster is its central.
    central = field["RA"].size + np.cumsum([0] + [c["RA"].size for c in cl[:-1]])
    params = {"DELTA0": -1.5, "DELTA1": 0.0, "SIGMA_M": 0.3, "LNW_CEN_MEAN": 0.3,
              "LNW_CEN_SIGMA": 0.25, "LNW_SAT_MEAN": -0.2, "LNW_SAT_SIGMA": 0.35,
              "LNW_FG_MEAN": -0.6, "LNW_FG_SIGMA": 0.45}
    reg = Region.build(gal, rs, cfg, area_deg2=box.area_deg2(), wcen_params=params)
    assert reg.centering_method() == "wcen"
    cat, mem = B.run_blind(reg, own=box)
    for (ra, dec, z, lam), c in zip(specs, central):
        d = np.hypot((cat["RA"] - ra) * np.cos(np.radians(dec)), cat["DEC"] - dec)
        i = int(np.argmin(np.where(cat["LAMBDA"] > 10, d, np.inf)))
        assert cat["ID_CENT"][i, 0] == gal["ID"][c]
        # the most probable candidate (another bright member may share the probability)
        assert cat["P_CEN"][i, 0] == cat["P_CEN"][i].max() and cat["P_CEN"][i, 0] > 0.3
        assert cat["NCENT_GOOD"][i] >= 1
        assert np.isfinite(cat["W"][i]) and np.isfinite(cat["LNCGLIKE"][i])
        assert abs(cat["Z_LAMBDA"][i] - z) < 0.02
    # The centre candidates are in the member table.
    assert np.any(mem["CENT_RANK"] == 0)


@pytest.mark.slow
def test_regions_merge_like_one_region(mock_region):
    """Two regions split at RA 11 give the one-region catalogue: exactly when each region reads
    the whole field, and with differences only near the split when it reads a 0.3 deg buffer."""
    reg, _ = mock_region
    field = Box(10.0, 12.0, -1.0, 1.0)
    one, _ = B.run_blind(reg, own=field)
    halves = [Box(10.0, 11.0, -1.0, 1.0), Box(11.0, 12.0, -1.0, 1.0)]
    full = [B.run_blind(reg, own=h, region_id=i + 1)[0] for i, h in enumerate(halves)]
    merged = {k: np.concatenate([c[k] for c in full]) for k in one}
    o1, o2 = np.argsort(one["ID_CENT"][:, 0]), np.argsort(merged["ID_CENT"][:, 0])
    np.testing.assert_array_equal(one["ID_CENT"][o1, 0], merged["ID_CENT"][o2, 0])
    np.testing.assert_array_equal(one["LAMBDA"][o1], merged["LAMBDA"][o2])
    assert np.all((merged["MEM_MATCH_ID"] >> 32) == np.repeat([1, 2], [len(c["RA"]) for c in full]))

    # Each region with only a 0.3 deg buffer (same zred and background as the field).
    cuts = []
    for i, h in enumerate(halves):
        data = h.buffered(0.3)
        sel = data.contains(reg.gal["RA"], reg.gal["DEC"])
        sub = Region.build({k: v[sel] for k, v in reg.gal.items()}, reg.rs, reg.cfg, bkg=reg.model.bkg)
        cuts.append(B.run_blind(sub, own=h, region_id=i + 1)[0])
    part = {k: np.concatenate([c[k] for c in cuts]) for k in one}
    big = one["LAMBDA"] >= 5
    same = np.isin(one["ID_CENT"][:, 0], part["ID_CENT"][:, 0])
    far = np.abs(one["RA"] - 11.0) > 0.35
    assert np.all(same[big & far])                       # far from the split: the same clusters
    j = {c: n for n, c in enumerate(part["ID_CENT"][:, 0])}
    k = np.flatnonzero(big & far)
    np.testing.assert_allclose(part["LAMBDA"][[j[c] for c in one["ID_CENT"][k, 0]]], one["LAMBDA"][k],
                               rtol=2e-4)


@pytest.mark.slow
def test_checkpoint_key_covers_the_inputs(mock_region, tmp_path):
    from rema.sky.maps import build_footprint

    reg, _ = mock_region
    seeds = B.select_seeds(reg)
    key = B._Checkpoint.for_run(tmp_path, reg, seeds).key
    assert B._Checkpoint.for_run(tmp_path, reg, seeds).key == key
    # Another footprint.
    rnd = {"RA": np.array([10.5, 11.5]), "DEC": np.array([0.0, 0.0]), "MASKBITS": np.zeros(2, np.int32),
           "EBV": np.zeros(2, np.float32)}
    for b in "GRIZ":
        rnd[f"NOBS_{b}"] = np.ones(2, np.int16)
        rnd[f"GALDEPTH_{b}"] = np.ones(2, np.float32) * 100
    import dataclasses
    with_fp = dataclasses.replace(reg, footprint=build_footprint(rnd, reg.cfg))
    assert B._Checkpoint.for_run(tmp_path, with_fp, seeds).key != key
    # A galaxy inserted before the seeds (same seed IDs, other row indices).
    gal = {k: np.concatenate([v[:1], v]) for k, v in reg.gal.items()}
    gal["ID"] = np.concatenate([[-1], reg.gal["ID"]])
    shifted = dataclasses.replace(reg, gal=gal)
    assert B._Checkpoint.for_run(tmp_path, shifted, seeds + 1).key != key
