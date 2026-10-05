"""Region bookkeeping (neighbour queries, completeness, z_lambda correction, centring choice),
empty blind and scan runs, and edge cases of the region planner, status and merge."""

import jax.numpy as jnp
import numpy as np
import pytest

from rema.calibration import ZlambdaCorrection
from rema.config import RemaConfig
from rema.core.richness import RadialQuad
from rema.io.tables import write_catalog
from rema.model.background import ChisqBkg
from rema.model.redsequence import RSModel
from rema.modes.blind import consolidate, run_blind
from rema.modes.common import Region
from rema.modes.scan import run_scan
from rema.pipeline import (_zstats, calib_suggest, close_pairs, plan_regions, region_status,
                           uncovered_tiles, write_plan)
from rema.sky.neighbors import unit_vectors
from rema.sky.regions import Box


def light_region(n=300, seed=8, **kw):
    """A Region with precomputed zred columns and a flat chi^2 background (nothing to fit)."""
    rng = np.random.default_rng(seed)
    gal = {"ID": np.arange(n, dtype=np.int64), "RA": rng.uniform(10, 11, n), "DEC": rng.uniform(0, 1, n),
           "FLUX": rng.uniform(1, 10, (n, 4)).astype(np.float32),
           "FLUX_IVAR": np.full((n, 4), 100.0, np.float32),
           "REFMAG": rng.uniform(17, 21, n).astype(np.float32),
           "REFMAG_ERR": np.full(n, 0.02, np.float32)}
    for k in ("ZRED", "ZRED_UNCORR"):
        gal[k] = np.full(n, -1.0, np.float32)
    for k in ("ZRED_E", "ZRED_UNCORR_E", "ZRED_CHISQ"):
        gal[k] = np.full(n, 1.0, np.float32)
    bkg = ChisqBkg(jnp.linspace(0.05, 0.9, 5), jnp.linspace(0.25, 19.75, 40), jnp.linspace(12.1, 23.9, 60),
                   jnp.ones((5, 40, 60)), jnp.ones(60))
    return Region.build(gal, RSModel.from_template(), RemaConfig(), bkg=bkg, area_deg2=1.0, **kw)


def test_region_bookkeeping():
    nan_wcen = dict.fromkeys(("DELTA0", "SIGMA_M"), np.nan)
    reg = light_region(wcen_params=nan_wcen)                 # not calibrated: no wcen model
    assert reg.wcen is None and reg.centering_method() == "bcg"
    with pytest.raises(ValueError, match="no wcen model"):
        reg.centering_method("wcen")
    with pytest.raises(ValueError, match="unknown centring"):
        reg.centering_method("median")
    # Neighbour queries, all galaxies or the bright ones only.
    pad = reg.query([10.5], [0.5], 0.2)
    cosd = unit_vectors(reg.gal["RA"], reg.gal["DEC"]) @ unit_vectors(10.5, 0.5)
    inside = np.flatnonzero(np.degrees(np.arccos(np.clip(cosd, -1, 1))) < 0.2)
    assert sorted(pad.idx[pad.valid]) == sorted(inside)
    bright = reg.query([10.5], [0.5], 0.2, mag_max=19.0)
    assert np.all(reg.gal["REFMAG"][bright.idx[bright.valid]] < 19.0)
    assert set(bright.idx[bright.valid]) <= set(pad.idx[pad.valid])
    for m in (17.5, 18.0, 18.5, 19.5):
        reg._subindex(m, max_cached=2)
    assert len(reg._sub) == 2
    # Neighbour arrays: per-neighbour or per-galaxy pfree, and the centre flag.
    ids = reg.gal["ID"][pad.idx[0, :1]]
    nb = reg.neighbors(pad, pfree_nb=np.full(pad.idx.shape, 0.5), center_ids=ids)
    assert float(nb.pfree[0, 0]) == 0.5 and bool(nb.is_center[0, 0]) and int(nb.is_center.sum()) == 1
    nb = reg.neighbors(pad, pfree=np.full(reg.gal["ID"].size, 0.25))
    assert float(nb.pfree[0, 0]) == 0.25
    # Completeness without a footprint is cached.
    q = RadialQuad.make()
    assert reg.completeness([1.0], [1.0], [0.3], q) is reg.completeness([2.0], [2.0], [0.4], q)
    # z_lambda correction, with failed values kept.
    assert reg.correct_zlambda([0.3], [0.01], [20.0])[0][0] == pytest.approx(0.3)
    nodes = jnp.linspace(0.05, 0.9, 5)
    reg.zlcorr = ZlambdaCorrection(nodes, jnp.full(5, 0.01), jnp.zeros(5), jnp.zeros(5))
    z, ze = reg.correct_zlambda([0.3, -1.0], [0.01, 0.02], [20.0, 20.0])
    np.testing.assert_allclose(z, [0.31, -1.0], atol=1e-6)
    np.testing.assert_allclose(ze, [0.01, 0.02], atol=1e-6)


def test_empty_blind_and_scan():
    reg = light_region()                                   # every zred failed: no seeds
    info = {}
    cat, mem = run_blind(reg, own=Box(10, 11, 0, 1), info=info)
    assert cat == {} and mem == {} and info["n_seeds"] == 0 and info["n_clusters"] == 0
    assert consolidate(reg, {}, {}) == ({}, {})
    assert run_scan(reg, np.zeros(0), np.zeros(0)) == ({}, {})


def _counts(rows):
    names = []
    for ra0, dec0, n in rows:
        d = lambda v: f"{'m' if v < 0 else 'p'}{abs(int(v)):03d}"
        names.append(f"sweep-{int(ra0):03d}{d(dec0)}-{int(ra0 + 5):03d}{d(dec0 + 5)}.fits")
    a = np.array(rows, float)
    return {"NAME": np.array(names), "RA0": a[:, 0], "DEC0": a[:, 1], "RA1": a[:, 0] + 5,
            "DEC1": a[:, 1] + 5, "NGAL": a[:, 2], "NZSPEC": a[:, 2] / 50, "EBVMEAN": np.full(len(rows), 0.02)}


def test_plan_edge_cases(caplog):
    with pytest.raises(ValueError, match="no tile"):
        plan_regions(_counts([(0, 0, 0)]))
    # A target area larger than the band: one ring per band.
    plan, meta = plan_regions(_counts([(ra, 0, 1e5) for ra in (0, 100, 200)]), target_area=5000,
                              band_height=5)
    assert len(plan["REGION_ID"]) == 1 and plan["OWN_RA1"][0] - plan["OWN_RA0"][0] == 360
    # A capped ring is split in RA; a single tile above the caps stays, with a warning.
    plan2, _ = plan_regions(_counts([(ra, 0, 1e5) for ra in (0, 100, 200)]), target_area=5000,
                            band_height=5, max_gal=1.5e5)
    assert len(plan2["REGION_ID"]) == 3
    plan3, _ = plan_regions(_counts([(0, 0, 1e6)]), max_gal=10)
    assert len(plan3["REGION_ID"]) == 1 and "single tile" in caplog.text
    # Dec rows are split when RA cannot be.
    plan4, _ = plan_regions(_counts([(0, 0, 1e6), (0, 5, 1e6)]), band_height=10, target_area=10,
                            max_gal=1.5e6, buffer=0.5)
    assert len(plan4["REGION_ID"]) == 2
    # Calibration areas: at most n.
    many = _counts([(ra, dec, 1e5) for ra in range(0, 50, 5) for dec in (-10, -5, 0)])
    assert len(calib_suggest(many, area=25, n=2)) == 2


def test_status_close_pairs_and_zstats(tmp_path):
    counts = _counts([(0, 0, 1e5), (5, 0, 1e5)])
    plan, meta = plan_regions(counts, target_area=25, band_height=5)
    write_plan(tmp_path / "plan.fits", plan, meta)
    for r in plan["REGION_ID"]:
        write_catalog(tmp_path / "runs" / f"{int(r):04d}" / "clusters.fits", {}, {}, None,
                      {"PLANHASH": meta["PLANHASH"]})
    st = region_status(tmp_path / "plan.fits", tmp_path / "runs")
    assert st["prime"] is None and len(st["done"]) == 2
    assert uncovered_tiles(counts, tmp_path / "no_index") == []
    i, j = close_pairs({"RA": np.array([1.0]), "DEC": np.array([0.0]), "Z_LAMBDA": np.array([0.3])})
    assert i.size == j.size == 0
    i, j = close_pairs({"RA": np.array([1.0, 3.0]), "DEC": np.zeros(2), "Z_LAMBDA": np.full(2, 0.3)})
    assert i.size == 0
    nz, bias, nmad = _zstats({})
    assert nz == 0 and np.isnan(bias) and np.isnan(nmad)
    n = 6
    cat = {"LAMBDA": np.full(n, 30.0), "N_MEMBERS": np.full(n, 5), "Z_LAMBDA": np.full(n, 0.31),
           "SPEC_Z_BOOT": np.array([0.3] * 5 + [np.nan])}
    nz, bias, nmad = _zstats(cat)
    assert nz == 5 and bias == pytest.approx(0.01 / 1.3) and nmad == pytest.approx(0.0, abs=1e-12)
    nz, bias, _ = _zstats({k: v[:2] for k, v in cat.items()})
    assert nz == 2 and np.isnan(bias)
