"""Aperture completeness of clusters at random positions: the values the cluster finder records
(SCALEVAL = 1/K, MASKFRAC = 1 - K_geo), from the footprint alone."""

import numpy as np
import pytest

from conftest import DR11_DIR, require
from rema.config import RemaConfig
from rema.core.richness import RadialQuad, Stage, percolation_quad
from rema.model.cosmo import CosmoTable
from rema.model.profiles import MStar
from rema.model.redsequence import RSModel
from rema.modes import blind as B
from rema.modes.common import Region
from rema.randoms.aperture import aperture_model, aperture_values, footprint_map
from rema.randoms.config import RandomsConfig
from rema.sky.maps import SparseMap, build_footprint, depth_ivar_from_mag
from rema.sky.regions import Box
from rema.validate import mocks

DEPTH5 = np.array([24.9, 24.7, 24.2, 23.6])


def _randoms(rng, box, n, masked, shallow=lambda ra, dec: np.zeros(ra.size, bool)):
    """Uniform randoms in ``box``: ``masked`` positions get a rejected MASKBITS bit, ``shallow``
    ones a z-band 5 sigma depth of 21 mag instead of 23.6."""
    ra = rng.uniform(box.ra_min, box.ra_max, n)
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(box.dec_min)),
                                           np.sin(np.radians(box.dec_max)), n)))
    r = {"RA": ra, "DEC": dec, "MASKBITS": np.where(masked(ra, dec), 2, 0).astype(np.int32),
         "EBV": np.zeros(n, np.float32)}
    for b, d in zip("GRIZ", DEPTH5):
        r[f"NOBS_{b}"] = np.full(n, 3, np.int16)
        r[f"GALDEPTH_{b}"] = np.full(n, depth_ivar_from_mag(d), np.float32)
    r["GALDEPTH_Z"] = np.where(shallow(ra, dec), depth_ivar_from_mag(21.0),
                               r["GALDEPTH_Z"]).astype(np.float32)
    return r


@pytest.fixture(scope="module")
def masked_region():
    """Mock clusters on a footprint with a masked corner and a shallow strip."""
    rng = np.random.default_rng(7)
    cfg = RemaConfig().replace(mask={"nside_fine": 512})
    rs = RSModel.from_template()
    ms, cosmo = MStar(cfg.model.mstar), CosmoTable.create()
    box = Box(10.0, 11.6, -0.8, 0.8)
    rnd = _randoms(rng, box, 400_000, masked=lambda ra, dec: (ra < 10.45) & (dec < -0.35),
                   shallow=lambda ra, dec: ra > 11.25)
    fp = build_footprint(rnd, cfg, box=box, density=400_000 / box.area_deg2())
    field = mocks.mock_field(rng, rs, box, density=6000, depth5=DEPTH5, mag_range=(12.0, 22.5))
    specs = [(10.45, -0.35, 0.3, 40.0), (11.3, 0.3, 0.45, 30.0), (10.9, 0.1, 0.25, 25.0)]
    cl = [mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, ra, dec, z, lam, DEPTH5, poisson=False)
          for ra, dec, z, lam in specs]
    gal = mocks.concat(field, *cl)
    n = gal["RA"].size
    gal["ID"] = np.arange(n, dtype=np.int64)
    gal["ZSPEC"] = np.full(n, -1.0, np.float32)
    reg = Region.build(gal, rs, cfg, footprint=fp, area_deg2=box.area_deg2())
    return reg, specs, fp


def test_aperture_matches_the_richness_solve(masked_region):
    """lambda, SCALEVAL and MASKFRAC of the finder at the cluster centres; the aperture, given
    lambda, gives back SCALEVAL and MASKFRAC (masked corner, shallow strip, clear area)."""
    reg, specs, fp = masked_region
    cfg = reg.cfg
    gi = np.array([int(np.argmin(np.hypot(reg.gal["RA"] - ra, reg.gal["DEC"] - dec)))
                   for ra, dec, _, _ in specs])
    z0 = np.array([z for _, _, z, _ in specs], np.float32)
    pc = cfg.richness.percolation
    st = Stage.make(pc.r0, pc.beta, cfg.richness.maxrad_factor)
    out = B._run_batched(reg, gi, z0, st, percolation_quad(cfg), "richness", log_every=0)
    assert np.all(out["LAMBDA"] > 3)
    ap = aperture_values(reg.gal["RA"][gi], reg.gal["DEC"][gi], z0, out["LAMBDA"],
                         footprint_map(fp, "z"), cfg)
    np.testing.assert_allclose(ap["R_LAMBDA"], out["R_LAMBDA"], rtol=1e-5)
    # The solve stops within a few 1e-3 of the root of sum pmem = lambda K, so the finder's
    # SCALEVAL = lambda / sum pmem equals 1/K to that precision (a complete aperture gets 0.998).
    np.testing.assert_allclose(ap["SCALEVAL"], out["SCALEVAL"], rtol=5e-3)
    np.testing.assert_allclose(ap["MASKFRAC"], out["MASKFRAC"], atol=1e-5)
    # The masked corner and the shallow strip are seen.
    assert out["MASKFRAC"][0] > 0.1 and out["SCALEVAL"][1] > 1.1
    assert out["MASKFRAC"][2] < 1e-3 and out["SCALEVAL"][2] == pytest.approx(1.0, abs=0.01)


def test_aperture_away_from_the_data(masked_region):
    """The same cluster placed elsewhere: complete in the clear area, masked in the corner, lossy
    in the shallow strip, absent outside the footprint; batching does not change the values."""
    _, _, fp = masked_region
    cfg = RemaConfig().replace(mask={"nside_fine": 512})
    fmap = footprint_map(fp, "z")
    ra = np.array([10.9, 10.1, 11.45, 20.0, 10.45])
    dec = np.array([0.4, -0.7, 0.0, 0.0, -0.35])
    z, lam = np.full(5, 0.4), np.full(5, 30.0)
    a = aperture_values(ra, dec, z, lam, fmap, cfg, batch=2)
    b = aperture_values(ra, dec, z, lam, fmap, cfg, batch=8, model=aperture_model(cfg))
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-6)
    assert a["MASKFRAC"][0] < 1e-3 and a["SCALEVAL"][0] == pytest.approx(1.0, abs=0.01)
    assert a["MASKFRAC"][1] == pytest.approx(1.0, abs=1e-3)
    assert a["MASKFRAC"][2] < 1e-3 and 1.02 < a["SCALEVAL"][2] < 1.2
    # The shallow strip (21 mag at 5 sigma) loses more of a cluster at higher redshift.
    hz = aperture_values(ra[2:3], dec[2:3], [0.6], [30.0], fmap, cfg)
    assert hz["SCALEVAL"][0] > 1.5 * a["SCALEVAL"][2]
    assert a["MASKFRAC"][3] == 1.0 and np.isinf(a["SCALEVAL"][3])
    assert 0.15 < a["MASKFRAC"][4] < 0.3                   # on the corner of the masked quadrant
    # A larger richness has a larger aperture: more of it falls on the mask edge.
    c = aperture_values(ra[4:], dec[4:], z[4:], [120.0], fmap, cfg)
    assert c["R_LAMBDA"][0] > a["R_LAMBDA"][4]


def test_aperture_model_and_quad():
    cfg = RemaConfig()
    q = percolation_quad(cfg)
    ref = RadialQuad.make(rmax=1.0 * 20.0**0.2 + 5 * cfg.model.rsig)
    np.testing.assert_array_equal(np.asarray(q.r), np.asarray(ref.r))
    # The photo-z filter's aperture uses the same profile and m*(z).
    m = aperture_model(cfg.replace(model={"filter": "photoz"}))
    assert m.filter == "redsequence" and m.rs is None and m.bkg is None
    full = SparseMap(64, np.arange(12 * 64**2), {"FRACGOOD": np.ones(12 * 64**2, np.float32),
                                                  "SIGF_Z": np.full(12 * 64**2, 1e-3, np.float32)})
    out = aperture_values([1.0, 2.0], [0.0, 0.0], [0.2, 0.6], [10.0, 100.0],
                          full.device(("FRACGOOD", "SIGF_Z")), cfg)
    np.testing.assert_allclose(out["R_LAMBDA"], [0.1**0.2, 1.0], rtol=1e-6)
    np.testing.assert_allclose(out["SCALEVAL"], 1.0, atol=1e-4)


def test_randoms_config():
    c = RandomsConfig()
    assert c.detection_cuts(RemaConfig().replace(richness={"minlambda": 5.0})) == (5.0, 0.2)
    d = c.updated({"generate": {"lambda_det_min": 20, "max_maskfrac": 0.1}})
    assert d.detection_cuts(RemaConfig()) == (20.0, 0.1)
    with pytest.raises(KeyError):
        c.updated({"vlim": {"nope": 1}})


def test_randoms_config_yaml(tmp_path):
    c = RandomsConfig().updated({"vlim": {"nsigma": 10.0, "band": "r"}})
    c.to_yaml(tmp_path / "r.yaml")
    assert RandomsConfig.from_yaml(tmp_path / "r.yaml") == c


@pytest.mark.data
def test_strip_closure():
    """On the laptop strip run: SCALEVAL and MASKFRAC recomputed from the footprint at the seed
    position and the likelihood-pass redshift (the inputs of the finder) equal the catalogue's."""
    from rema.io.tables import read_table
    from rema.sky.maps import Footprint

    strip = DR11_DIR / "rema" / "strip"
    cat_path, fp_path = strip / "blind_strip_fp.fits", strip / "footprint_strip.fits"
    ck_path, gal_path = strip / "ck_strip_fp" / "likelihood.npz", strip / "galaxies_3sweeps_v2.fits"
    require(cat_path, fp_path, ck_path, gal_path)
    from rema.cli import _catalog_cfg

    cfg = _catalog_cfg(cat_path)
    cat = read_table(cat_path, hdu="CLUSTERS",
                     columns=["SEED_ID", "RA_SEED", "DEC_SEED", "LAMBDA", "SCALEVAL", "MASKFRAC",
                              "Z_LAMBDA_RAW"])
    ck = np.load(ck_path)
    ids = read_table(gal_path, columns=["ID"])["ID"]           # the run read the whole table
    zc = dict(zip(ids[ck["GI"]], ck["Z_LAMBDA"]))
    z0 = np.array([zc[s] for s in cat["SEED_ID"]])
    fp = Footprint.read(fp_path)
    ap = aperture_values(cat["RA_SEED"], cat["DEC_SEED"], z0, cat["LAMBDA"], footprint_map(fp, "z"), cfg)
    # SCALEVAL to the precision of the lambda solve (see above), MASKFRAC to float32.
    dev = np.abs(ap["SCALEVAL"] / cat["SCALEVAL"] - 1)
    assert np.quantile(dev, 0.999) < 5e-3 and dev.max() < 0.02
    # MASKFRAC to float32, but for 7 of 12888 clusters near mask edges (a few aperture points
    # change pixel; at most 0.01).
    dmf = np.abs(ap["MASKFRAC"] - cat["MASKFRAC"])
    assert np.quantile(dmf, 0.999) < 2e-4 and dmf.max() < 0.01
    # At Z_LAMBDA_RAW (what the randoms use) the values stay close.
    raw = aperture_values(cat["RA_SEED"], cat["DEC_SEED"], cat["Z_LAMBDA_RAW"], cat["LAMBDA"],
                          footprint_map(fp, "z"), cfg)
    assert np.median(np.abs(raw["SCALEVAL"] / cat["SCALEVAL"] - 1)) < 2e-3
