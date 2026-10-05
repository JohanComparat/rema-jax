"""Red-sequence calibration driver on mock data: spectroscopic seeds, the diagnostic plots and
an end-to-end ``rema calibrate`` run."""

from types import SimpleNamespace

import numpy as np
import pytest

from rema import cli
from rema.calib.driver import spec_seeds
from rema.calib.plots import plot_redsequence, plot_zlambda
from rema.calibration import Calibration
from rema.config import RemaConfig
from rema.io.legacy import survey_hash
from rema.io.tables import write_table
from rema.model.cosmo import CosmoTable
from rema.model.profiles import MStar
from rema.model.redsequence import RSModel
from rema.sky.regions import Box
from rema.validate import mocks

pytest.importorskip("matplotlib")

BOX = Box(10.0, 11.2, -0.6, 0.6)
DEPTH5 = np.array([24.9, 24.7, 24.2, 23.6])
CLUSTERS = [(10.3, -0.3, 0.15), (10.9, -0.3, 0.25), (10.3, 0.3, 0.35), (10.9, 0.3, 0.45),
            (10.6, 0.0, 0.55), (10.6, -0.45, 0.3)]


def mock_spec_field(seed=61, clusters=CLUSTERS, nspec=25, density=4000):
    """Field galaxies and lambda = 40 clusters whose ``nspec`` brightest members have ZSPEC;
    also returns the row ranges of the clusters."""
    rng = np.random.default_rng(seed)
    cfg = RemaConfig()
    rs = RSModel.from_template()
    ms, cosmo = MStar(cfg.model.mstar), CosmoTable.create()
    field = mocks.mock_field(rng, rs, BOX, density=density, depth5=DEPTH5, mag_range=(14.0, 22.0))
    parts, zsp, rows, start = [field], [np.full(field["RA"].size, -1.0)], [], field["RA"].size
    for ra, dec, z in clusters:
        cl = mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, ra, dec, z, 40.0, DEPTH5,
                                poisson=False, central_dmag=-1.5)
        zz = np.full(cl["RA"].size, -1.0)
        o = np.argsort(cl["REFMAG"])[:nspec]
        zz[o] = z + rng.normal(0.0, 0.002, o.size)
        parts.append(cl)
        zsp.append(zz)
        rows.append(np.arange(start, start + cl["RA"].size))
        start += cl["RA"].size
    gal = mocks.concat(*parts)
    gal["ID"] = np.arange(gal["RA"].size, dtype=np.int64)
    gal["ZSPEC"] = np.concatenate(zsp).astype(np.float32)
    return gal, rows, rs


def test_spec_seeds():
    gal, rows, rs = mock_spec_field(clusters=CLUSTERS[:2], nspec=10, density=1500)
    # Spectroscopic redshifts for some blue field galaxies (off the red sequence) and one
    # cluster member at a wrong redshift.
    rng = np.random.default_rng(3)
    nf = rows[0][0]
    blue = rng.choice(nf, 30, replace=False)
    gal["ZSPEC"][blue] = 0.2
    wrong = rows[0][np.argsort(gal["REFMAG"][rows[0]])[0]]
    gal["ZSPEC"][wrong] = 0.7
    cfg = RemaConfig()
    # spec_seeds only needs the table, the model and m*(z) of a Region.
    ms = MStar(cfg.model.mstar)
    reg = SimpleNamespace(gal=gal, rs=rs, band_idx=[0, 1, 2, 3],
                          mstar_np=lambda z: np.asarray(ms(np.asarray(z)), np.float64))
    seeds = spec_seeds(reg, cfg)
    members = np.concatenate([r[gal["ZSPEC"][r] > 0] for r in rows])
    members = members[members != wrong]
    # Red members are seeds; the mis-assigned member is not; few blue galaxies pass chi^2.
    assert np.isin(members, seeds).mean() > 0.9
    assert wrong not in seeds
    assert np.isin(blue, seeds).mean() < 0.3
    assert np.all(gal["ZSPEC"][seeds] > 0)
    # Brighter limit: fewer seeds; none without spectroscopy.
    assert spec_seeds(reg, cfg, dmag=-1.0).size < seeds.size
    gal2 = dict(reg.gal, ZSPEC=np.full(gal["ID"].size, -1.0, np.float32))
    reg.gal = gal2
    assert spec_seeds(reg, cfg).size == 0


def test_plots(tmp_path):
    rng = np.random.default_rng(5)
    rs = RSModel.from_template()
    n = 400
    z = rng.uniform(0.1, 0.7, n)
    refmag = np.asarray(MStar("des_z03")(z)) + rng.uniform(-1, 1, n)
    flux, ivar = mocks.noisy_fluxes(rng, mocks.red_sequence_mags(rng, rs, z, refmag), DEPTH5)
    p = plot_redsequence(rs, flux, ivar, z, rng.uniform(0, 1, n), refmag, tmp_path / "rs.png")
    assert p.exists() and p.read_bytes()[:4] == b"\x89PNG"
    zs = rng.uniform(0.1, 0.7, 50)
    zl = zs + rng.normal(0, 0.01, zs.size)
    zl[:3] = -1.0                                        # failed z_lambda are left out
    q = plot_zlambda(zs, zl, np.full(zs.size, 0.01), tmp_path / "zl.png")
    assert q.exists() and q.stat().st_size > 1000


@pytest.mark.slow
def test_cli_calibrate(tmp_path, monkeypatch):
    """`rema calibrate` on six spectroscopically seeded mock clusters: one EM iteration, the
    z_lambda correction, a two-step wcen fit and the plots."""
    monkeypatch.setattr(cli, "_setup_jax", lambda: None)
    gal, rows, _ = mock_spec_field()
    cfg = RemaConfig().replace(calib={"niter": 1, "wcen_niter": 2})
    write_table(tmp_path / "gal.fits", gal, header={"SURVHASH": survey_hash(cfg)}, extname="GALAXIES")
    cfg.to_yaml(tmp_path / "cfg.yaml")
    (tmp_path / "x_pars.fit").touch()
    assert cli.main(["calibrate", "--galaxies", str(tmp_path / "gal.fits"), "--config",
                     str(tmp_path / "cfg.yaml"), "--box", *map(str, BOX.as_tuple()),
                     "--init-pars", str(tmp_path / "x_pars.fit"), "--plots", str(tmp_path / "plots"),
                     "--out", str(tmp_path / "cal.fits")]) == 0
    cal = Calibration.read(tmp_path / "cal.fits")
    assert cal.config == cfg and cal.bkg is not None and cal.zbkg is not None
    assert cal.zredcorr is not None and cal.zlcorr is not None
    assert cal.meta["NCLUSTER"] == len(CLUSTERS) and cal.meta["NSPECGAL"] == 25 * len(CLUSTERS)
    assert cal.meta["ZLNMAD"] < 0.02 and cal.meta["NWCEN"] >= 3
    assert (cal.meta["BOXRA0"], cal.meta["BOXDEC1"]) == (BOX.ra_min, BOX.dec_max)
    # The calibrated red sequence still describes the mock clusters' colours.
    import jax.numpy as jnp

    zz = jnp.asarray([0.2, 0.35, 0.5])
    np.testing.assert_allclose(np.asarray(cal.rs.at(zz).mean),
                               np.asarray(RSModel.from_template().at(zz).mean), atol=0.08)
    assert set(cal.wcen) >= {"DELTA0", "SIGMA_M", "LNW_FG_MEAN", "LNW_SAT_MEAN", "LNW_CEN_MEAN"}
    assert {p.name for p in (tmp_path / "plots").iterdir()} == {"redsequence.png", "zlambda.png"}
