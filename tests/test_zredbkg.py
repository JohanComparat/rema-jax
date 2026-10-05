"""Zred background (redMaPPer ZREDBKG): build, lookup, FITS round trip."""

import jax.numpy as jnp
import numpy as np
import pytest

from rema.calibration import Calibration
from rema.model.background import ZredBkg, build_zred_bkg
from rema.model.redsequence import RSModel


def _uniform(rng, n=400_000):
    zred = rng.uniform(0.0, 1.1, n)
    refmag = rng.uniform(12.0, 24.5, n)
    chisq = rng.uniform(0.0, 120.0, n)
    return zred, chisq, refmag


def test_build_recovers_density(rng):
    zred, chisq, refmag = _uniform(rng)
    area = 50.0
    zb = build_zred_bkg(zred, chisq, refmag, lambda m: area, zred_range=(0.01, 1.0),
                        mag_min=12.0, mag_max=24.0, chisq_max=100.0)
    # Galaxies per deg^2 per mag per unit zred that pass chi^2 < 100, away from the table edges.
    expect = zred.size * (100.0 / 120.0) / (area * 1.1 * 12.5)
    inner = np.asarray(zb.sigma_g)[2:-2, 2:-2]
    assert np.median(inner) == pytest.approx(expect, rel=0.03)
    assert zb.sigma_g.shape == (99, 60)


def test_lookup_edges(rng):
    zred, chisq, refmag = _uniform(rng, 50_000)
    zb = build_zred_bkg(zred, chisq, refmag, lambda m: 10.0, zred_range=(0.05, 0.9))
    v = np.asarray(zb.lookup(jnp.asarray([0.3, 0.02, -1.0, 0.95, 0.3, 0.3]),
                             jnp.asarray([20.0, 20.0, 20.0, 20.0, 11.0, 30.0])))
    assert np.isfinite(v[0]) and v[0] > 0
    assert np.isinf(v[1]) and np.isinf(v[2])            # below the zred table, failed zred
    assert v[3] == pytest.approx(float(zb.lookup(0.895, 20.0)), rel=0.1)   # clamped above
    assert np.isinf(v[4])                               # brighter than the table
    assert np.isfinite(v[5])                            # fainter: clamped
    # Bilinear between bin centres.
    z0, z1 = float(zb.zred[10]), float(zb.zred[11])
    m0 = float(zb.mag[30])
    mid = float(zb.lookup(0.5 * (z0 + z1), m0))
    assert mid == pytest.approx(0.5 * float(zb.sigma_g[10, 30] + zb.sigma_g[11, 30]), rel=1e-4)


def test_count_floor_and_area():
    # Two galaxies only: every other cell is floored at 0.1 count.
    zb = build_zred_bkg(np.array([0.3, 0.31]), np.array([1.0, 1.0]), np.array([20.0, 20.1]),
                        lambda m: 2.0 if m < 23 else 0.0, zred_range=(0.05, 0.9), mag_max=24.0)
    sg = np.asarray(zb.sigma_g)
    floor = 0.1 / (2.0 * 0.2 * 0.01)
    assert np.min(sg[np.isfinite(sg)]) == pytest.approx(floor)
    assert np.all(np.isinf(sg[:, np.asarray(zb.mag) > 23]))   # no area: no background


def test_calibration_roundtrip_with_zbkg(tmp_path, rng):
    zred, chisq, refmag = _uniform(rng, 20_000)
    zb = build_zred_bkg(zred, chisq, refmag, lambda m: 5.0)
    wcen = {"delta0": -1.2, "SIGMA_M": 0.3, "LNW_CEN_MEAN": 0.2}
    cal = Calibration(rs=RSModel.from_template(), zbkg=zb, wcen=wcen)
    back = Calibration.read(cal.write(tmp_path / "cal.fits"))
    np.testing.assert_allclose(np.asarray(back.zbkg.sigma_g), np.asarray(zb.sigma_g), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(back.zbkg.zred), np.asarray(zb.zred))
    assert back.wcen == {"DELTA0": -1.2, "SIGMA_M": 0.3, "LNW_CEN_MEAN": 0.2}
    assert isinstance(back.zbkg, ZredBkg)
