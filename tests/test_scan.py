"""Scan mode: redshift grid, ZMAX and the optical centre on mock clusters."""

import numpy as np
import pytest

from rema.config import RemaConfig
from rema.model.cosmo import CosmoTable
from rema.model.profiles import MStar
from rema.model.redsequence import RSModel
from rema.modes.common import Region
from rema.modes.scan import run_scan
from rema.sky.regions import Box
from rema.validate import mocks


@pytest.mark.slow
def test_scan_recovers_mock_clusters():
    rng = np.random.default_rng(2)
    cfg = RemaConfig()
    rs = RSModel.from_template()
    ms, cosmo = MStar(cfg.model.mstar), CosmoTable.create()
    depth5 = np.array([24.9, 24.7, 24.2, 23.6])
    box = Box(10.0, 11.5, -0.75, 0.75)
    field = mocks.mock_field(rng, rs, box, density=12000, depth5=depth5, mag_range=(12.0, 22.5))
    specs = np.array([(10.5, -0.3, 0.30, 40.0), (11.0, 0.3, 0.55, 30.0)])
    cl = [mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, ra, dec, z, lam, depth5, poisson=False,
                             central_dmag=-1.5) for ra, dec, z, lam in specs]
    gal = mocks.concat(field, *cl)
    n = gal["RA"].size
    gal["ID"] = np.arange(n, dtype=np.int64)
    gal["ZSPEC"] = np.full(n, -1.0, np.float32)
    reg = Region.build(gal, rs, cfg, area_deg2=box.area_deg2())
    cat, mem = run_scan(reg, specs[:, 0], specs[:, 1])

    # lambda(z) is computed at every step of the grid, and ZMAX, LMAX are read from it.
    sc = cfg.scan
    nz = int(round((sc.zrange[1] - sc.zrange[0]) / sc.zstep)) + 1
    assert cat["Z_STEPS"].shape[1] == cat["LAMBDA_STEPS"].shape[1] == cat["LIKELIHOOD_STEPS"].shape[1] == nz
    i = np.arange(len(specs))
    np.testing.assert_array_equal(cat["ZMAX"], cat["Z_STEPS"][i, cat["MAX_IND"]])
    np.testing.assert_array_equal(cat["LMAX"], cat["LAMBDA_STEPS"][i, cat["MAX_IND"]])
    assert np.all(np.abs(cat["ZMAX"] - specs[:, 2]) < 0.015)
    assert not np.any(cat["ZMAX_EDGE"])

    # Refined redshift and richness at the optical centre, which is the injected central.
    assert np.all(np.abs(cat["Z_LAMBDA_OPT"] - specs[:, 2]) < 0.015)
    assert np.all((cat["LAMBDA_OPT"] > 0.7 * specs[:, 3]) & (cat["LAMBDA_OPT"] < 1.4 * specs[:, 3]))
    np.testing.assert_allclose(cat["RA_OPT"], specs[:, 0], atol=1e-6)
    np.testing.assert_allclose(cat["DEC_OPT"], specs[:, 1], atol=1e-6)
    assert set(np.unique(mem["MEM_MATCH_ID"])) <= set(cat["MEM_MATCH_ID"])
