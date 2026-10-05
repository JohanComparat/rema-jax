"""The JAX build computes richness reproducibly and correctly.

The conda-forge jaxlib 0.10.2 CPU build returns a different lambda at every call of the same
compiled richness on the same inputs (and wrong values single-threaded); the PyPI wheels of the
same version do not. This fast check catches such a build before the slow tests do.
"""

import jax.numpy as jnp
import numpy as np

from rema.config import RemaConfig
from rema.core import richness as R
from rema.core.context import FilterModel
from rema.model.background import build_chisq_bkg
from rema.model.cosmo import CosmoTable
from rema.model.profiles import MStar
from rema.model.redsequence import RSModel
from rema.sky.neighbors import NeighborIndex
from rema.sky.regions import Box
from rema.validate import mocks


def test_richness_is_reproducible_and_recovers_a_mock_cluster():
    cfg, rs = RemaConfig(), RSModel.from_template()
    cosmo, ms = CosmoTable.create(), MStar("des_z03")
    rng = np.random.default_rng(11)
    depth5 = np.array([24.9, 24.7, 24.2, 23.6])
    box = Box(0.0, 2.0, -1.0, 1.0)
    field = mocks.mock_field(rng, rs, box, density=30000, depth5=depth5)
    specs = [(0.5, -0.5, 0.4, 40.0), (1.5, 0.5, 0.4, 40.0)]
    gal = mocks.concat(field, *[mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, ra, dec, z, lam,
                                                   depth5, poisson=False)
                                for ra, dec, z, lam in specs])
    bkg = build_chisq_bkg(gal["FLUX"], gal["FLUX_IVAR"], gal["REFMAG"], rs, ms, lambda m: box.area_deg2(),
                          zrange=(0.05, 0.9), iref=rs.iref, mag_max=24.0)
    model = FilterModel.create(rs, bkg, cfg, cosmo=cosmo, mstar=ms)
    stage, quad = R.Stage.make(1.0, 0.2), R.RadialQuad.make()
    S = np.array(specs)
    pad = NeighborIndex(gal["RA"], gal["DEC"]).query(
        S[:, 0], S[:, 1], float(stage.maxrad) / float(cosmo.mpc_per_deg(0.4)))
    take = lambda a: jnp.asarray(a[pad.idx])
    shape = pad.idx.shape
    nb = R.Neighbors(theta=jnp.asarray(pad.theta, jnp.float32), refmag=take(gal["REFMAG"]),
                     refmag_err=take(gal["REFMAG_ERR"]), flux=take(gal["FLUX"]), ivar=take(gal["FLUX_IVAR"]),
                     zred=jnp.zeros(shape), zred_e=jnp.ones(shape), pfree=jnp.ones(shape),
                     valid=jnp.asarray(pad.valid), is_center=jnp.zeros(shape, bool))
    ones = jnp.ones((len(S), quad.r.shape[0]))
    z = jnp.full(len(S), 0.4, jnp.float32)
    lams = [np.asarray(R.richness(nb, z, ones, ones, quad, model, stage).lam) for _ in range(3)]
    np.testing.assert_array_equal(lams[1], lams[0])
    np.testing.assert_array_equal(lams[2], lams[0])
    assert np.all(np.abs(lams[0] / 40.0 - 1.0) < 0.25), lams[0]
