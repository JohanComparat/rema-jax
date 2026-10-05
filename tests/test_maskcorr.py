"""Aperture completeness from the footprint map: selection completeness at the local depth and
the geometric fraction of an aperture cut by a mask."""

import jax.numpy as jnp
import numpy as np
import pytest

from rema.config import RemaConfig
from rema.core.context import FilterModel
from rema.core.maskcorr import no_footprint, radial_completeness, selection_completeness
from rema.core.richness import RadialQuad
from rema.model.redsequence import RSModel
from rema.sky.maps import SparseMap, build_footprint, depth_ivar_from_mag
from rema.sky.regions import Box


def _S(sigf, mstar=19.0, maxmag=20.75, snr_min=5.0, mag_max=30.0):
    sigf = jnp.asarray(sigf, jnp.float32)
    shape = sigf.shape
    return np.asarray(selection_completeness(sigf, jnp.full(shape, mstar), jnp.full(shape, maxmag),
                                             -1.0, snr_min, mag_max))


def test_selection_completeness_limits():
    # Flux error of a galaxy at the 5 sigma limit m5: 10^(-0.4 (m5 - 22.5)) / 5.
    sig = lambda m5: 10 ** (-0.4 * (m5 - 22.5)) / 5.0
    s = _S([sig(28.0), sig(22.0), sig(20.75), sig(18.0), sig(14.0)])
    assert s[0] == pytest.approx(1.0, abs=1e-3)           # deep: everything selected
    assert np.all(np.diff(s) <= 0)                        # shallower: fewer selected
    assert s[1] == pytest.approx(1.0, abs=1e-3)           # 1.25 mag beyond the faint end
    assert 0.5 < s[2] < 0.99                              # limit at the aperture's faint end
    assert s[3] < 0.1 and s[4] < 1e-3                     # far too shallow
    # No depth information (sigma <= 0): nothing selected.
    np.testing.assert_array_equal(_S([0.0, -1.0]), [0.0, 0.0])
    # A magnitude limit brighter than the faint end removes galaxies even at infinite depth.
    assert _S([sig(28.0)], mag_max=20.0)[0] < 0.9 * _S([sig(28.0)])[0]


def _randoms(rng, box, n, masked):
    """Uniform randoms in ``box``; ``masked(ra, dec)`` get a rejected MASKBITS bit."""
    ra = rng.uniform(box.ra_min, box.ra_max, n)
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(box.dec_min)),
                                           np.sin(np.radians(box.dec_max)), n)))
    r = {"RA": ra, "DEC": dec, "MASKBITS": np.where(masked(ra, dec), 2, 0).astype(np.int32),
         "EBV": np.zeros(n, np.float32)}
    for b, d in zip("GRIZ", (24.9, 24.7, 24.2, 23.6)):
        r[f"NOBS_{b}"] = np.full(n, 3, np.int16)
        r[f"GALDEPTH_{b}"] = np.full(n, depth_ivar_from_mag(d), np.float32)
    return r


@pytest.fixture(scope="module")
def half_masked():
    """Footprint of a 4 x 4 deg box whose Dec < 0 half is masked, and a filter model."""
    rng = np.random.default_rng(41)
    cfg = RemaConfig().replace(mask={"nside_fine": 256})
    box = Box(10.0, 14.0, -2.0, 2.0)
    fp = build_footprint(_randoms(rng, box, 300_000, lambda ra, dec: dec < 0), cfg, box=box)
    model = FilterModel.create(RSModel.from_template(), None, cfg)
    return fp, model


def test_radial_completeness_in_a_half_masked_field(half_masked):
    fp, model = half_masked
    quad = RadialQuad.make(rmax=1.5)
    fmap = fp.fine.device(("FRACGOOD", "SIGF_Z"))
    ra = jnp.asarray([12.0, 12.0, 12.0, 30.0], jnp.float32)
    dec = jnp.asarray([1.0, -1.0, 0.0, 0.0], jnp.float32)
    z = jnp.asarray([0.3, 0.3, 0.3, 0.3], jnp.float32)
    frad, fgeo = (np.asarray(a) for a in radial_completeness(ra, dec, z, fmap, quad, model))
    assert frad.shape == fgeo.shape == (4, quad.r.shape[0])
    # Clear side: fully observed; masked side and outside the map: nothing.
    np.testing.assert_allclose(fgeo[0], 1.0, atol=0.02)
    np.testing.assert_allclose(fgeo[1], 0.0, atol=0.02)
    np.testing.assert_array_equal(fgeo[3], 0.0)
    # On the mask edge, half of every annulus is observed.
    np.testing.assert_allclose(fgeo[2], 0.5, atol=0.08)
    # Depth losses only lower the radial completeness; at z = 0.3 a 23.6 mag depth is complete.
    assert np.all(frad <= fgeo + 1e-6)
    np.testing.assert_allclose(frad[0], fgeo[0], atol=0.02)
    # A shallow map (5 sigma limit 18.5 mag, m* + 0.1 at z = 0.3) loses most galaxies.
    shallow = SparseMap(fp.nside, fp.fine.pixels,
                        {"FRACGOOD": fp.fine.values["FRACGOOD"],
                         "SIGF_Z": np.full(fp.fine.pixels.size, 10 ** (-0.4 * (18.5 - 22.5)) / 5,
                                           np.float32)})
    frad_s, fgeo_s = radial_completeness(ra[:1], dec[:1], z[:1], shallow.device(("FRACGOOD", "SIGF_Z")),
                                         quad, model)
    np.testing.assert_allclose(np.asarray(fgeo_s), fgeo[:1], atol=1e-6)
    assert np.all(np.asarray(frad_s) < 0.5 * fgeo[0])


def test_no_footprint():
    quad = RadialQuad.make()
    frad, fgeo = no_footprint(3, quad)
    assert frad.shape == (3, quad.r.shape[0])
    np.testing.assert_array_equal(np.asarray(frad), 1.0)
    np.testing.assert_array_equal(np.asarray(fgeo), 1.0)
