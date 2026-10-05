"""Cosmology tables, splines, radial/luminosity/colour filters, HEALPix indexing."""

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import exp1
from scipy.stats import chi2

from rema.model import profiles as P
from rema.model.cosmo import CosmoTable
from rema.model.splines import NaturalSpline
from rema.sky.healpix import ang2pix_nest


# ----------------------------------------------------------------- cosmology
@pytest.fixture(scope="module")
def cosmo():
    return CosmoTable.create(Omega_m=0.3, h=0.7)


def test_cosmology_vs_astropy(cosmo):
    ref = FlatLambdaCDM(H0=100.0, Om0=0.3, Tcmb0=0.0)   # Mpc/h, no radiation
    z = np.linspace(0.01, 1.5, 60)
    np.testing.assert_allclose(np.asarray(cosmo.da(z)), ref.angular_diameter_distance(z).value, rtol=2e-4)
    np.testing.assert_allclose(np.asarray(cosmo.ez(z)), ref.efunc(z), rtol=3e-4)
    np.testing.assert_allclose(np.asarray(cosmo.dvdz(z)),
                               ref.differential_comoving_volume(z).value, rtol=5e-4)


def test_volume_factor(cosmo):
    assert float(cosmo.volume_factor(0.9, 0.9)) == pytest.approx(1.0)
    assert float(cosmo.volume_factor(0.3, 0.9)) < 1.0


def test_cosmo_is_a_pytree(cosmo):
    f = jax.jit(lambda c, z: c.mpc_per_deg(z))
    assert float(f(cosmo, 0.3)) == pytest.approx(float(cosmo.da(0.3)) * np.pi / 180, rel=1e-6)


# ----------------------------------------------------------------- splines
def test_spline_matches_scipy_natural():
    x = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9])
    y = np.sin(3 * x) + x**2
    z = np.linspace(0.05, 0.9, 200)
    ref = CubicSpline(x, y, bc_type="natural")(z)
    sp = NaturalSpline(x)
    np.testing.assert_allclose(sp(z, y), ref, atol=1e-12)
    np.testing.assert_allclose(np.asarray(sp.eval_jax(jnp.asarray(z), jnp.asarray(y))), ref, atol=2e-6)


def test_spline_linear_extrapolation_and_vector_values():
    x = np.array([0.1, 0.3, 0.6])
    y = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 0.0]])
    sp = NaturalSpline(x)
    z = np.array([-0.2, 0.0, 0.7, 1.0])
    out = np.asarray(sp.eval_jax(jnp.asarray(z), jnp.asarray(y)))
    host = np.stack([sp(z, y[:, 0]), sp(z, y[:, 1])], axis=-1)
    np.testing.assert_allclose(out, host, atol=1e-5)
    # linear beyond the ends: equal steps give equal increments
    lo = sp(np.array([-0.2, -0.1, 0.0]), y[:, 0])
    assert lo[1] - lo[0] == pytest.approx(lo[2] - lo[1])


def test_spline_gradient_in_z():
    x = np.linspace(0.0, 1.0, 6)
    y = jnp.asarray(x**3)
    sp = NaturalSpline(x)
    g = jax.grad(lambda z: sp.eval_jax(z, y))(0.43)
    h = 1e-3
    fd = (sp(0.43 + h, np.asarray(y)) - sp(0.43 - h, np.asarray(y))) / (2 * h)
    assert float(g) == pytest.approx(float(fd), rel=1e-3)


# ----------------------------------------------------------------- NFW filter
def _F_exact(x):
    if x < 1:
        return (1 - np.arccosh(1 / x) / np.sqrt(1 - x * x)) / (x * x - 1)
    if x > 1:
        return (1 - np.arccos(1 / x) / np.sqrt(x * x - 1)) / (x * x - 1)
    return 1.0 / 3.0


@pytest.mark.parametrize("x", [0.3, 0.7, 0.9, 0.97, 0.999, 1.0, 1.001, 1.03, 1.2, 3.0, 20.0])
def test_nfw_F(x):
    with jax.enable_x64(True):
        got = float(P.nfw_F(jnp.float64(x)))
    want = _F_exact(x) if abs(x - 1) > 1e-2 else _F_exact(x) if abs(x - 1) > 1e-6 else 1 / 3
    assert got == pytest.approx(want, rel=1e-6 if abs(x - 1) > 1e-2 else 1e-4)


def test_nfw_enclosed_matches_quadrature():
    for r in (0.05, 0.1, 0.15, 0.5, 1.0, 2.0):
        num = quad(lambda s: 2 * np.pi * s * float(P.nfw_sigma(s)), 0, r, points=[0.1, 0.15], limit=200)[0]
        assert float(P.nfw_enclosed(r)) == pytest.approx(num, rel=2e-5)


def test_nfw_gradients_finite_near_one():
    g = jax.grad(lambda r: P.nfw_sigma(r))(0.15)
    assert np.isfinite(float(g))
    g2 = jax.grad(lambda r: P.nfw_norm(r))(1.0)
    assert np.isfinite(float(g2)) and float(g2) < 0


# ----------------------------------------------------------------- luminosity and colour
def test_lumnorm_alpha_minus_one_is_exp1():
    for mstar, maxmag in ((18.0, 19.75), (20.0, 21.0), (16.0, 18.5)):
        x = 10 ** (0.4 * (mstar - maxmag))
        assert float(P.lumnorm(mstar, maxmag, -1.0)) == pytest.approx(2.5 / np.log(10) * exp1(x), rel=1e-5)


def test_lumnorm_general_alpha():
    mstar, maxmag, alpha = 19.0, 20.75, -0.8
    num = quad(lambda m: float(P.schechter(m, mstar, alpha)), 5.0, maxmag)[0]
    assert float(P.lumnorm(mstar, maxmag, alpha)) == pytest.approx(num, rel=1e-5)


def test_chisq_pdf():
    x = np.array([0.1, 1.0, 3.0, 10.0])
    for k in (1, 2, 3):
        np.testing.assert_allclose(np.asarray(P.chisq_pdf(x, k)), chi2.pdf(x, k), rtol=1e-5)


def test_mstar_tables():
    a, b = P.MStar("des_z03"), P.MStar("legacy_z_ezgal")
    assert float(a(0.1)) == pytest.approx(15.85, abs=0.02)
    assert abs(float(a(0.9)) - float(b(0.9))) < 0.1


def test_lsst_and_euclid_tables():
    """The LSST and Euclid tables load like the DECam ones and seed a red-sequence model."""
    from importlib import resources
    from pathlib import Path

    from rema.io.tables import read_table
    from rema.model.redsequence import RSModel

    for name in ("lsst_i03", "lsst_r03", "lsst_z03", *[f"lsst_{b}_ezgal" for b in "ugrizy"],
                 *[f"euclid_{b}_ezgal" for b in ("vis", "y", "j", "h")]):
        ms = P.MStar(name)
        assert ms.zmin == pytest.approx(0.01)
        assert ms.zmax == pytest.approx(1.51 if name.endswith("03") else 2.5)
    # Empirical (redMaPPer) and model m* agree in LSST z at low redshift.
    assert abs(float(P.MStar("lsst_z03")(0.5)) - float(P.MStar("lsst_z_ezgal")(0.5))) < 0.05
    # A red population is brighter in the redder bands.
    m = [float(P.MStar(f"euclid_{b}_ezgal")(0.5)) for b in ("vis", "y", "j", "h")]
    assert m == sorted(m, reverse=True)
    for bands, ref, template, tbands, mstar in [
            (tuple("grizy"), "z", "bc03_lsst_ugrizy", tuple("ugrizy"), "lsst_z03"),
            (("vis", "y", "j", "h"), "h", "bc03_euclid_visyjh", ("vis", "y", "j", "h"), "euclid_h_ezgal")]:
        rs = RSModel.from_template(bands=bands, ref_band=ref, template=template, template_bands=tbands,
                                   mstar=mstar)
        t = read_table(Path(str(resources.files("rema.data"))) / f"colors_{template}.fits")
        j0 = tbands.index(bands[0])
        want = [np.interp(0.5, t["Z"], t["COLOR"][:, j0 + j]) for j in range(len(bands) - 1)]
        np.testing.assert_allclose(np.asarray(rs.at(0.5).mean), want, atol=1e-6)


# ----------------------------------------------------------------- HEALPix
@pytest.mark.parametrize("nside", [1, 2, 64, 1024, 8192])
def test_ang2pix_nest_matches_healpy(nside, rng):
    n = 200_000
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    want = hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)
    with jax.enable_x64(True):
        got = np.asarray(ang2pix_nest(nside, jnp.asarray(ra), jnp.asarray(dec)))
    assert np.array_equal(got, want)
    # float32 positions resolve ~0.1 arcsec: points that close to a pixel edge may flip.
    got32 = np.asarray(ang2pix_nest(nside, ra.astype(np.float32), dec.astype(np.float32)))
    assert np.mean(got32 == want) > (0.999 if nside <= 1024 else 0.995)
