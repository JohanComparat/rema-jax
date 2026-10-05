"""Photometry of passively evolving stellar populations: the m*(z) and colour tables of
:mod:`rema.data` (built by :mod:`rema.data.build`).

A simple stellar population (SSP) grid gives F_nu at 10 pc on a frequency grid and an age grid.
A composite population with star formation rate psi(t) = exp(-t/tau) has, at age T,

    F_csp(T) = int_0^T psi(T - t) F_ssp(t) dt,

with F_ssp linear in t between the grid ages (EzGal's interpolation). On every grid interval the
integral of exp(-(T - t)/tau) (a + b t) is closed form, so F_csp(T) = sum_j w_j(T) F_ssp(t_j)
exactly, with weights from :func:`exponential_weights`. The normalisation (here int_0^inf psi =
tau) is a constant, absorbed when magnitudes are normalised.

A galaxy formed at z_f has age T(z) = t_L(z_f) - t_L(z) at redshift z (t_L: lookback time). Its
observed AB magnitude through a bandpass R(nu) is

    m = -2.5 log10[(1 + z) int F_csp(nu (1 + z)) R(nu) dnu/nu / (3631 Jy int R(nu) dnu/nu)] + mu(z),

with F_nu at 10 pc and mu(z) the distance modulus, as in EzGal (Mancone & Gonzalez 2012) and
BC03's ``cm_evolution``. Integrals use the trapezoid rule on the bandpass grid.

>>> import numpy as np
>>> t = np.linspace(0.0, 2.0, 201)
>>> w = exponential_weights(1.0, t, 0.1)           # F = 1 at every age: int_0^1 e^-(1-t)/0.1 dt
>>> bool(np.isclose(float(w.sum()), 0.1 * (1 - np.exp(-10.0))))
True
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

C_ANGSTROM = 2.99792458e18     # speed of light [Angstrom/s]
AB_FNU = 3.631e-20             # AB zero point [erg/s/cm^2/Hz]
HUBBLE_TIME_GYR = 977.7922216807891   # 1/H0 [Gyr] for H0 = 1 km/s/Mpc
_LENGTH_TO_ANGSTROM = {"angstrom": 1.0, "a": 1.0, "nm": 10.0, "micron": 1e4, "um": 1e4}


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SSPGrid:
    """Simple stellar population spectra.

    Attributes
    ----------
    nu : frequencies [Hz], increasing.
    age : ages [Gyr], increasing, ``age[0] = 0``.
    fnu : F_nu at 10 pc [erg/s/cm^2/Hz], shape [len(nu), len(age)].
    """

    nu: jnp.ndarray
    age: jnp.ndarray
    fnu: jnp.ndarray

    @classmethod
    def from_ezgal(cls, path) -> "SSPGrid":
        """Read an EzGal model file (F_nu on HDU 0, ``vs`` [Hz] on HDU 1, ``ages`` [yr] on HDU 2)."""
        from astropy.io import fits

        with fits.open(path) as h:
            fnu = np.asarray(h[0].data, np.float64)
            nu = np.asarray(h[1].data["vs"], np.float64)
            age = np.asarray(h[2].data["ages"], np.float64) / 1e9
        o = np.argsort(nu)
        return cls(nu=jnp.asarray(nu[o]), age=jnp.asarray(age), fnu=jnp.asarray(fnu[o]))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Bandpass:
    """Filter response R(nu) on increasing frequencies [Hz]."""

    nu: jnp.ndarray
    response: jnp.ndarray
    name: str = field(default="", metadata=dict(static=True))

    @classmethod
    def from_wavelength(cls, wavelength, response, unit: str = "angstrom", name: str = "") -> "Bandpass":
        lam = np.asarray(wavelength, np.float64) * _LENGTH_TO_ANGSTROM[unit.lower()]
        nu = C_ANGSTROM / lam
        o = np.argsort(nu)
        return cls(nu=jnp.asarray(nu[o]), response=jnp.asarray(np.asarray(response, np.float64)[o]),
                   name=name)

    @classmethod
    def from_ecsv(cls, path, name: str = "") -> "Bandpass":
        """speclite filter file (columns ``wavelength``, with its unit, and ``response``)."""
        from astropy.table import Table

        t = Table.read(str(path), format="ascii.ecsv")
        unit = str(t["wavelength"].unit or "angstrom").lower()
        unit = "angstrom" if unit in ("aa", "angstrom") else unit
        return cls.from_wavelength(t["wavelength"], t["response"], unit, name)

    @classmethod
    def from_ascii(cls, path, unit: str = "angstrom", name: str = "") -> "Bandpass":
        """Two columns, wavelength and response (EzGal filter files)."""
        a = np.loadtxt(str(path), usecols=(0, 1))
        return cls.from_wavelength(a[:, 0], a[:, 1], unit, name)


def exponential_weights(T, age, tau):
    """Weights w [len(age)] with int_0^T exp(-(T - t)/tau) F(t) dt = sum_j w_j F(age_j).

    F is linear between the ages; ages above T do not contribute. T, age and tau in the same
    units.
    """
    T = jnp.asarray(T)
    t0, t1 = age[:-1], age[1:]
    a, b = jnp.clip(t0, 0.0, T), jnp.clip(t1, 0.0, T)
    ea, eb = jnp.exp((a - T) / tau), jnp.exp((b - T) / tau)
    i0 = tau * (eb - ea)                                         # int e^{(t-T)/tau} dt
    i1 = tau * (eb * (b - t0 - tau) - ea * (a - t0 - tau))       # int e^{(t-T)/tau} (t - t0) dt
    lin = i1 / (t1 - t0)
    w = jnp.zeros_like(age)
    return w.at[:-1].add(i0 - lin).at[1:].add(lin)


def csp_fnu(ssp: SSPGrid, T, tau):
    """F_nu at 10 pc [len(nu), len(T)] of the exponential population at ages T [Gyr]."""
    w = jax.vmap(exponential_weights, in_axes=(0, None, None))(jnp.atleast_1d(T), ssp.age, tau)
    return ssp.fnu @ w.T


def ab_mag_10pc(nu, fnu, z, band: Bandpass):
    """Observed-frame AB magnitude at 10 pc of a rest-frame F_nu (on ``nu``) seen at redshift z."""
    f = jnp.interp(band.nu * (1.0 + z), nu, fnu)
    flux = (1.0 + z) * jnp.trapezoid(f * band.response / band.nu, band.nu)
    ref = AB_FNU * jnp.trapezoid(band.response / band.nu, band.nu)
    return -2.5 * jnp.log10(flux / ref)


def lookback_time(z, cosmo, zmax: float = 10.0, n: int = 20001):
    """Lookback time [Gyr] of a ggah_mod cosmology (trapezoid rule on n points up to zmax)."""
    from ggah_mod.cosmology import hubble_e

    zg = jnp.linspace(0.0, zmax, n)
    f = 1.0 / ((1.0 + zg) * hubble_e(zg, cosmo))
    tl = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (f[1:] + f[:-1]) * jnp.diff(zg))])
    return HUBBLE_TIME_GYR / (100.0 * cosmo.h) * jnp.interp(z, zg, tl)


def passive_mags(ssp: SSPGrid, bands, z, zf: float, tau: float, cosmo, norm=None):
    """Apparent AB magnitudes [len(z), len(bands)] of an exponential population formed at zf.

    ``norm = (band, z0, m0)`` shifts all magnitudes so that the galaxy has magnitude m0 through
    ``band`` at z0. ``cosmo`` is a ggah_mod cosmology.
    """
    from ggah_mod.cosmology import distance_modulus

    def apparent(zz, bl):
        zz = jnp.atleast_1d(jnp.asarray(zz, jnp.float64))
        T = lookback_time(zf, cosmo) - lookback_time(zz, cosmo)
        seds = csp_fnu(ssp, T, tau)
        mags = [jax.vmap(ab_mag_10pc, in_axes=(None, 1, 0, None))(ssp.nu, seds, zz, b) for b in bl]
        return jnp.stack(mags, axis=1) + distance_modulus(zz, cosmo)[:, None]

    mags = apparent(z, list(bands))
    if norm is not None:
        band, z0, m0 = norm
        mags = mags + (m0 - apparent(z0, [band])[0, 0])
    return mags
