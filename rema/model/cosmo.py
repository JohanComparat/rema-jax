"""Cosmological distances, tabulated once from :mod:`ggah_mod.cosmology`.

Lengths are in h^-1 Mpc, the redMaPPer convention: radii such as r_lambda = r0 (lambda/100)^beta
are physical h^-1 Mpc, so the angular scale of a cluster is set by the angular diameter
distance D_A(z) in Mpc/h.

The table is a pytree, so it can be passed to jitted functions and differentiated through
(the interpolation is linear in the tabulated values and piecewise linear in z).

>>> import numpy as np
>>> tab = CosmoTable.create(Omega_m=0.3, h=0.7)
>>> bool(np.isclose(float(tab.da(0.5)), 881.3, rtol=1e-3))
True
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

DEG = np.pi / 180.0


@contextlib.contextmanager
def _float64():
    """Enable 64-bit JAX inside the block (the API name moved between JAX releases)."""
    ctx = getattr(jax, "enable_x64", None)
    if ctx is None:  # pragma: no cover - older JAX
        from jax.experimental import enable_x64 as ctx
        with ctx():
            yield
        return
    with ctx(True):
        yield


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CosmoTable:
    """D_A, comoving distance, E(z) and dV/dz/dOmega on a regular redshift grid.

    Attributes
    ----------
    z : grid, ``z[0] = 0``.
    da : angular diameter distance [Mpc/h].
    dc : comoving distance [Mpc/h].
    ez : E(z) = H(z)/H0.
    dvdz : comoving volume element dV/dz/dOmega [(Mpc/h)^3 / sr].
    """

    z: jnp.ndarray
    da_tab: jnp.ndarray
    dc_tab: jnp.ndarray
    ez_tab: jnp.ndarray
    dvdz_tab: jnp.ndarray
    Omega_m: float = field(default=0.3, metadata=dict(static=True))
    h: float = field(default=0.7, metadata=dict(static=True))

    @classmethod
    def create(cls, Omega_m: float = 0.3, h: float = 0.7, zmax: float = 2.0,
               dz: float = 1e-3) -> "CosmoTable":
        """Tabulate the distances of a flat cosmology with ggah_mod, in float64."""
        from ggah_mod.cosmology import (Cosmology, angular_diameter_distance,
                                        comoving_distance, comoving_volume_element,
                                        hubble_e)

        z = np.arange(0.0, zmax + dz / 2, dz)
        with _float64():
            cosmo = Cosmology.create(Omega_m=Omega_m, h=h)
            zz = jnp.asarray(z, dtype=jnp.float64)
            da = np.asarray(angular_diameter_distance(zz, cosmo), dtype=np.float64)
            dc = np.asarray(comoving_distance(zz, cosmo), dtype=np.float64)
            ez = np.asarray(hubble_e(zz, cosmo), dtype=np.float64)
            dvdz = np.asarray(comoving_volume_element(zz, cosmo), dtype=np.float64)
        # Device arrays (not numpy): numpy leaves would push every jitted call that receives
        # the table onto JAX's slow dispatch path.
        return cls(z=jnp.asarray(z), da_tab=jnp.asarray(da), dc_tab=jnp.asarray(dc),
                   ez_tab=jnp.asarray(ez), dvdz_tab=jnp.asarray(dvdz),
                   Omega_m=float(Omega_m), h=float(h))

    # Lookups (linear interpolation; work on numpy or traced inputs) -----------
    def da(self, z):
        """Angular diameter distance [Mpc/h]."""
        return jnp.interp(z, self.z, self.da_tab)

    def dc(self, z):
        """Comoving distance [Mpc/h]."""
        return jnp.interp(z, self.z, self.dc_tab)

    def ez(self, z):
        """E(z) = H(z)/H0."""
        return jnp.interp(z, self.z, self.ez_tab)

    def dvdz(self, z):
        """Comoving volume element [(Mpc/h)^3 / sr]."""
        return jnp.interp(z, self.z, self.dvdz_tab)

    def mpc_per_deg(self, z):
        """Physical h^-1 Mpc subtended by one degree at redshift z (redMaPPer ``mpc_scale``)."""
        return self.da(z) * DEG

    def volume_factor(self, z, zref):
        """redMaPPer's zred volume factor, Delta D_c(zref) / Delta D_c(z) = E(z) / E(zref)."""
        return self.ez(z) / self.ez(zref)
