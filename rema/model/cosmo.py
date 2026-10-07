"""Cosmological distances, tabulated from :mod:`ggah_mod.cosmology`.

Lengths are in h^-1 Mpc, the redMaPPer convention: radii such as r_lambda = r0 (lambda/100)^beta
are physical h^-1 Mpc, so the angular scale of a cluster is set by the angular diameter
distance D_A(z) in Mpc/h.

The table is a pytree whose leaves are all arrays, so it can be passed to jitted functions,
swapped for the table of another cosmology without recompiling them, and differentiated through
(the interpolation is linear in the tabulated values and piecewise linear in z).
:meth:`CosmoTable.from_cosmology` builds it inside a trace, and :meth:`CosmoTable.jvp` gives its
derivatives with respect to the cosmological parameters.

>>> import numpy as np
>>> tab = CosmoTable.create(Omega_m=0.3, h=0.7)
>>> bool(np.isclose(float(tab.da(0.5)), 881.3, rtol=1e-3)), round(float(tab.Omega_m), 6)
(True, 0.3)
"""

from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..config import CosmologyConfig

DEG = np.pi / 180.0

#: Cosmological parameters carried by a table, in the order of :attr:`CosmoTable.params`.
PARAMS = tuple(f.name for f in dataclasses.fields(CosmologyConfig))


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


def _grid(zmax: float, dz: float) -> np.ndarray:
    return np.arange(0.0, zmax + dz / 2, dz)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CosmoTable:
    """D_A, comoving distance, E(z) and dV/dz/dOmega on a regular redshift grid.

    Attributes
    ----------
    z : grid, ``z[0] = 0``.
    da_tab : angular diameter distance [Mpc/h] (lookup: :meth:`da`).
    dc_tab : comoving distance [Mpc/h].
    ez_tab : E(z) = H(z)/H0.
    dvdz_tab : comoving volume element dV/dz/dOmega [(Mpc/h)^3 / sr].
    params : the parameters of :data:`PARAMS` (a leaf, so another cosmology does not change the
        tree structure).
    """

    z: jnp.ndarray
    da_tab: jnp.ndarray
    dc_tab: jnp.ndarray
    ez_tab: jnp.ndarray
    dvdz_tab: jnp.ndarray
    params: jnp.ndarray

    @classmethod
    def from_cosmology(cls, cosmo, z) -> "CosmoTable":
        """Tables of a ggah_mod cosmology on the grid ``z``; traceable.

        Inside a trace build ``cosmo`` with the plain ``Cosmology(...)`` constructor
        (``Cosmology.create`` validates on concrete values).
        """
        from ggah_mod.cosmology.background import comoving_distance, hubble_e, transverse_distance

        z = jnp.asarray(z)
        chi = comoving_distance(z, cosmo)
        fk = transverse_distance(chi, cosmo)
        ez = hubble_e(z, cosmo)
        params = jnp.stack([jnp.asarray(getattr(cosmo, p), dtype=chi.dtype) for p in PARAMS])
        return cls(z=z, da_tab=fk / (1.0 + z), dc_tab=chi, ez_tab=ez,
                   dvdz_tab=cosmo.hubble_distance * fk**2 / ez, params=params)

    @classmethod
    def from_config(cls, c: CosmologyConfig | None = None, zmax: float = 2.0,
                    dz: float = 1e-3) -> "CosmoTable":
        """Tabulate in float64, stored in the default JAX precision."""
        c = c or CosmologyConfig()
        with _float64():
            tab = cls.from_cosmology(c.to_ggah(), jnp.asarray(_grid(zmax, dz), dtype=jnp.float64))
            tab = jax.tree_util.tree_map(lambda a: np.asarray(a, np.float64), tab)
        # Device arrays (not numpy): numpy leaves would push every jitted call that receives
        # the table onto JAX's slow dispatch path.
        return jax.tree_util.tree_map(jnp.asarray, tab)

    @classmethod
    def create(cls, Omega_m: float = 0.3, h: float = 0.7, zmax: float = 2.0,
               dz: float = 1e-3, **kw) -> "CosmoTable":
        """Tabulate the distances of a flat cosmology with ggah_mod, in float64.

        ``kw``: the other fields of :class:`~rema.config.CosmologyConfig`.
        """
        return cls.from_config(CosmologyConfig(Omega_m=Omega_m, h=h, **kw), zmax=zmax, dz=dz)

    @classmethod
    def jvp(cls, c: CosmologyConfig | None, params: Sequence[str], zmax: float = 2.0,
            dz: float = 1e-3) -> tuple["CosmoTable", dict[str, "CosmoTable"]]:
        """The table of ``c`` and, for each parameter, its derivative table d(table)/d(param).

        Forward mode in float64, stored in the default precision. A derivative table goes with
        the table as the tangent of ``jax.jvp`` through any function of a :class:`CosmoTable`.
        Its ``z`` is the tangent of the grid, zero: read its values (``da_tab``, ...) on the
        table's grid, not with its lookup methods. The other parameters are held fixed
        (``Omega_m`` at fixed ``Omega_b``).
        """
        c = c or CosmologyConfig()
        unknown = set(params) - set(PARAMS)
        if unknown:
            raise ValueError(f"unknown cosmological parameter(s) {sorted(unknown)}; known: {PARAMS}")
        out = {}
        with _float64():
            z = jnp.asarray(_grid(zmax, dz), dtype=jnp.float64)
            cosmo = c.to_ggah()
            zero = jax.tree_util.tree_map(lambda x: jnp.zeros((), jnp.float64), cosmo)
            fn = lambda cc: cls.from_cosmology(cc, z)
            tab = fn(cosmo)
            for p in params:
                dcos = dataclasses.replace(zero, **{p: jnp.ones((), jnp.float64)})
                _, out[p] = jax.jvp(fn, (cosmo,), (dcos,))
            tab, out = jax.tree_util.tree_map(lambda a: np.asarray(a, np.float64), (tab, out))
        return jax.tree_util.tree_map(jnp.asarray, (tab, out))

    # Parameters ----------------------------------------------------------------
    @property
    def Omega_m(self):
        return self.params[PARAMS.index("Omega_m")]

    @property
    def h(self):
        return self.params[PARAMS.index("h")]

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
