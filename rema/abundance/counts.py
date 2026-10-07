"""Predicted cluster counts N(lambda_i, z_j) for a cosmology and a mass-richness relation.

    N_ij = Omega_j  int dz dV/dz/dOmega(z) K_ij(z)  int dln M  dn/dln M(M, z) P_i(M, z),

with Omega_j the area of redshift bin j (sr), dn/dln M the Tinker et al. (2008) mass function of
M200m (ggah_mod: emu_pk linear spectrum of the cold matter, sigma(M) and its slope by autodiff),
P_i the probability that the richness measured in the catalogue falls in bin i
(:mod:`rema.abundance.mor`), and K_ij the probability that z_lambda falls in bin j: Gaussian with
the clusters' median z_lambda error, a bias ``dz_bias``.

The finder's response to the cosmology (:mod:`rema.abundance.response`) enters when the
mass-richness relation describes the richness the finder would measure in the true cosmology
theta, while the catalogue was made in the fiducial one: the measured
ln lambda_fid = ln lambda_theta - R(theta; z, lambda) and z_fid = z_theta - Delta z(theta). The
mean mass and bias of each bin are also returned (weak-lensing term, super-sample covariance).

Everything is a jittable, differentiable function of the parameters (a dict of scalars):
cosmology (``Omega_m``, ``ln10A_s``, ``h``, ``n_s``, ``Omega_b``, ``sum_mnu``, ``w0``, ``wa``),
mass-richness (``mor_a``, ``mor_b``, ``mor_c``, ``sigma_int``, ``ln_s0``, ``s1``) and ``dz_bias``.
ggah_mod needs 64-bit JAX: call it under ``jax_enable_x64`` (or :func:`rema.model.cosmo._float64`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr

from .data import DataVector
from .mor import MassRichness, mean_lnlam, p_bins, var_lnlam
from .response import ResponseTable, delta_lnlam, delta_z

#: Defaults of the cosmological parameters (the rema fiducial, Planck 2018 n_s and A_s).
COSMO_DEFAULTS = {"Omega_m": 0.3, "Omega_b": 0.0493, "h": 0.7, "n_s": 0.9649, "ln10A_s": 3.044,
                  "sum_mnu": 0.06, "w0": -1.0, "wa": 0.0}
#: Mass-richness parameter names -> MassRichness fields.
MOR_FIELDS = {"mor_a": "a", "mor_b": "b", "mor_c": "c", "sigma_int": "sigma_int", "ln_s0": "ln_s0",
              "s1": "s1"}
DEG2_SR = (np.pi / 180.0) ** 2


def cosmology(theta: Mapping):
    """ggah_mod Cosmology of ``theta`` (defaults :data:`COSMO_DEFAULTS`); traceable."""
    from ggah_mod.cosmology import Cosmology

    kw = {k: theta.get(k, v) for k, v in COSMO_DEFAULTS.items()}
    return Cosmology(**kw)


def mass_richness(theta: Mapping) -> MassRichness:
    ref = MassRichness()
    return dataclasses.replace(ref, **{f: theta.get(k, getattr(ref, f)) for k, f in MOR_FIELDS.items()})


@dataclass(frozen=True)
class CountsSetup:
    """Binning, area, photo-z errors and integration grids (numpy, fixed)."""

    lam_edges: np.ndarray
    z_edges: np.ndarray
    area_deg2: np.ndarray            # [n_z]
    sigma_z: np.ndarray              # [n_lambda, n_zt] photo-z error at the true redshifts
    zt: np.ndarray                   # true-redshift grid
    lnm: np.ndarray                  # ln M200m [h^-1 Msun]
    k: np.ndarray                    # [h/Mpc]
    lam_ref: np.ndarray              # [n_lambda] richness at which Delta z is evaluated

    @classmethod
    def from_data(cls, dv: DataVector, dz: float = 0.01, zpad: float = 0.15,
                  log10m: tuple[float, float] = (12.8, 16.0), nm: int = 120, nk: int = 512,
                  sigma_z_floor: float = 0.005) -> "CountsSetup":
        le, ze = np.asarray(dv.lam_edges, np.float64), np.asarray(dv.z_edges, np.float64)
        zt = np.arange(max(dz, ze[0] - zpad), ze[-1] + zpad + dz / 2, dz)
        zc = 0.5 * (ze[1:] + ze[:-1])
        sz = np.asarray(dv.sigma_z, np.float64)
        fill = np.nanmedian(sz) if np.isfinite(sz).any() else 0.02
        sig = np.empty((le.size - 1, zt.size))
        for i in range(le.size - 1):
            row = sz[i] if np.isfinite(sz[i]).any() else np.full(zc.size, fill)
            ok = np.isfinite(row)
            sig[i] = np.interp(zt, zc[ok], row[ok])
        lam_ref = np.where(np.isfinite(le[1:]), np.sqrt(le[:-1] * np.where(np.isfinite(le[1:]), le[1:], 1.0)),
                           1.5 * le[:-1])
        return cls(le, ze, np.asarray(dv.area, np.float64), np.maximum(sig, sigma_z_floor), zt,
                   np.linspace(np.log(10**log10m[0]), np.log(10**log10m[1]), nm),
                   np.logspace(-4, np.log10(200.0), nk), lam_ref)


def _trapz_weights(x: np.ndarray) -> np.ndarray:
    w = np.zeros_like(x)
    d = np.diff(x)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w


class CountsModel:
    """N(lambda_i, z_j), ln <M | i, j> and the mean bias of each bin; see the module docstring.

    ``response``: the finder's :class:`~rema.abundance.response.ResponseTable` (None: the
    catalogue does not depend on the cosmology). ``selection``: optional corrections
    c[n_param, n_lambda, n_z] of the response table's parameters, N -> N exp(c . dtheta).
    """

    def __init__(self, setup: CountsSetup, response: ResponseTable | None = None, selection=None,
                 pk=None, hmf: str = "tinker08", delta: float = 200.0):
        from ggah_mod.cosmology import make_pk

        self.setup = setup
        self.response = response
        self.selection = None if selection is None else np.asarray(selection, np.float64)
        self.pk = pk if pk is not None else make_pk("emu_pk")
        self.hmf, self.delta = hmf, float(delta)
        s = setup
        le = np.asarray(s.lam_edges, np.float64)
        self._ln_edges = np.log(np.where(np.isfinite(le), le, 1e8))
        self._wz = _trapz_weights(np.asarray(s.zt))
        self._wm = _trapz_weights(np.asarray(s.lnm))
        self._omega = np.asarray(s.area_deg2) * DEG2_SR

    def halo_tables(self, cosmo):
        """dn/dln M [(h^-1 Mpc)^-3] and the Tinker et al. (2010) bias, [n_zt, n_m]."""
        from ggah_mod.halos.linear_bias import bias_tinker10
        from ggah_mod.halos.mass_function import dndm
        from ggah_mod.halos.variance import dln_sigma_dln_mass, sigma_of_mass

        s = self.setup
        k, m = jnp.asarray(s.k), jnp.exp(jnp.asarray(s.lnm))
        pcb = self.pk.pk_cb(k, jnp.asarray(s.zt), cosmo)

        def one(p, z):
            sig = sigma_of_mass(m, k, p, cosmo.rho_cold)
            ds = dln_sigma_dln_mass(m, k, p, cosmo.rho_cold)
            n = m * dndm(m, sig, ds, cosmo.rho_cold, model=self.hmf, z=z, delta=self.delta)
            return n, bias_tinker10(sig, delta=self.delta)

        return jax.vmap(one)(pcb, jnp.asarray(s.zt))

    def predict(self, theta: Mapping) -> dict:
        """{"N": [n_lambda, n_z], "lnM": ln <M200m | bin> [h^-1 Msun], "bias": <b | bin>}."""
        from ggah_mod.cosmology import comoving_volume_element

        s = self.setup
        cosmo = cosmology(theta)
        mor = mass_richness(theta)
        n, b = self.halo_tables(cosmo)
        zt, lnm = jnp.asarray(s.zt), jnp.asarray(s.lnm)
        dvdz = comoving_volume_element(zt, cosmo)
        mu = mean_lnlam(mor, lnm[None, :], zt[:, None])
        var = var_lnlam(mor, mu)
        shift_z = jnp.zeros((s.lam_ref.size, zt.size))
        if self.response is not None:
            dth = self.response.dtheta(theta)
            mu = mu - delta_lnlam(self.response, mu, zt[:, None], dth)
            shift_z = delta_z(self.response, jnp.log(jnp.asarray(s.lam_ref))[:, None], zt[None, :], dth)
        P = p_bins(mu, var, self._ln_edges)                               # [n_zt, n_m, n_lambda]
        # z_fid = z + dz_bias - Delta z(theta) + noise
        mean = zt[None, :] + theta.get("dz_bias", 0.0) - shift_z           # [n_lambda, n_zt]
        sig = jnp.asarray(s.sigma_z)
        ze = jnp.asarray(s.z_edges)
        cdf = ndtr((ze[None, :, None] - mean[:, None, :]) / sig[:, None, :])
        K = cdf[:, 1:, :] - cdf[:, :-1, :]                                  # [n_lambda, n_z, n_zt]
        wm, m = jnp.asarray(self._wm), jnp.exp(lnm)
        inner = jnp.einsum("tm,tmi,m->it", n, P, wm)
        inner_m = jnp.einsum("tm,tmi,m->it", n, P, wm * m)
        inner_b = jnp.einsum("tm,tmi,m->it", n * b, P, wm)
        w = jnp.asarray(self._wz) * dvdz
        om = jnp.asarray(self._omega)[None, :]
        N = om * jnp.einsum("ijt,it,t->ij", K, inner, w)
        Nm = om * jnp.einsum("ijt,it,t->ij", K, inner_m, w)
        Nb = om * jnp.einsum("ijt,it,t->ij", K, inner_b, w)
        tiny = 1e-300
        out = {"N": N, "lnM": jnp.log(jnp.maximum(Nm, tiny) / jnp.maximum(N, tiny)),
               "bias": Nb / jnp.maximum(N, tiny)}
        if self.selection is not None and self.response is not None:
            out["N"] = N * jnp.exp(jnp.einsum("p,pij->ij", self.response.dtheta(theta),
                                              jnp.asarray(self.selection)))
        return out

    def sigma8(self, theta: Mapping):
        """sigma_8 of the total matter at z = 0."""
        from ggah_mod.cosmology.amplitude import sigma8

        cosmo = cosmology(theta)
        k = jnp.asarray(self.setup.k)
        return sigma8(self.pk.pk(k, 0.0, cosmo), k)
