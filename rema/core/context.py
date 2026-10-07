"""Everything the cluster kernels need, bundled as one pytree.

:class:`FilterModel` holds the red-sequence model, the chi^2 background, the cosmology table and
m*(z), plus the static constants of the matched filter. It is passed as an argument to the jitted
kernels (array leaves are traced, constants are static), so a recalibrated model does not trigger
recompilation as long as array shapes are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from ..config import RemaConfig
from ..model.background import ChisqBkg
from ..model.cosmo import CosmoTable
from ..model.profiles import MStar, maxmag_from_mstar
from ..model.redsequence import RSModel


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FilterModel:
    rs: RSModel
    bkg: ChisqBkg
    cosmo: CosmoTable
    mstar_z: jnp.ndarray
    mstar_m: jnp.ndarray
    alpha: float = field(default=-1.0, metadata=dict(static=True))
    lval: float = field(default=0.2, metadata=dict(static=True))
    chisq_max: float = field(default=20.0, metadata=dict(static=True))
    nfw_rs: float = field(default=0.15, metadata=dict(static=True))
    nfw_rcore: float = field(default=0.1, metadata=dict(static=True))
    rsig: float = field(default=0.05, metadata=dict(static=True))
    eps: float = field(default=0.015, metadata=dict(static=True))
    chisq_mode: str = field(default="lupt", metadata=dict(static=True))
    zeropoint: float = field(default=22.5, metadata=dict(static=True))

    @property
    def iref(self) -> int:
        return self.rs.iref

    @property
    def ncol(self) -> int:
        return self.rs.ncol

    def mstar(self, z):
        return jnp.interp(z, self.mstar_z, self.mstar_m)

    def maxmag(self, z):
        return maxmag_from_mstar(self.mstar(z), self.lval)

    def mpc_per_deg(self, z):
        return self.cosmo.mpc_per_deg(z)

    @classmethod
    def create(cls, rs: RSModel, bkg: ChisqBkg, cfg: RemaConfig,
               cosmo: CosmoTable | None = None, mstar: MStar | None = None) -> "FilterModel":
        cosmo = cosmo or CosmoTable.from_config(cfg.cosmology)
        mstar = mstar or MStar(cfg.model.mstar)
        m = cfg.model
        return cls(rs=rs, bkg=bkg, cosmo=cosmo, mstar_z=jnp.asarray(mstar.z),
                   mstar_m=jnp.asarray(mstar.m), alpha=m.alpha, lval=m.lval_reference,
                   chisq_max=m.chisq_max, nfw_rs=m.nfw_rs, nfw_rcore=m.nfw_rcore, rsig=m.rsig,
                   eps=cfg.survey.flux_floor, chisq_mode=m.chisq_mode, zeropoint=cfg.survey.zeropoint)
