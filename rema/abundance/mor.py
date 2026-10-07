"""Mass-richness relation: P(ln lambda | M, z), and the DES Y1 weak-lensing calibration.

The richness of a halo of mass M (M200m, h^-1 Msun) at redshift z is log-normal,

    <ln lambda_DES | M, z> = a + b ln(M / M_piv) + c ln((1 + z) / (1 + z_piv)),
    Var(ln lambda | M, z) = sigma_int^2 + (e^mu - 1) / e^(2 mu)   (intrinsic + Poisson, Costanzi
                                                                   et al. 2019),

in the units of the DES Y1 redMaPPer catalogue, and rema's richness is
ln lambda_rema = ln lambda_DES + ln_s0 + s1 (z - z_norm) (the normalisation measured on common
clusters, ``INVESTIGATE.md`` item 5).

McClintock et al. (2019, Table 4) calibrated <M200m | lambda, z> = M0 (lambda/40)^F
((1+z)/1.35)^G with weak lensing on DES Y1 redMaPPer (lambda >= 20, 0.2 <= z <= 0.65):
log10 M0 = 14.489 +- 0.011 (stat) +- 0.019 (sys) in Msun for h = 0.7 (their flat LCDM with
Omega_m = 0.3, H0 = 70), F = 1.356 +- 0.051 +- 0.008, G = -0.30 +- 0.30 +- 0.06. Its inverse
gives the default (a, b, c) = (ln 40, 1/F, -G/F) at M_piv = M0, but the relation is a mean mass
at fixed richness, not a mean richness at fixed mass: :func:`mcclintock_lnm` is used as a
weak-lensing measurement of <M | lambda> in the likelihood instead
(:mod:`rema.abundance.likelihood`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr

#: McClintock et al. (2019), Table 4 (masses in Msun for h = 0.7).
MCCLINTOCK19 = {"log10_m0_msun": 14.489, "sig_log10_m0": float(np.hypot(0.011, 0.019)),
                "f": 1.356, "sig_f": float(np.hypot(0.051, 0.008)),
                "g": -0.30, "sig_g": float(np.hypot(0.30, 0.06)),
                "lam_piv": 40.0, "z_piv": 0.35, "h": 0.7, "lam_min": 20.0, "z_range": (0.2, 0.65)}
#: log10 M0 in h^-1 Msun.
LOG10_M0 = MCCLINTOCK19["log10_m0_msun"] + np.log10(MCCLINTOCK19["h"])
#: ln(lambda_rema / lambda_DES): 1.05 at z = 0.2, 0.85 at z = 0.6 (matched clusters, lambda_DES >= 20).
LN_S0 = float(np.log(1.05) + 0.5 * (np.log(0.85) - np.log(1.05)))
S1 = float((np.log(0.85) - np.log(1.05)) / 0.4)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MassRichness:
    """Parameters of P(ln lambda | M, z) (see the module docstring)."""

    a: float = float(np.log(40.0))
    b: float = 1.0 / MCCLINTOCK19["f"]
    c: float = -MCCLINTOCK19["g"] / MCCLINTOCK19["f"]
    sigma_int: float = 0.25
    ln_s0: float = LN_S0
    s1: float = S1
    m_piv: float = field(default=float(10**LOG10_M0), metadata=dict(static=True))
    z_piv: float = field(default=0.35, metadata=dict(static=True))
    z_norm: float = field(default=0.4, metadata=dict(static=True))


def mean_lnlam(mor: MassRichness, lnm, z):
    """<ln lambda_rema | M, z>."""
    mu_des = (mor.a + mor.b * (lnm - jnp.log(mor.m_piv))
              + mor.c * jnp.log((1.0 + z) / (1.0 + mor.z_piv)))
    return mu_des + mor.ln_s0 + mor.s1 * (z - mor.z_norm)


def var_lnlam(mor: MassRichness, mu):
    """Var(ln lambda | M, z): intrinsic plus Poisson."""
    lam = jnp.exp(mu)
    return mor.sigma_int**2 + jnp.maximum(lam - 1.0, 0.0) / lam**2


def p_bins(mu, var, ln_edges):
    """Probabilities [..., n_bins] that ln lambda ~ N(mu, var) falls in each bin of ``ln_edges``.

    >>> import numpy as np
    >>> p = p_bins(np.log(30.0), 0.04, np.log([1e-3, 20, 40, 1e6]))
    >>> bool(abs(float(p.sum()) - 1) < 1e-6), bool(p[1] > 0.8)
    (True, True)
    """
    s = jnp.sqrt(var)[..., None]
    e = jnp.asarray(ln_edges)
    cdf = ndtr((e - jnp.asarray(mu)[..., None]) / s)
    return cdf[..., 1:] - cdf[..., :-1]


def mcclintock_lnm(lam_des, z, log10_m0: float = LOG10_M0, f: float = MCCLINTOCK19["f"],
                   g: float = MCCLINTOCK19["g"]):
    """ln <M200m | lambda_DES, z> [h^-1 Msun] of McClintock et al. (2019)."""
    return (jnp.log(10.0) * log10_m0 + f * jnp.log(lam_des / MCCLINTOCK19["lam_piv"])
            + g * jnp.log((1.0 + z) / (1.0 + MCCLINTOCK19["z_piv"])))


def mcclintock_cov(lam_des, z) -> np.ndarray:
    """Covariance of :func:`mcclintock_lnm` at points (lam_des, z), from the stat + sys errors of
    (log10 M0, F, G), which are common to all points."""
    lam_des, z = np.asarray(lam_des, np.float64), np.asarray(z, np.float64)
    J = np.stack([np.full(lam_des.shape, np.log(10.0)), np.log(lam_des / MCCLINTOCK19["lam_piv"]),
                  np.log((1.0 + z) / (1.0 + MCCLINTOCK19["z_piv"]))], axis=-1)
    var = np.array([MCCLINTOCK19["sig_log10_m0"], MCCLINTOCK19["sig_f"], MCCLINTOCK19["sig_g"]]) ** 2
    return (J * var) @ J.T
