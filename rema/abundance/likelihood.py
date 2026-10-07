"""Likelihood of the cluster counts, with the weak-lensing calibration of the richness.

    ln P(theta | data) = ln L_counts + ln L_WL + ln prior.

- Counts: Gaussian with a fixed covariance (Poisson at a reference model plus the super-sample
  term of :mod:`rema.abundance.covariance`), or Poisson.
- Weak lensing: in the bins inside the range of McClintock et al. (2019) (DES Y1, lambda_DES >= 20,
  0.2 <= z <= 0.65), the model's ln <M200m | bin> against their <M | lambda, z> at the bin's mean
  richness in DES units, lambda_DES = lambda_rema exp(-ln_s0 - s1 (z - z_norm)), with the
  covariance of their (M0, F, G) (stat + sys). This calibrates the mass-richness relation the way
  the DES Y1 cluster analysis did, rather than inverting <M | lambda> (which ignores the
  Eddington bias).
- Priors: :data:`DEFAULT_PRIORS`; Planck 2018 on h and n_s, BBN on omega_b, the measured
  normalisation of rema's richness against DES Y1, flat elsewhere; the emu_pk training box is
  enforced (outside it the log-posterior is -inf).

Parameters are passed as a vector over ``free`` names; the others take ``fixed`` values or the
defaults (:data:`DEFAULTS`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..model.cosmo import _float64
from .counts import COSMO_DEFAULTS, CountsModel
from .data import DataVector
from .mor import LN_S0, MCCLINTOCK19, S1, MassRichness, mcclintock_cov, mcclintock_lnm

_MOR0 = MassRichness()
DEFAULTS = {**COSMO_DEFAULTS, "mor_a": _MOR0.a, "mor_b": _MOR0.b, "mor_c": _MOR0.c,
            "sigma_int": _MOR0.sigma_int, "ln_s0": LN_S0, "s1": S1, "dz_bias": 0.0}

#: name -> ("flat", lo, hi) or ("gauss", mean, sigma); "omega_b" is a Gaussian on Omega_b h^2.
DEFAULT_PRIORS = {
    "Omega_m": ("flat", 0.15, 0.55), "ln10A_s": ("flat", 1.7, 4.0), "w0": ("flat", -1.5, -0.5),
    "wa": ("flat", -1.0, 0.6), "sum_mnu": ("flat", 0.0, 0.6),
    "h": ("gauss", 0.6736, 0.0054), "n_s": ("gauss", 0.9649, 0.0042),
    "Omega_b": ("omega_b", 0.02237, 0.00015),
    "mor_a": ("flat", 2.0, 5.0), "mor_b": ("flat", 0.2, 1.5), "mor_c": ("flat", -3.0, 3.0),
    "sigma_int": ("flat", 0.02, 0.8),
    "ln_s0": ("gauss", LN_S0, 0.05), "s1": ("gauss", S1, 0.25), "dz_bias": ("gauss", 0.0, 0.005),
}


def _emu_box():
    from emu_pk.box import BOX

    return BOX


@dataclass
class WLData:
    """The bins of the weak-lensing term: indices, mean richness (rema) and redshift, and the
    covariance of McClintock's ln M at the fiducial richness normalisation."""

    i: np.ndarray
    j: np.ndarray
    lam: np.ndarray
    z: np.ndarray
    icov: np.ndarray

    @classmethod
    def from_data(cls, dv: DataVector, extra_sigma: float = 0.05) -> "WLData | None":
        """``extra_sigma``: an uncorrelated error of ln M per bin (the relation's parameter
        covariance alone has rank 3), of the order of the per-bin errors of the stacked masses."""
        lam, z = np.asarray(dv.lam_mean), np.asarray(dv.z_mean)
        lam_des = lam * np.exp(-(LN_S0 + S1 * (z - _MOR0.z_norm)))
        z0, z1 = MCCLINTOCK19["z_range"]
        ok = np.isfinite(lam) & (lam_des >= MCCLINTOCK19["lam_min"]) & (z >= z0) & (z <= z1)
        i, j = np.nonzero(ok)
        if i.size == 0:
            return None
        cov = mcclintock_cov(lam_des[i, j], z[i, j]) + extra_sigma**2 * np.eye(i.size)
        return cls(i, j, lam[i, j], z[i, j], np.linalg.inv(cov))


class Likelihood:
    """See the module docstring. ``cov``: covariance of the counts (flattened [n_lambda * n_z]);
    by default Poisson at the model of ``reference`` (default: the starting point)."""

    def __init__(self, data: DataVector, model: CountsModel, free: Sequence[str], *,
                 fixed: Mapping[str, float] | None = None, priors: Mapping | None = None,
                 wl: bool = True, kind: str = "gaussian", cov: np.ndarray | None = None,
                 reference: Mapping[str, float] | None = None):
        unknown = set(free) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown parameter(s) {sorted(unknown)}; known: {sorted(DEFAULTS)}")
        if kind not in ("gaussian", "poisson"):
            raise ValueError(f"kind {kind!r}: 'gaussian' or 'poisson'")
        self.data, self.model, self.kind = data, model, kind
        self.free = tuple(free)
        self.fixed = {**DEFAULTS, **(fixed or {})}
        self.priors = {**DEFAULT_PRIORS, **(priors or {})}
        self.counts = np.asarray(data.counts, np.float64)
        self.wl = WLData.from_data(data) if wl else None
        self._predict = jax.jit(model.predict)
        self._logpost = jax.jit(self._logpost_impl)
        self._grad = jax.jit(jax.value_and_grad(self._logpost_impl))
        if cov is None and kind == "gaussian":
            ref = {**self.fixed, **(reference or {})}
            cov = np.diag(np.maximum(np.asarray(self.predict(ref)["N"]).ravel(), 1.0))
        self.cov = None if cov is None else np.asarray(cov, np.float64)
        self._icov = None if cov is None else np.linalg.inv(self.cov)

    # ------------------------------------------------------------------ parameters
    def x0(self) -> np.ndarray:
        """Starting point: the ``fixed`` values, or the prior mean of a parameter with a Gaussian
        prior when its value is more than 3 sigma away (the rema fiducial h = 0.7 and
        Omega_b = 0.0493 are not Planck's)."""
        out = []
        h = self.fixed["h"]
        if "h" in self.free and self.priors["h"][0] == "gauss":
            _, a, b = self.priors["h"]
            h = h if abs(h - a) <= 3 * b else a
        for p in self.free:
            v = self.fixed[p]
            kind, a, b = self.priors[p]
            if kind == "gauss" and abs(v - a) > 3 * b:
                v = a
            elif kind == "omega_b" and abs(v * h**2 - a) > 3 * b:
                v = a / h**2
            out.append(v)
        return np.array(out, np.float64)

    def theta(self, x) -> dict:
        th = dict(self.fixed)
        th.update({p: x[k] for k, p in enumerate(self.free)})
        return th

    def bounds_of(self, p: str) -> tuple[float, float]:
        kind, a, b = self.priors[p]
        return (a, b) if kind == "flat" else (a - 6 * b, a + 6 * b)

    def bounds(self) -> list[tuple[float, float]]:
        """Box of each free parameter: flat priors, Gaussians +- 6 sigma."""
        out = []
        for p in self.free:
            kind, a, b = self.priors[p]
            if kind == "flat":
                out.append((a, b))
            elif kind == "omega_b":
                hl, hh = (self.bounds_of("h") if "h" in self.free else (self.fixed["h"],) * 2)
                out.append(((a - 6 * b) / hh**2, (a + 6 * b) / hl**2))
            else:
                out.append((a - 6 * b, a + 6 * b))
        return out

    # ------------------------------------------------------------------ pieces
    def predict(self, theta: Mapping) -> dict:
        with _float64():
            return self._predict({k: jnp.asarray(v, jnp.float64) for k, v in theta.items()})

    def logprior(self, th) -> jnp.ndarray:
        lp = jnp.asarray(0.0)
        for p in self.free:
            kind, a, b = self.priors[p]
            v = th[p]
            if kind == "flat":
                lp = lp + jnp.where((v >= a) & (v <= b), 0.0, -jnp.inf)
            elif kind == "omega_b":
                lp = lp - 0.5 * ((v * th["h"] ** 2 - a) / b) ** 2
            else:
                lp = lp - 0.5 * ((v - a) / b) ** 2
        # The emulator's training box (checked on concrete values only inside emu_pk).
        h2 = th["h"] ** 2
        omega_nu = th["sum_mnu"] / 93.14
        vals = {"omega_b": th["Omega_b"] * h2, "omega_cdm": (th["Omega_m"] - th["Omega_b"]) * h2 - omega_nu,
                "h": th["h"], "n_s": th["n_s"], "ln10A_s": th["ln10A_s"], "sum_mnu": th["sum_mnu"],
                "w0": th["w0"], "wa": th["wa"]}
        for k, v in vals.items():
            lo, hi = _emu_box()[k]
            lp = lp + jnp.where((v >= lo) & (v <= hi), 0.0, -jnp.inf)
        return lp

    def loglike_counts(self, pred) -> jnp.ndarray:
        N = pred["N"]
        if self.kind == "poisson":
            n = jnp.asarray(self.counts)
            return jnp.sum(jnp.where(N > 0, n * jnp.log(jnp.maximum(N, 1e-300)) - N, 0.0))
        r = (jnp.asarray(self.counts) - N).ravel()
        return -0.5 * r @ jnp.asarray(self._icov) @ r

    def wl_residual(self, th, pred) -> jnp.ndarray:
        """Model ln <M | bin> minus McClintock's at the bins' mean richness in DES units."""
        w = self.wl
        lam_des = jnp.asarray(w.lam) * jnp.exp(-(th["ln_s0"] + th["s1"] * (jnp.asarray(w.z) - _MOR0.z_norm)))
        return pred["lnM"][w.i, w.j] - mcclintock_lnm(lam_des, jnp.asarray(w.z))

    def loglike_wl(self, th, pred) -> jnp.ndarray:
        if self.wl is None:
            return jnp.asarray(0.0)
        r = self.wl_residual(th, pred)
        return -0.5 * r @ jnp.asarray(self.wl.icov) @ r

    def _logpost_impl(self, x) -> jnp.ndarray:
        th = self.theta(x)
        lp = self.logprior(th)
        pred = self.model.predict(th)
        ll = self.loglike_counts(pred) + self.loglike_wl(th, pred)
        return jnp.where(jnp.isfinite(lp), lp + ll, -jnp.inf)

    # ------------------------------------------------------------------ public
    def logpost(self, x) -> float:
        with _float64():
            return float(self._logpost(jnp.asarray(x, jnp.float64)))

    def value_and_grad(self, x):
        with _float64():
            v, g = self._grad(jnp.asarray(x, jnp.float64))
        return float(v), np.asarray(g, np.float64)

    def chi2(self, x) -> dict:
        """chi^2 of the counts and of the weak-lensing term at ``x``."""
        with _float64():
            xx = jnp.asarray(x, jnp.float64)
            th = self.theta(xx)
            pred = self.model.predict(th)
            return {"counts": float(-2 * self.loglike_counts(pred)) if self.kind == "gaussian" else np.nan,
                    "wl": float(-2 * self.loglike_wl(th, pred)),
                    "n_counts": int(self.counts.size), "n_wl": 0 if self.wl is None else int(self.wl.i.size)}
