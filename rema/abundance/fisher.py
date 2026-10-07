"""Fisher matrices, forecasts and parameter shifts of the counts likelihood (forward-mode autodiff).

The Jacobians are ``jax.jacfwd`` of the model through the halo mass function, the volume, the
mass-richness relation and the finder's response, in float64. With a Gaussian likelihood of fixed
covariance C,

    F = J^T C^-1 J + J_WL^T C_WL^-1 J_WL + F_prior,

and the shift of the best fit when the data differ from the model by delta (e.g. a model that
ignores the finder's response fitted to counts that have it) is
dtheta = F^-1 J^T C^-1 delta (+ the weak-lensing part).
"""

from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from ..model.cosmo import _float64
from .likelihood import Likelihood


def jacobians(lik: Likelihood, x) -> dict:
    """dN/dx [n_bins, n_free], the weak-lensing residual's Jacobian, and dsigma8/dx."""
    model = lik.model

    def counts(xx):
        return model.predict(lik.theta(xx))["N"].ravel()

    def wl(xx):
        th = lik.theta(xx)
        return lik.wl_residual(th, model.predict(th))

    def s8(xx):
        return model.sigma8(lik.theta(xx))

    with _float64():
        xx = jnp.asarray(x, jnp.float64)
        out = {"N": np.asarray(jax.jit(jax.jacfwd(counts))(xx)),
               "sigma8": np.asarray(jax.jit(jax.grad(s8))(xx)),
               "sigma8_value": float(jax.jit(s8)(xx))}
        if lik.wl is not None:
            out["WL"] = np.asarray(jax.jit(jax.jacfwd(wl))(xx))
    return out


def prior_fisher(lik: Likelihood, x) -> np.ndarray:
    """Gaussian priors of the free parameters (flat priors add nothing)."""
    F = np.zeros((len(lik.free),) * 2)
    th = lik.theta(np.asarray(x, np.float64))
    for k, p in enumerate(lik.free):
        kind, a, b = lik.priors[p]
        if kind == "gauss":
            F[k, k] += 1.0 / b**2
        elif kind == "omega_b":
            F[k, k] += (float(th["h"]) ** 2 / b) ** 2
    return F


def fisher(lik: Likelihood, x, jac: Mapping | None = None, counts_cov: np.ndarray | None = None) -> np.ndarray:
    """Fisher matrix of the free parameters at ``x``: counts (Gaussian with the likelihood's or
    ``counts_cov`` covariance, Poisson otherwise), weak lensing and priors."""
    jac = jac or jacobians(lik, x)
    J = jac["N"]
    if counts_cov is not None:
        F = J.T @ np.linalg.solve(counts_cov, J)
    elif lik.kind == "gaussian":
        F = J.T @ lik._icov @ J
    else:
        N = np.asarray(lik.predict(lik.theta(np.asarray(x)))["N"]).ravel()
        F = J.T @ (J / np.maximum(N, 1e-12)[:, None])
    if "WL" in jac:
        F = F + jac["WL"].T @ lik.wl.icov @ jac["WL"]
    return F + prior_fisher(lik, x)


def constraints(F: np.ndarray, names, floor: float = 1e-12) -> dict:
    """1 sigma marginal errors (eigenvalues of F floored at ``floor`` times the largest)."""
    w, v = np.linalg.eigh(F)
    w = np.maximum(w, floor * w.max())
    cov = (v / w) @ v.T
    return {n: float(np.sqrt(cov[k, k])) for k, n in enumerate(names)} | {"_cov": cov}


def sigma8_error(jac: Mapping, cov: np.ndarray) -> float:
    g = np.asarray(jac["sigma8"])
    return float(np.sqrt(g @ cov @ g))


def shift(lik: Likelihood, F: np.ndarray, jac: Mapping, delta_counts: np.ndarray,
          delta_wl: np.ndarray | None = None) -> np.ndarray:
    """Shift of the best-fit parameters for a change ``delta_counts`` of the data (flattened)."""
    d = np.asarray(delta_counts, np.float64).ravel()
    if lik.kind == "gaussian":
        b = jac["N"].T @ lik._icov @ d
    else:
        N = np.asarray(lik.predict(lik.theta(lik.x0()))["N"]).ravel()
        b = jac["N"].T @ (d / np.maximum(N, 1e-12))
    if delta_wl is not None and "WL" in jac:
        b = b + jac["WL"].T @ lik.wl.icov @ np.asarray(delta_wl, np.float64)
    return np.linalg.solve(F, b)
