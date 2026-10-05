"""M-step of the red-sequence calibration: a joint fit of the red-sequence nodes.

Given galaxies with membership weights w_i (pmem from spectroscopically seeded cluster runs) at
their cluster redshifts z_i, the node values of the mean colours, slopes, log scatters and
partial correlations maximise

    sum_i w_i ln L_i(theta) - prior(theta),   ln L_i = -(chi^2_i + ln det S_i) / 2,

with chi^2 the photometric likelihood of :mod:`rema.model.likelihood` (asinh magnitudes by
default) and a smoothness prior on second differences of the nodes, which also fills nodes
without data. This is the M-step of an EM iteration: the weights are the posterior membership
probabilities from the previous model (redMaPPer fits each colour separately with scipy;
here all colours and their correlations are fitted together with gradients).

Pivot magnitudes are set to the weighted median reference magnitude of the members per node.
"""

from __future__ import annotations

import dataclasses
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..model.likelihood import chisq as chisq_fn
from ..model.redsequence import RSModel

log = logging.getLogger(__name__)

# Prior widths of the second differences, per parameter group.
PRIOR_SCALE = {"mean": 0.02, "slope": 0.02, "log_sigma": 0.2, "corr": 0.3}
FIT_KEYS = ("mean", "slope", "log_sigma", "corr")


def _second_diff(x):
    return x[2:] - 2 * x[1:-1] + x[:-2] if x.shape[0] >= 3 else jnp.zeros((0,) + x.shape[1:])


def fit_redsequence(rs0: RSModel, flux, ivar, z, w, *, mode: str = "lupt", eps: float = 0.015,
                    smooth: float = 1.0, maxiter: int = 300, tol: float = 1e-7,
                    fit=FIT_KEYS) -> tuple[RSModel, dict]:
    """Fit the red-sequence nodes. Returns (model, info)."""
    with jax.enable_x64(True):
        flux = jnp.asarray(np.asarray(flux, np.float64))
        ivar = jnp.asarray(np.asarray(ivar, np.float64))
        z = jnp.asarray(np.asarray(z, np.float64))
        w = jnp.asarray(np.asarray(w, np.float64))
        rs64 = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64), rs0)
        params0 = {k: getattr(rs64, k) for k in fit}
        wsum = jnp.sum(w)

        def loss(params):
            rs = dataclasses.replace(rs64, **params)
            chi2, lndet = chisq_fn(flux, ivar, rs.at(z), rs.iref, mode, eps)
            chi2 = jnp.minimum(chi2, 1e4)
            data = 0.5 * jnp.sum(w * (chi2 + lndet)) / wsum
            prior = 0.0
            for k, v in params.items():
                d2 = _second_diff(v) / PRIOR_SCALE[k]
                prior = prior + 0.5 * jnp.sum(d2**2)
            return data + smooth * prior / wsum

        opt = optax.lbfgs()
        value_and_grad = optax.value_and_grad_from_state(loss)

        @jax.jit
        def step(params, state):
            value, grad = value_and_grad(params, state=state)
            updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=loss)
            return optax.apply_updates(params, updates), state, value

        params, state = params0, opt.init(params0)
        prev = np.inf
        history = []
        for it in range(maxiter):
            params, state, value = step(params, state)
            value = float(value)
            history.append(value)
            if abs(prev - value) < tol * max(1.0, abs(value)):
                break
            prev = value
        rs = dataclasses.replace(rs64, **params)
        rs = jax.tree_util.tree_map(lambda a: np.asarray(a, np.float64), rs)
    log.info("red-sequence fit: %d galaxies (sum w = %.1f), %d iterations, loss %.5f -> %.5f",
             int(np.size(w)), float(wsum), len(history), history[0], history[-1])
    rs = dataclasses.replace(rs, **{k: jnp.asarray(getattr(rs, k)) for k in
                                    ("mean", "slope", "log_sigma", "corr", "pivot")})
    return rs, {"loss": np.asarray(history), "niter": len(history)}


def update_pivot(rs: RSModel, refmag, z, w, halfwidth: float = 0.05, min_weight: float = 20.0) -> RSModel:
    """Pivot nodes = weighted median reference magnitude of the members near each node."""
    zp = np.asarray(rs.z_pivot)
    piv = np.asarray(rs.pivot, np.float64).copy()
    refmag, z, w = (np.asarray(a, np.float64) for a in (refmag, z, w))
    for k, z0 in enumerate(zp):
        sel = np.abs(z - z0) < halfwidth
        if w[sel].sum() < min_weight:
            continue
        o = np.argsort(refmag[sel])
        cw = np.cumsum(w[sel][o])
        piv[k] = refmag[sel][o][np.searchsorted(cw, 0.5 * cw[-1])]
    return dataclasses.replace(rs, pivot=jnp.asarray(piv))
