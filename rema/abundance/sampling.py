"""Best fit, Laplace approximation and NUTS sampling of the counts likelihood.

``nuts`` needs blackjax (the ``cosmo`` extra): the parameters are mapped from the real line onto
the prior box by a logistic function, whose log-Jacobian is added to the log-posterior.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from ..model.cosmo import _float64
from .likelihood import Likelihood


def map_fit(lik: Likelihood, x0=None, maxiter: int = 1000, tol: float = 1e-12) -> tuple[np.ndarray, dict]:
    """Maximum of the log-posterior (L-BFGS-B in coordinates scaled to the prior box)."""
    from scipy.optimize import minimize

    lo, hi = (np.array(b, np.float64) for b in zip(*lik.bounds()))
    span = hi - lo
    x0 = lik.x0() if x0 is None else np.asarray(x0, np.float64)
    u0 = np.clip((x0 - lo) / span, 0.0, 1.0)
    t0 = time.time()
    def f(u):
        v, g = lik.value_and_grad(lo + span * u)
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            # Outside the emulator box: a high, flat value makes the line search shrink its step
            # (a slope back to the start would send it there).
            return 1e10, np.zeros_like(u)
        return -v, -g * span

    res = minimize(f, u0, jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * u0.size,
                   options={"maxiter": maxiter, "ftol": tol, "gtol": 1e-8, "maxcor": 30})
    x = lo + span * res.x
    return x, {"logpost": -float(res.fun), "success": bool(res.success), "nit": int(res.nit),
               "message": str(res.message), "time": time.time() - t0,
               "at_bound": [n for n, u in zip(lik.free, res.x) if u < 1e-6 or u > 1 - 1e-6]}


def laplace(lik: Likelihood, x) -> np.ndarray:
    """Covariance of the parameters from the Hessian of the log-posterior at ``x``."""
    with _float64():
        H = np.asarray(jax.jit(jax.hessian(lik._logpost_impl))(jnp.asarray(x, jnp.float64)))
    return np.linalg.inv(-H)


def nuts(lik: Likelihood, x0=None, warmup: int = 500, samples: int = 1000, seed: int = 0,
         progress: bool = False) -> dict:
    """NUTS chain (blackjax window adaptation, then sampling) of the free parameters.

    Returns {"samples": [samples, n_free], "logpost": [samples], "accept": mean acceptance,
    "names": free names}.
    """
    import blackjax

    lo, hi = (np.array(b, np.float64) for b in zip(*lik.bounds()))

    def to_x(u):
        return lo_j + span * jax.nn.sigmoid(u)

    def logdensity(u):
        s = jax.nn.sigmoid(u)
        logjac = jnp.sum(jnp.log(span) + jnp.log(s) + jnp.log1p(-s))
        return lik._logpost_impl(to_x(u)) + logjac

    x0 = lik.x0() if x0 is None else np.asarray(x0, np.float64)
    p = np.clip((x0 - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    with _float64():
        u0 = jnp.asarray(np.log(p / (1 - p)), jnp.float64)
        lo_j, span = jnp.asarray(lo, jnp.float64), jnp.asarray(hi - lo, jnp.float64)
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        adapt = blackjax.window_adaptation(blackjax.nuts, logdensity)
        (state, params), _ = adapt.run(k1, u0, num_steps=warmup)
        kernel = jax.jit(blackjax.nuts(logdensity, **params).step)
        keys = jax.random.split(k2, samples)
        us, lps, acc = [], [], []
        for i, k in enumerate(keys):
            state, info = kernel(k, state)
            us.append(np.asarray(state.position))
            lps.append(float(state.logdensity))
            acc.append(float(info.acceptance_rate))
            if progress and i % 100 == 0:
                print(f"nuts {i}/{samples}", flush=True)
        xs = np.asarray(to_x(jnp.asarray(np.array(us))))
    return {"samples": xs, "logpost": np.array(lps), "accept": float(np.mean(acc)),
            "names": lik.free, "step_size": float(params["step_size"])}
