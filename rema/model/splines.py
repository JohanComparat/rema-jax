"""Natural cubic splines as linear operators on the node values.

redMaPPer describes every red-sequence quantity (mean colour, slope, scatter, pivot magnitude,
corrections) by its values at a few redshift nodes joined by a natural cubic spline. Here the
spline is written as ``s(z) = W(z) @ y`` with a basis ``W`` that depends only on the node
positions, so fits are linear in the node values ``y`` and the model stays differentiable in
both ``y`` and ``z``.

Outside the nodes the spline continues linearly with the end slope (a natural spline has zero
curvature at its ends), instead of extending the end cubic.

>>> import numpy as np
>>> sp = NaturalSpline([0.1, 0.3, 0.5, 0.7])
>>> y = np.array([1.0, 1.5, 1.8, 2.4])
>>> np.allclose(sp(np.array([0.1, 0.3, 0.5, 0.7]), y), y)
True
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


class NaturalSpline:
    """Natural cubic spline through fixed nodes, linear in the node values.

    Parameters
    ----------
    nodes : strictly increasing node positions (at least one).
    """

    def __init__(self, nodes):
        x = np.asarray(nodes, dtype=np.float64)
        if x.ndim != 1 or x.size < 1:
            raise ValueError("nodes must be a non-empty 1-d array")
        if x.size > 1 and np.any(np.diff(x) <= 0):
            raise ValueError("nodes must be strictly increasing")
        self.nodes = x
        self.n = x.size
        self._G = self._second_derivative_operator(x)

    @staticmethod
    def _second_derivative_operator(x: np.ndarray) -> np.ndarray:
        """Matrix G with M = G @ y, the spline second derivatives at the nodes (M_0 = M_-1 = 0)."""
        n = x.size
        G = np.zeros((n, n))
        if n < 3:
            return G
        h = np.diff(x)
        m = n - 2
        A = np.zeros((m, m))
        R = np.zeros((m, n))
        for k in range(m):
            i = k + 1
            A[k, k] = 2.0 * (h[i - 1] + h[i])
            if k > 0:
                A[k, k - 1] = h[i - 1]
            if k < m - 1:
                A[k, k + 1] = h[i]
            R[k, i - 1] = 6.0 / h[i - 1]
            R[k, i] = -6.0 / h[i - 1] - 6.0 / h[i]
            R[k, i + 1] = 6.0 / h[i]
        G[1:-1] = np.linalg.solve(A, R)
        return G

    # ------------------------------------------------------------------ basis
    def basis(self, z) -> np.ndarray:
        """Basis matrix W with ``s(z) = W @ y``; shape ``z.shape + (n,)`` (numpy, float64)."""
        z = np.asarray(z, dtype=np.float64)
        flat = z.reshape(-1)
        W = np.zeros((flat.size, self.n))
        if self.n == 1:
            W[:, 0] = 1.0
            return W.reshape(z.shape + (self.n,))
        x, G = self.nodes, self._G
        i = np.clip(np.searchsorted(x, flat, side="right") - 1, 0, self.n - 2)
        h = x[i + 1] - x[i]
        b = (flat - x[i]) / h
        a = 1.0 - b
        inside = (flat >= x[0]) & (flat <= x[-1])
        rows = np.arange(flat.size)
        # Interior: cubic Hermite form of the natural spline.
        ci = np.where(inside, (a**3 - a) * h**2 / 6.0, 0.0)
        cj = np.where(inside, (b**3 - b) * h**2 / 6.0, 0.0)
        np.add.at(W, (rows, i), np.where(inside, a, 0.0))
        np.add.at(W, (rows, i + 1), np.where(inside, b, 0.0))
        W += ci[:, None] * G[i] + cj[:, None] * G[i + 1]
        # Below the first node: y0 + s'(x0) (z - x0), s'(x0) = (y1-y0)/h0 - h0 M1 / 6.
        lo = flat < x[0]
        if np.any(lo):
            h0 = x[1] - x[0]
            dz = (flat[lo] - x[0])[:, None]
            slope = np.zeros(self.n)
            slope[0] -= 1.0 / h0
            slope[1] += 1.0 / h0
            slope = slope - h0 / 6.0 * G[1]
            e0 = np.zeros(self.n)
            e0[0] = 1.0
            W[lo] = e0 + dz * slope
        # Above the last node: y_{n-1} + s'(x_{n-1}) (z - x_{n-1}), s' = (y_{n-1}-y_{n-2})/h + h M_{n-2}/6.
        hi = flat > x[-1]
        if np.any(hi):
            hl = x[-1] - x[-2]
            dz = (flat[hi] - x[-1])[:, None]
            slope = np.zeros(self.n)
            slope[-1] += 1.0 / hl
            slope[-2] -= 1.0 / hl
            slope = slope + hl / 6.0 * G[-2]
            el = np.zeros(self.n)
            el[-1] = 1.0
            W[hi] = el + dz * slope
        return W.reshape(z.shape + (self.n,))

    def __call__(self, z, y):
        """Evaluate with numpy (host)."""
        return self.basis(z) @ np.asarray(y, dtype=np.float64)

    # ------------------------------------------------------------------ JAX
    def eval_jax(self, z, y):
        """Evaluate at traced ``z`` with node values ``y`` (both may be JAX arrays).

        ``y`` may carry trailing dimensions: shape ``(n, ...)`` gives ``z.shape + (...)``.
        """
        y = jnp.asarray(y)
        z = jnp.asarray(z)
        if self.n == 1:
            return jnp.broadcast_to(y[0], z.shape + y.shape[1:])
        x = jnp.asarray(self.nodes)
        M = jnp.tensordot(jnp.asarray(self._G, dtype=y.dtype), y, axes=1,
                          precision=jax.lax.Precision.HIGHEST)
        i = jnp.clip(jnp.searchsorted(x, z, side="right") - 1, 0, self.n - 2)
        h = x[i + 1] - x[i]
        b = (z - x[i]) / h
        a = 1.0 - b
        ex = (Ellipsis,) + (None,) * (y.ndim - 1)
        yi, yj, Mi, Mj = y[i], y[i + 1], M[i], M[i + 1]
        inner = (a[ex] * yi + b[ex] * yj
                 + (h**2 / 6.0)[ex] * ((a**3 - a)[ex] * Mi + (b**3 - b)[ex] * Mj))
        h0 = x[1] - x[0]
        hl = x[-1] - x[-2]
        slope_lo = (y[1] - y[0]) / h0 - h0 * M[1] / 6.0
        slope_hi = (y[-1] - y[-2]) / hl + hl * M[-2] / 6.0
        below = y[0] + (z - x[0])[ex] * slope_lo
        above = y[-1] + (z - x[-1])[ex] * slope_hi
        out = jnp.where((z < x[0])[ex], below, inner)
        return jnp.where((z > x[-1])[ex], above, out)
