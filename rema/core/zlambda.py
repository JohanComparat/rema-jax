"""Cluster photometric redshift z_lambda.

redMaPPer's iteration (``Zlambda.calc_zlambda``): at the current z, compute the richness and the
colour membership probabilities pcol; keep a soft top fraction of the members,
pw_i = 1 / (exp((p_thresh - pcol_i)/0.04) + 1) with p_thresh the n-th largest pcol,
n = max(3, topfrac * sum pcol); maximise L(z) = sum_i pw_i ln L_i(z) over a local grid around z
(parabola through the three best points); repeat until |dz| < tol (at most ``maxiter`` steps,
with a convergence mask so the batch stays vectorised).

The error and the redshift distribution come from p(z) proportional to exp(L(z)) V(z) on 21 bins
spanning +-4 sigma (sigma from the curvature of L, clipped to [0.005, 0.1]):
z_lambda_e is the standard deviation of p(z).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from ..model.likelihood import chisq as chisq_fn
from ..model.redsequence import RSAt
from .context import FilterModel
from .richness import RadialQuad, Richness, Stage, grid_chunk, photoz_chisq, richness_one

ZGRID_BUDGET = 7 * 2048     # redshift-grid points x neighbours per vectorised step


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ZLambda:
    z: jnp.ndarray            # [B]
    z_e: jnp.ndarray
    niter: jnp.ndarray
    pzbins: jnp.ndarray       # [B, npz]
    pz: jnp.ndarray           # [B, npz]
    rich: Richness            # richness at z


def _member_lnl_one(nb, z, model: FilterModel):
    """ln L_i(z) = -chi^2/2 - ln det/2 for every neighbour at one redshift: [K].

    With the photo-z filter, chi^2 = ((ZPHOT - z)/s)^2 and ln det = 2 ln s: ln p_i(z).
    """
    if model.filter == "photoz":
        chi2, lndet, _ = photoz_chisq(nb, z)
        return -0.5 * chi2 - 0.5 * lndet
    rsz = model.rs.at(z)
    at = RSAt(rsz.mean[None], rsz.slope[None], rsz.cint[None], rsz.pivot[None])
    chi2, lndet = chisq_fn(nb.flux, nb.ivar, at, model.iref, model.chisq_mode, model.eps)
    return -0.5 * chi2 - 0.5 * lndet


def _weighted_lnl(nb, pw, zgrid, model: FilterModel, chunk: int = 7):
    """L(z) = sum_i pw_i ln L_i(z) on a grid, up to ``chunk`` grid points at a time.

    Memory is O(chunk K); for large K fewer points are evaluated together (see ``grid_chunk``).
    """
    w = jnp.where(nb.valid, pw, 0.0)
    one = lambda z: jnp.sum(w * _member_lnl_one(nb, z, model))
    g = zgrid.shape[0]
    c = grid_chunk(g, w.shape[-1], ZGRID_BUDGET, chunk)
    if c == 1:
        return jax.lax.map(one, zgrid)
    return jax.lax.map(jax.vmap(one), zgrid.reshape(g // c, c)).reshape(g)


def _weights(pcol, topfrac: float, soft: float):
    n = jnp.maximum(3.0, topfrac * jnp.sum(pcol))
    srt = jnp.sort(pcol)[::-1]
    k = jnp.clip(jnp.floor(n).astype(jnp.int32) - 1, 0, pcol.shape[0] - 1)
    pth = srt[k]
    return jnp.where(pcol > 0, 1.0 / (jnp.exp((pth - pcol) / soft) + 1.0), 0.0)


def _parabola_vertex(zg, L):
    """Vertex of the parabola through the best grid point and its neighbours."""
    g = zg.shape[0]
    i = jnp.clip(jnp.argmax(L), 1, g - 2)
    y0, y1, y2 = L[i - 1], L[i], L[i + 1]
    h = zg[1] - zg[0]
    a = 0.5 * (y0 - 2 * y1 + y2) / (h * h)
    b = 0.5 * (y2 - y0) / h
    ok = (a < 0) & jnp.isfinite(a) & jnp.isfinite(b)
    zv = zg[i] - b / (2.0 * jnp.where(ok, a, -1.0))
    zv = jnp.where(ok, jnp.clip(zv, zg[i] - h, zg[i] + h), zg[i])
    return zv, jnp.where(ok, a, -1.0 / (2.0 * 0.05**2))


def zlambda_one(nb, z0, frad, fgeo, quad: RadialQuad, model: FilterModel, stage: Stage,
                maxiter: int = 5, tol: float = 2e-4, ngrid: int = 21, half_width: float = 0.03,
                npz: int = 21, topfrac: float = 0.7, soft: float = 0.04, zmin: float = 0.01,
                zmax: float = 1.5, calc_err: bool = True):
    """z_lambda of one cluster starting from z0 (without z_e and p(z) unless ``calc_err``)."""
    offs = jnp.linspace(-half_width, half_width, ngrid)

    def body(carry):
        z, done, it = carry
        rich = richness_one(nb, z, frad, fgeo, quad, model, stage)
        pw = _weights(rich.pcol, topfrac, soft)
        zg = jnp.clip(z + offs, zmin, zmax)
        L = _weighted_lnl(nb, pw, zg, model)
        zn, _ = _parabola_vertex(zg, L)
        ok = (rich.lam > 0) & jnp.any(pw > 1e-3)
        zn = jnp.where(ok, zn, z)
        done = (jnp.abs(zn - z) < tol) | ~ok
        return zn, done, it + 1

    def cond(carry):
        _, done, it = carry
        return (~done) & (it < maxiter)

    # Iterate until converged (stops early; under vmap until every cluster has converged).
    z, done, niter = jax.lax.while_loop(cond, body, (z0, jnp.bool_(False), jnp.int32(0)))
    rich = richness_one(nb, z, frad, fgeo, quad, model, stage)
    good = rich.lam > 0
    if not calc_err:
        none = jnp.full((npz,), -1.0, z.dtype)
        return ZLambda(z=jnp.where(good, z, -1.0), z_e=jnp.full_like(z, -1.0), niter=niter,
                       pzbins=none, pz=none, rich=rich)
    pw = _weights(rich.pcol, topfrac, soft)
    # Curvature at z for the p(z) grid width.
    zg = jnp.clip(z + offs * 0.2, zmin, zmax)
    L = _weighted_lnl(nb, pw, zg, model)
    _, a = _parabola_vertex(zg, L)
    sig = jnp.clip(jnp.sqrt(-0.5 / a), 0.005, 0.1)
    pzb = jnp.clip(z + jnp.linspace(-4.0, 4.0, npz) * sig, zmin, zmax)
    Lp = _weighted_lnl(nb, pw, pzb, model)
    lv = jnp.log(model.cosmo.volume_factor(pzb, 1.0))
    lp = Lp + lv
    pz = jnp.exp(lp - jnp.max(lp))
    pz = pz / jnp.maximum(jnp.trapezoid(pz, pzb), 1e-30)
    mean = jnp.trapezoid(pz * pzb, pzb)
    var = jnp.trapezoid(pz * (pzb - mean) ** 2, pzb)
    z_e = jnp.sqrt(jnp.maximum(var, 1e-10))
    return ZLambda(z=jnp.where(good, z, -1.0), z_e=jnp.where(good, z_e, -1.0), niter=niter,
                   pzbins=pzb, pz=pz, rich=rich)


@partial(jax.jit, static_argnames=("maxiter", "ngrid", "npz", "calc_err"))
def zlambda(nb, z0, frad, fgeo, quad: RadialQuad, model: FilterModel, stage: Stage,
            maxiter: int = 5, tol: float = 2e-4, ngrid: int = 21, half_width: float = 0.03,
            npz: int = 21, topfrac: float = 0.7, soft: float = 0.04,
            calc_err: bool = True) -> ZLambda:
    """z_lambda of a batch of clusters (arrays [B, K]; z0 [B]).

    ``calc_err=False`` (first pass, as in redMaPPer) skips the curvature and p(z) grids;
    z_e, pzbins and pz are then -1.
    """
    fn = partial(zlambda_one, quad=quad, model=model, stage=stage, maxiter=maxiter, tol=tol,
                 ngrid=ngrid, half_width=half_width, npz=npz, topfrac=topfrac, soft=soft,
                 calc_err=calc_err)
    return jax.vmap(fn)(nb, z0, frad, fgeo)
