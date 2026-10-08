"""Richness lambda of a cluster: the matched-filter membership solve.

For a cluster at redshift z and a neighbour i at projected radius r_i (h^-1 Mpc), reference
magnitude m_i and chi^2_i (against the red sequence at z):

    u_i = 2 pi r_i Sigma_NFW(r_i) phi(m_i)/lumnorm rho_chi2(chi^2_i)      (cluster filter)
    b_i = 2 pi r_i Sigma_g(z, chi^2_i, m_i) / D(z)^2                        (background)
    p_i(lambda) = lambda N(r_lambda) u_i / (lambda N(r_lambda) u_i + b_i)
    pmem_i = p_i theta_i pfree_i theta_r(r_i; r_lambda),   r_lambda = r0 (lambda/100)^beta

and lambda solves  sum_i pmem_i(lambda) = lambda K(r_lambda),  where K is the expected fraction of
the cluster's galaxies (within r_lambda, brighter than 0.2 L*) that are observable given the
mask and depth (1 without a footprint; redMaPPer writes K = 1 - C). The solve is a bisection
on a log-spaced grid of lambda values followed by Newton steps, wrapped in ``jax.lax.custom_root`` so that lambda has
exact implicit derivatives with respect to every model parameter.

With the photo-z filter (``model.filter = "photoz"``) the colour term rho_chi2(chi^2_i) becomes
the galaxy's photo-z distribution at the cluster redshift, p_i(z) = N(z; ZPHOT_i, s_i), and the
background the stacked photo-z distributions Sigma_pz(z, m_i) per unit z (as AMICO, Bellagamba
et al. 2018); chi^2_i is then x_i^2 = ((ZPHOT_i - z)/s_i)^2, cut at ``pz_nsig_max``.

Outputs follow redMaPPer: LAMBDA, LAMBDA_E = sqrt((1 - <p>) lambda S) with
<p> = sum pmem^2 / sum pmem and S = SCALEVAL = lambda / sum pmem, R_LAMBDA, MASKFRAC
(1 - geometric completeness within r_lambda), LNLAMLIKE = -sum pmem - sum ln(1 - pmem) over
members other than the central galaxy, and per neighbour P, PMEM, PCOL, THETA_I, THETA_R, CHISQ.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from ..model import profiles as prof
from ..model.likelihood import chisq as chisq_fn
from ..model.redsequence import RSAt
from .context import FilterModel


# --------------------------------------------------------------------------- inputs
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Neighbors:
    """Padded neighbour arrays of a batch of clusters ([B, K] unless noted)."""

    theta: jnp.ndarray        # angular separation from the centre [deg]
    refmag: jnp.ndarray
    refmag_err: jnp.ndarray
    flux: jnp.ndarray         # [B, K, nband] (model bands)
    ivar: jnp.ndarray         # [B, K, nband]
    zred: jnp.ndarray
    zred_e: jnp.ndarray
    pfree: jnp.ndarray
    valid: jnp.ndarray        # bool
    is_center: jnp.ndarray    # bool, the galaxy at (or nearest) the centre
    zphot: jnp.ndarray | None = None     # photo-z filter: ZPHOT and its calibrated width s
    zphot_e: jnp.ndarray | None = None   # (-1 without a usable photo-z)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Stage:
    """Aperture of a richness stage (traced, so stages share one compilation)."""

    r0: jnp.ndarray
    beta: jnp.ndarray
    maxrad: jnp.ndarray       # neighbours beyond this radius (h^-1 Mpc) are ignored

    @classmethod
    def make(cls, r0: float, beta: float, maxrad_factor: float = 1.2) -> "Stage":
        maxrad = maxrad_factor * r0 * 3.0**beta if beta > 0 else maxrad_factor * r0
        return cls(jnp.asarray(float(r0)), jnp.asarray(float(beta)), jnp.asarray(float(maxrad)))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RadialQuad:
    """Gauss-Legendre nodes in projected radius for the completeness integral."""

    r: jnp.ndarray            # [n]
    w: jnp.ndarray            # [n]

    @classmethod
    def make(cls, rmax: float = 2.3, rcore: float = 0.1, n_core: int = 4, n_outer: int = 20) -> "RadialQuad":
        r, w = _gl_split(float(rmax), float(rcore), int(n_core), int(n_outer))
        return cls(jnp.asarray(r), jnp.asarray(w))


@lru_cache(maxsize=16)
def _gl_split(rmax, rcore, n_core, n_outer):
    xc, wc = np.polynomial.legendre.leggauss(n_core)
    xo, wo = np.polynomial.legendre.leggauss(n_outer)
    r = np.concatenate([0.5 * rcore * (xc + 1), rcore + 0.5 * (rmax - rcore) * (xo + 1)])
    w = np.concatenate([0.5 * rcore * wc, 0.5 * (rmax - rcore) * wo])
    return r, w


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Richness:
    lam: jnp.ndarray          # [B]   (-1 when lambda < 1)
    lam_e: jnp.ndarray
    r_lambda: jnp.ndarray
    scaleval: jnp.ndarray
    maskfrac: jnp.ndarray
    lnlamlike: jnp.ndarray
    p: jnp.ndarray            # [B, K]
    pmem: jnp.ndarray
    pcol: jnp.ndarray
    theta_i: jnp.ndarray
    theta_r: jnp.ndarray
    chisq: jnp.ndarray
    r: jnp.ndarray            # projected radius [h^-1 Mpc]


# --------------------------------------------------------------------------- kernel
BAD_CHISQ = 1e6             # chi^2 of a galaxy without a usable photo-z (finite: no 0 * inf)
SQRT2PI = float(np.sqrt(2.0 * np.pi))


def photoz_chisq(nb, z):
    """(x^2, 2 ln s, usable) of the neighbours' photo-z at the cluster redshift z ([K])."""
    usable = (nb.zphot >= 0) & (nb.zphot_e > 0)
    s = jnp.where(usable, nb.zphot_e, 1.0)
    x = (nb.zphot - z) / s
    return jnp.where(usable, x * x, BAD_CHISQ), 2.0 * jnp.log(s), usable


def _filter_terms(nb, z, model: FilterModel, stage: Stage):
    """lambda-independent per-neighbour terms for one cluster (arrays [K])."""
    if model.filter == "photoz":
        return _terms_pz(nb, z, model, stage)
    return _terms_rs(nb, z, model, stage)


def _terms_pz(nb, z, model: FilterModel, stage: Stage):
    """:func:`_filter_terms` of the photo-z filter: rho = p_i(z), Sigma_g = Sigma_pz(z, m)."""
    D = model.mpc_per_deg(z)
    r = jnp.maximum(nb.theta * D, 1e-6)
    mstar = model.mstar(z)
    maxmag = model.maxmag(z)
    chi2, lndet, usable = photoz_chisq(nb, z)
    sg = model.pzbkg.lookup(z, nb.refmag)
    ok = (nb.valid & usable & (chi2 < model.pz_nsig_max**2) & (r < stage.maxrad)
          & (nb.refmag < model.pz_mag_max) & jnp.isfinite(sg) & (sg > 0))
    rho = jnp.where(ok, jnp.exp(-0.5 * chi2 - 0.5 * lndet) / SQRT2PI, 0.0)
    sg = jnp.where(ok, sg, 1.0)
    phi = prof.schechter(nb.refmag, mstar, model.alpha) / prof.lumnorm(mstar, maxmag, model.alpha)
    thi = prof.theta_i(nb.refmag, maxmag, nb.refmag_err)
    b = 2.0 * jnp.pi * r * sg / (D * D)
    u0 = jnp.where(ok, 2.0 * jnp.pi * r * prof.nfw_sigma(r, model.nfw_rs, model.nfw_rcore) * phi * rho, 0.0)
    w = jnp.where(ok, thi * nb.pfree, 0.0)
    return dict(r=r, u0=u0, b=b, w=w, chi2=chi2, lndet=lndet, rho=rho, phi=phi, sg=sg, thi=thi,
                D=D, maxmag=maxmag, ok=ok)


def _terms_rs(nb, z, model: FilterModel, stage: Stage):
    """:func:`_filter_terms` of the red-sequence filter."""
    D = model.mpc_per_deg(z)
    r = jnp.maximum(nb.theta * D, 1e-6)
    mstar = model.mstar(z)
    maxmag = model.maxmag(z)
    rsz = model.rs.at(z)
    at = RSAt(rsz.mean[None], rsz.slope[None], rsz.cint[None], rsz.pivot[None])
    chi2, lndet = chisq_fn(nb.flux, nb.ivar, at, model.iref, model.chisq_mode, model.eps)
    ok = nb.valid & (chi2 < model.chisq_max) & (r < stage.maxrad) & jnp.isfinite(chi2)
    rho = prof.chisq_pdf(jnp.where(ok, chi2, 1.0), model.ncol)
    phi = prof.schechter(nb.refmag, mstar, model.alpha) / prof.lumnorm(mstar, maxmag, model.alpha)
    thi = prof.theta_i(nb.refmag, maxmag, nb.refmag_err)
    sg = model.bkg.lookup(z, jnp.where(ok, chi2, 0.0), nb.refmag)
    b = 2.0 * jnp.pi * r * sg / (D * D)
    u0 = jnp.where(ok, 2.0 * jnp.pi * r * prof.nfw_sigma(r, model.nfw_rs, model.nfw_rcore) * phi * rho, 0.0)
    w = jnp.where(ok, thi * nb.pfree, 0.0)
    return dict(r=r, u0=u0, b=b, w=w, chi2=chi2, lndet=lndet, rho=rho, phi=phi, sg=sg, thi=thi,
                D=D, maxmag=maxmag, ok=ok)


def _completeness(rl, frad, quad: RadialQuad, model: FilterModel):
    """K(r_lambda): profile-weighted mean of the radial completeness ``frad`` within r_lambda."""
    prof_w = quad.w * 2.0 * jnp.pi * quad.r * prof.nfw_sigma(quad.r, model.nfw_rs, model.nfw_rcore)
    thr = prof.theta_r(quad.r, rl, model.rsig)
    return jnp.sum(prof_w * thr * frad) / jnp.sum(prof_w * thr)


def grid_chunk(n: int, k: int, budget: int, cmax: int | None = None) -> int:
    """Number of grid points to evaluate together over K = ``k`` neighbours.

    The largest divisor c of n (at most ``cmax``) with c * k <= ``budget``, or 1. XLA's GPU
    reduction emitter unrolls (grid points x K / 256) evaluations, times a factor from the
    batch axes, into every thread, and ptxas time grows faster than linearly with that count.
    32 lambda-grid points at K = 8192 (1024 erf evaluations per thread) take ptxas several
    minutes.
    """
    c = min(n, cmax or n)
    while c > 1 and (c * k > budget or n % c):
        c -= 1
    return c


def _sum_pmem(lam, t, stage, model):
    rl = stage.r0 * (lam / 100.0) ** stage.beta
    N = prof.nfw_norm(rl, model.nfw_rs, model.nfw_rcore)
    thr = prof.theta_r(t["r"], rl, model.rsig)
    lnu = lam * N * t["u0"]
    p = jnp.where(t["u0"] > 0, lnu / (lnu + t["b"]), 0.0)
    return jnp.sum(p * t["w"] * thr), p, thr, rl


def _solve_lambda(t, frad, quad, stage, model, n_grid: int, n_newton: int, lam_lo: float, lam_hi: float):
    """Root of f(x) = ln sum pmem - x - ln K, x = ln lambda, with exact implicit derivatives.

    f is evaluated on ``n_grid`` log-spaced values of lambda; the root is bracketed by the last
    grid point with f > 0 and refined by ``n_newton`` safeguarded Newton steps.
    """
    def f(x):
        lam = jnp.exp(x)
        s, _, _, rl = _sum_pmem(lam, t, stage, model)
        return jnp.log(s + 1e-30) - x - jnp.log(_completeness(rl, frad, quad, model))

    def solve(fun, x0):
        xs = jnp.linspace(jnp.log(lam_lo), jnp.log(lam_hi), n_grid)
        # One grid point per step: a vectorised (grid x K) reduction is unrolled by XLA's GPU
        # emitter into kernels that take ptxas minutes (see grid_chunk); n_grid sequential
        # reductions cost ~1% of a z_lambda call.
        fs = jax.lax.map(fun, xs)
        pos = fs > 0
        # last grid point with f > 0 (0 if none); its right neighbour brackets the root
        k = jnp.clip(jnp.max(jnp.where(pos, jnp.arange(n_grid), 0)), 0, n_grid - 2)
        lo, hi = xs[k], xs[k + 1]
        flo, fhi = fs[k], fs[k + 1]
        x = jnp.where(jnp.isfinite(flo) & jnp.isfinite(fhi) & (flo != fhi),
                      lo - flo * (hi - lo) / (fhi - flo), 0.5 * (lo + hi))    # secant start
        x = jnp.clip(x, lo, hi)
        dfun = jax.grad(fun)
        for _ in range(n_newton):
            fx = fun(x)
            step = fx / dfun(x)
            xn = x - jnp.where(jnp.isfinite(step), step, 0.0)
            # stay inside the (closed) bracket; fall back to the midpoint if Newton leaves it
            x = jnp.where((xn >= lo) & (xn <= hi), xn, 0.5 * (lo + hi))
            fn = fun(x)
            lo = jnp.where(fn > 0, x, lo)
            hi = jnp.where(fn < 0, x, hi)
        return x

    def tangent_solve(g, y):
        return y / g(1.0)

    x = jax.lax.custom_root(f, jnp.log(10.0), solve, tangent_solve)
    return jnp.exp(x), f(jnp.log(lam_lo))


def richness_one(nb, z, frad, fgeo, quad: RadialQuad, model: FilterModel, stage: Stage,
                 n_grid: int = 32, n_newton: int = 4, lam_lo: float = 0.5, lam_hi: float = 2000.0):
    """Richness of one cluster (unbatched arrays); see :func:`richness`."""
    t = _filter_terms(nb, z, model, stage)
    lam, f_lo = _solve_lambda(t, frad, quad, stage, model, n_grid, n_newton, lam_lo, lam_hi)
    s, p, thr, rl = _sum_pmem(lam, t, stage, model)
    pmem = p * t["w"] * thr
    maskfrac = 1.0 - _completeness(rl, fgeo, quad, model)
    scaleval = lam / jnp.maximum(s, 1e-30)
    pbar = jnp.sum(pmem**2) / jnp.maximum(s, 1e-30)
    lam_e = jnp.sqrt(jnp.maximum((1.0 - pbar) * lam * scaleval, 0.0))
    # Colour-luminosity membership, ignoring the radial position (used for z_lambda).
    lamphirho = lam * t["phi"] * t["rho"]
    bkg_area = jnp.pi * rl**2 * t["sg"] / (t["D"] ** 2)
    pcol = jnp.where(t["ok"] & (t["r"] < rl) & (nb.refmag < t["maxmag"]),
                     lamphirho / (lamphirho + bkg_area), 0.0)
    incut = (pmem > 0) & ~nb.is_center
    lnlamlike = -s - jnp.sum(jnp.where(incut, jnp.log1p(-jnp.minimum(pmem, 1 - 1e-6)), 0.0))
    good = (f_lo > 0) & (lam >= 1.0) & jnp.isfinite(lam)
    return Richness(lam=jnp.where(good, lam, -1.0), lam_e=jnp.where(good, lam_e, -1.0),
                    r_lambda=jnp.where(good, rl, -1.0), scaleval=jnp.where(good, scaleval, -1.0),
                    maskfrac=maskfrac, lnlamlike=jnp.where(good, lnlamlike, -jnp.inf),
                    p=p, pmem=jnp.where(good, pmem, 0.0), pcol=jnp.where(good, pcol, 0.0),
                    theta_i=t["thi"], theta_r=thr, chisq=t["chi2"], r=t["r"])


@partial(jax.jit, static_argnames=("n_grid", "n_newton", "lam_lo", "lam_hi"))
def richness(nb: Neighbors, z, frad, fgeo, quad: RadialQuad, model: FilterModel, stage: Stage,
             n_grid: int = 32, n_newton: int = 4, lam_lo: float = 0.5,
             lam_hi: float = 2000.0) -> Richness:
    """Richness of a batch of clusters.

    Parameters
    ----------
    nb : neighbours, arrays [B, K].
    z : cluster redshifts [B].
    frad : radial completeness (mask x selection) on ``quad`` nodes [B, n]; ones without a footprint.
    fgeo : geometric (mask only) radial completeness [B, n], for MASKFRAC.
    """
    fn = partial(richness_one, quad=quad, model=model, stage=stage, n_grid=n_grid,
                 n_newton=n_newton, lam_lo=lam_lo, lam_hi=lam_hi)
    return jax.vmap(fn)(nb, z, frad, fgeo)
