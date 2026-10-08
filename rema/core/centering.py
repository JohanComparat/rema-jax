"""Choice of the central galaxy.

``center_bcg`` follows redMaPPer's CenteringBCG: among the neighbours within r_lambda (and, in
scan mode, within ``maxrad`` of the input position), the brightest galaxy that is a likely member
(pmem > ``pmem_min``) or whose zred agrees with the cluster redshift within ``nsig`` zred_e.
P_CEN is 1 for the chosen galaxy; :func:`as_centering` puts the choice in the wcen layout.

``center_wcen`` follows redMaPPer's CenteringWcenZred. The candidates are the neighbours within
r_lambda (and ``maxrad``) with pfree >= ``pbcg_cut``, a zred fit with chi^2 < ``zred_chisq_max``,
and pmem > 0 or |zred - z| < 5 zred_e. The connectivity of a candidate c is

    w_c = ln[ sum_j p_j L_j / d_cj / ((1/r_lambda) sum_j p_j L_j) ]

over the galaxies j != c with p > 0 (raw membership probability) and a softened distance
d_cj = sqrt(D^2 theta_cj^2 + rsoft^2) < r_lambda (w = 1e-3 when there is none), with
L = 10^(0.4 (m* - m)) (1 without ``uselum``) and D in h^-1 Mpc per degree. With N(x; mu, s) a
normalised Gaussian and sigscale = sqrt((min(lambda, maxlambda)/S)/pivot) dividing every ln w
width:

    ucen = N(m; m* + Delta0 + Delta1 ln(lambda/pivot), sigma_m) N(zred; z, zred_e) N(ln w; cen)
    usat = phi(m)/lumnorm N(zred; z, zred_e) N(ln w; sat)
    bcounts = N(ln w; fg) Sigma_g(zred, m) pi r_lambda^2 / D^2
    P_C = min(pfree ucen / (ucen + (lambda/S - 1) usat + bcounts), 0.99999)

with ucen, usat below 1e-10 set to 0 and non-finite P_C set to 0. ln w is the log of w, itself
a log (redMaPPer's convention, and that of the calibration). The ``maxcen`` candidates with the
largest P_C > 0 are kept (NCENT_GOOD of them); with u_i = P_i prod_{j != i} (1 - P_j) and
q0 = prod (1 - P_j) over the kept ones,

    P_CEN = u / sum u,  Q_CEN = u / (q0 + sum u),  Q_MISS = q0 / (q0 + sum u),
    P_SAT, P_FG = (1 - P_CEN) [(lambda - 1) usat, bcounts] / ((lambda - 1) usat + bcounts).

The zred term is centred on z itself, as in redMaPPer, whose percolation and zscan construct the
centring without a z_lambda correction (run_percolation.py:285, run_zscan.py:251).

Deviations from redMaPPer: Q_MISS is stored (redMaPPer computes it and writes 0); unused slots
have index -1 and probabilities 0; no candidate gives NGOOD = 0 (redMaPPer raises); a failed zred
is never a candidate; ties in P_C go to the lower neighbour index. Every richness quantity comes
from the one :class:`~rema.core.richness.Richness` the caller passes (percolation: at z_lambda at
the seed), where redMaPPer mixes lambda, p and pmem at the input z with r and m* at z_lambda.
At most ``ncand`` candidates enter the pairwise w: those with the largest pfree phi_cen gz among
the candidates whose ucen can reach 1e-10 (phi_cen gz / (sqrt(2 pi) sigma) >= 1e-10). This is
exact whenever at most ``ncand`` candidates pass that test; the others get P_C = 0.

:func:`lncglike` is the likelihood-pass central term LNCGLIKE (redMaPPer's lnbcglike). Its zred
term is centred on zrmod(z), redMaPPer's z -> zred_uncorr mapping of the z_lambda correction (the
median zred of centrals at raw z_lambda = z; run_likelihoods.py:213-216), carried by the
:class:`WcenModel` as a table; without one zrmod(z) = z. :func:`w_column` is the catalogue
connectivity W of an accepted cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from ..model import profiles as prof
from .richness import grid_chunk
from .zlambda import ZGRID_BUDGET

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RemaConfig
    from ..model.background import ZredBkg
    from .context import FilterModel

ZRED_NSIG = 5.0             # candidate if |zred - z| < ZRED_NSIG zred_e (or pmem > 0)
UFLOOR = 1e-10              # ucen, usat below this are set to 0
PCEN_CLIP = 0.99999         # upper clip of P_C (keeps the odds finite)
W_FLOOR = 1e-3              # w of a candidate without members within r_lambda
LN_G_FLOOR = float(np.log(1e-10))   # floor of the zred term of LNCGLIKE (ln of 1e-10)
R_MIN_LIKE = 1e-5           # LNCGLIKE: closer galaxies (the central, r = 1e-6) are not members
SQRT2PI = float(np.sqrt(2.0 * np.pi))
WCEN_KEYS = ("DELTA0", "DELTA1", "SIGMA_M", "LNW_CEN_MEAN", "LNW_CEN_SIGMA", "LNW_SAT_MEAN",
             "LNW_SAT_SIGMA", "LNW_FG_MEAN", "LNW_FG_SIGMA")
# zrmod(z) = z without a z -> zred_uncorr mapping: exact for every z, as _interpol extrapolates.
ZRMOD_IDENTITY = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))


def center_bcg_one(nb, pmem, r, r_lambda, z, maxrad, pmem_min: float = 0.8, nsig: float = 2.0,
                   use_zphot: bool = False):
    """(index of the central among the neighbours, found flag) for one cluster.

    The redshift test uses zred, or the photo-z (and its calibrated width) with ``use_zphot``.
    """
    zg, ze = (nb.zphot, nb.zphot_e) if use_zphot else (nb.zred, nb.zred_e)
    zok = jnp.abs(zg - z) < nsig * jnp.maximum(ze, 1e-3)
    cand = nb.valid & (r < jnp.minimum(r_lambda, maxrad)) & ((pmem > pmem_min) | zok)
    score = jnp.where(cand, -nb.refmag, -jnp.inf)
    i = jnp.argmax(score)
    return i, jnp.any(cand)


@partial(jax.jit, static_argnames=("use_zphot",))
def center_bcg(nb, pmem, r, r_lambda, z, maxrad, pmem_min: float = 0.8, nsig: float = 2.0,
               use_zphot: bool = False):
    """Batched :func:`center_bcg_one`: indices [B] and found flags [B]."""
    fn = partial(center_bcg_one, pmem_min=pmem_min, nsig=nsig, use_zphot=use_zphot)
    return jax.vmap(fn, in_axes=(0, 0, 0, 0, 0, None))(nb, pmem, r, r_lambda, z, maxrad)


# --------------------------------------------------------------------------- wcen model
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class WcenModel:
    """Calibrated wcen parameters (0-d float32 leaves), the zred background and the
    z -> zred_uncorr table of LNCGLIKE."""

    zbkg: ZredBkg             # Sigma_g(zred, m) [1/(deg^2 mag zred)]
    delta0: jnp.ndarray       # central magnitude m* + delta0 + delta1 ln(lambda/pivot)
    delta1: jnp.ndarray
    sigma_m: jnp.ndarray
    lnw_cen_mean: jnp.ndarray  # Gaussians in ln w, widths at lambda/S = pivot
    lnw_cen_sigma: jnp.ndarray
    lnw_sat_mean: jnp.ndarray
    lnw_sat_sigma: jnp.ndarray
    lnw_fg_mean: jnp.ndarray
    lnw_fg_sigma: jnp.ndarray
    zrmod_z: jnp.ndarray      # [n] LNCGLIKE: zrmod(z) interpolated linearly in (zrmod_z, zrmod)
    zrmod: jnp.ndarray        # [n] (ZRMOD_IDENTITY without a mapping); unused by the centring
    pivot: float = field(default=30.0, metadata=dict(static=True))
    rsoft: float = field(default=0.05, metadata=dict(static=True))
    maxlambda: float = field(default=100.0, metadata=dict(static=True))
    zred_chisq_max: float = field(default=100.0, metadata=dict(static=True))
    pbcg_cut: float = field(default=0.5, metadata=dict(static=True))
    maxcen: int = field(default=5, metadata=dict(static=True))
    ncand: int = field(default=64, metadata=dict(static=True))
    uselum: bool = field(default=True, metadata=dict(static=True))

    @classmethod
    def create(cls, params: dict, zbkg: ZredBkg, cfg: RemaConfig, zrmod=None) -> "WcenModel":
        """From the calibration's WCEN values (keys as :data:`WCEN_KEYS`, any case; optional
        PIVOT, default ``cfg.centering.wcen_pivot``) and the static settings of
        ``cfg.centering``. ``zrmod``: the (z, zred) table of the z -> zred_uncorr mapping
        (:meth:`rema.calibration.ZlambdaCorrection.zrmod_table`); None for zrmod(z) = z."""
        p = {str(k).upper(): v for k, v in params.items()}
        c = cfg.centering
        pivot = float(p.get("PIVOT", c.wcen_pivot))
        if not np.isfinite(pivot) or pivot <= 0:
            pivot = float(c.wcen_pivot)
        leaves = {k.lower(): jnp.asarray(float(p[k]), jnp.float32) for k in WCEN_KEYS}
        zz, zr = ZRMOD_IDENTITY if zrmod is None else zrmod
        leaves.update(zrmod_z=jnp.asarray(zz, jnp.float32), zrmod=jnp.asarray(zr, jnp.float32))
        return cls(zbkg=zbkg, **leaves, pivot=pivot, rsoft=float(c.wcen_rsoft),
                   maxlambda=float(c.wcen_maxlambda), zred_chisq_max=float(c.wcen_zred_chisq_max),
                   pbcg_cut=float(c.pbcg_cut), maxcen=int(c.maxcen), ncand=int(c.wcen_ncand),
                   uselum=bool(c.wcen_uselum))


def wcen_calibrated(params: dict | None) -> bool:
    """True if ``params`` holds a usable wcen calibration: every key of :data:`WCEN_KEYS`
    finite, and LNW_CEN_SIGMA > 0 and SIGMA_M > 0 (redMaPPer's placeholders are -9999 and 0)."""
    if not params:
        return False
    p = {str(k).upper(): v for k, v in params.items()}
    try:
        vals = {k: float(p[k]) for k in WCEN_KEYS}
    except (KeyError, TypeError, ValueError):
        return False
    return (all(np.isfinite(v) for v in vals.values()) and vals["LNW_CEN_SIGMA"] > 0
            and vals["SIGMA_M"] > 0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Centering:
    """Centre candidates of a batch of clusters, best first ([B, maxcen] unless noted)."""

    index: jnp.ndarray        # int32 neighbour index (into the K axis), -1 in unused slots
    p_cen: jnp.ndarray        # P_CEN (sums to 1 over the kept candidates)
    q_cen: jnp.ndarray        # Q_CEN (sums to 1 - Q_MISS)
    p_sat: jnp.ndarray        # P_SAT
    p_fg: jnp.ndarray         # P_FG
    p_c: jnp.ndarray          # P_C: clipped single-candidate probability (not normalised)
    q_miss: jnp.ndarray       # [B] probability that the centre is none of the candidates
    ngood: jnp.ndarray        # [B] int32 NCENT_GOOD (0: no centre found)
    ncand: jnp.ndarray        # [B] int32 candidates before the ``ncand`` cap


def _gauss(x, mu, s):
    """Normalised Gaussian N(x; mu, s) (redMaPPer's gaussFunction with A = 1/(sqrt(2 pi) s))."""
    return jnp.exp(-0.5 * ((x - mu) / s) ** 2) / (SQRT2PI * s)


def _interpol(v, x, xout):
    """v(x) at ``xout``, linear between the points of the increasing x and extrapolating the end
    segments (redMaPPer's ``interpol``, utilities.py:557-583; jnp.interp would clamp)."""
    s = jnp.clip(jnp.searchsorted(x, xout) - 1, 0, x.shape[0] - 2)
    return (xout - x[s]) * (v[s + 1] - v[s]) / (x[s + 1] - x[s]) + v[s]


def _connectivity(ci, xyz, wt, mem, D, r_lambda, rsoft: float):
    """w of the candidates ``ci`` [kc] (indices into the K neighbours).

    The [candidates x K] sums run over chunks of candidates (``lax.map``) so that no GPU
    reduction over K is vectorised over many candidates (see ``grid_chunk``).
    """
    K = xyz.shape[0]
    kc = ci.shape[0]
    jj = jnp.arange(K)

    def chunk(cc):
        dx = xyz[cc][:, None, :] - xyz[None, :, :]                     # [c, K, 3]
        d = jnp.sqrt(dx[..., 0] ** 2 + dx[..., 1] ** 2 + dx[..., 2] ** 2)
        dist = jnp.rad2deg(2.0 * jnp.arcsin(jnp.clip(0.5 * d, 0.0, 1.0))) * D
        pdis = jnp.sqrt(dist**2 + rsoft**2)
        # Members other than the candidate itself, within r_lambda of it (softened distance).
        a = mem[None, :] & (jj[None, :] != cc[:, None]) & (pdis < r_lambda)
        num = jnp.sum(jnp.where(a, wt[None, :] / pdis, 0.0), axis=-1)
        den = jnp.sum(jnp.where(a, wt[None, :], 0.0), axis=-1)
        any_a = jnp.any(a, axis=-1)
        ratio = jnp.where(any_a, num, 1.0) / ((1.0 / r_lambda) * jnp.where(any_a, den, 1.0))
        return jnp.where(any_a, jnp.log(ratio), W_FLOOR)

    c = grid_chunk(kc, K, ZGRID_BUDGET)
    return jax.lax.map(chunk, ci.reshape(kc // c, c)).reshape(kc)


def center_wcen_one(nb, xyz, zchi, rich, z, maxrad, model: FilterModel, wc: WcenModel) -> Centering:
    """wcen centring of one cluster (unbatched arrays [K]); see :func:`center_wcen`."""
    K = nb.refmag.shape[0]
    kc = min(wc.ncand, K)
    ng = min(wc.maxcen, kc)
    D = model.mpc_per_deg(z)
    mstar = model.mstar(z)
    lam, scaleval, rl = rich.lam, rich.scaleval, rich.r_lambda
    lam_ok = lam > 0
    lam_s = jnp.where(lam_ok, lam, 1.0)

    # Candidates (a failed zred has zred = zred_e = chi^2 = -1 and is never one).
    zok = (nb.zred_e > 0) & (zchi >= 0)
    ze = jnp.where(zok, nb.zred_e, 1.0)
    use = (nb.valid & lam_ok & zok & (rich.r < jnp.minimum(rl, maxrad))
           & (nb.pfree >= wc.pbcg_cut) & (zchi < wc.zred_chisq_max)
           & ((rich.pmem > 0) | (jnp.abs(z - nb.zred) < ZRED_NSIG * ze)))

    # Central magnitude and zred filters (zred on z: no zrmod, as redMaPPer's centring); widths of
    # the ln w Gaussians.
    mbar = mstar + wc.delta0 + wc.delta1 * jnp.log(lam_s / wc.pivot)
    gz = _gauss(nb.zred, z, ze)
    s = _gauss(nb.refmag, mbar, wc.sigma_m) * gz
    sigscale = jnp.sqrt((jnp.minimum(lam_s, wc.maxlambda) / scaleval) / wc.pivot)
    sig_cen = wc.lnw_cen_sigma / sigscale
    sig_sat = wc.lnw_sat_sigma / sigscale
    sig_fg = wc.lnw_fg_sigma / sigscale

    # Candidate cap: N(ln w; cen) <= 1/(sqrt(2 pi) sig), so a candidate below the bound can never
    # reach ucen >= 1e-10. The ``kc`` best of the others by pfree phi_cen gz enter the pairwise w,
    # in neighbour order (redMaPPer's order, which breaks ties in P_C).
    fwmax = 1.0 / (SQRT2PI * sig_cen)
    pre = use & jnp.isfinite(s) & (s * fwmax >= UFLOOR)
    _, ci = jax.lax.top_k(jnp.where(pre, nb.pfree * s, -1.0), kc)
    ci = jnp.sort(ci)

    # Connectivity w, weighted by the raw p (and luminosity) of the members.
    lum = 10.0 ** ((mstar - nb.refmag) / 2.5) if wc.uselum else 1.0
    mem = nb.valid & (rich.p > 0)
    wt = jnp.where(mem, rich.p * lum, 0.0)
    w = _connectivity(ci, xyz, wt, mem, D, rl, wc.rsoft)
    lnw = jnp.log(w)

    # Central, satellite and foreground/background terms of the capped candidates.
    m_c, zr_c, gz_c = nb.refmag[ci], nb.zred[ci], gz[ci]
    ucen = s[ci] * _gauss(lnw, wc.lnw_cen_mean, sig_cen)
    ucen = jnp.where(ucen < UFLOOR, 0.0, ucen)
    phi_sat = prof.schechter(m_c, mstar, model.alpha) / prof.lumnorm(mstar, model.maxmag(z), model.alpha)
    usat = phi_sat * gz_c * _gauss(lnw, wc.lnw_sat_mean, sig_sat)
    usat = jnp.where(usat < UFLOOR, 0.0, usat)
    sigma_g = wc.zbkg.lookup(zr_c, m_c)
    bcounts = _gauss(lnw, wc.lnw_fg_mean, sig_fg) * (sigma_g / (D * D)) * (jnp.pi * rl**2)
    pf = nb.pfree[ci]
    other = (lam_s / scaleval - 1.0) * usat + bcounts
    pc = jnp.minimum(pf * (ucen / (ucen + other)), PCEN_CLIP)
    pc = jnp.where(jnp.isfinite(pc) & pre[ci], pc, 0.0)
    # 1 - P_C without cancellation: P_C close to 1 is common, and the odds P/(1 - P) of the
    # renormalisation need 1 - P to float32 relative precision.
    qc = jnp.maximum(((1.0 - pf) * ucen + other) / (ucen + other), 1.0 - PCEN_CLIP)

    # The (up to) maxcen best candidates, renormalised with explicit products over j != i (not a
    # division by 1 - P).
    v, gi = jax.lax.top_k(pc, ng)
    keep = v > 0
    ngood = jnp.sum(keep).astype(jnp.int32)
    found = ngood > 0
    pk = jnp.where(keep, v, 0.0)
    q1 = jnp.where(keep, qc[gi], 1.0)
    others = ~jnp.eye(ng, dtype=bool)
    pu = pk * jnp.prod(jnp.where(others, q1[None, :], 1.0), axis=1)
    q0 = jnp.prod(q1)
    spu = jnp.where(found, jnp.sum(pu), 1.0)
    p_cen = jnp.where(keep, pu / spu, 0.0)
    q_cen = jnp.where(keep, pu / (q0 + spu), 0.0)
    q_miss = jnp.where(found, q0 / (q0 + spu), 1.0)
    not_cen = jnp.sum(jnp.where(others, pu[None, :], 0.0), axis=1) / spu     # 1 - P_CEN

    # Satellite / foreground split, with lambda - 1 (redMaPPer; P_C uses lambda/S - 1).
    ls = (lam_s - 1.0) * usat[gi]
    b = bcounts[gi]
    pfg = b / (ls + b)
    psat = ls / (ls + b)
    pfg = jnp.where(keep & jnp.isfinite(pfg), not_cen * pfg, 0.0)
    psat = jnp.where(keep & jnp.isfinite(psat), not_cen * psat, 0.0)
    index = jnp.where(keep, ci[gi], -1).astype(jnp.int32)
    p_c = jnp.where(keep, v, 0.0)

    if ng < wc.maxcen:      # fewer neighbours than slots
        pad = lambda a, val: jnp.concatenate([a, jnp.full((wc.maxcen - ng,), val, a.dtype)])
        index = pad(index, -1)
        p_cen, q_cen, psat, pfg, p_c = (pad(a, 0.0) for a in (p_cen, q_cen, psat, pfg, p_c))
    return Centering(index=index, p_cen=p_cen, q_cen=q_cen, p_sat=psat, p_fg=pfg, p_c=p_c,
                     q_miss=q_miss, ngood=ngood, ncand=jnp.sum(use).astype(jnp.int32))


@jax.jit
def center_wcen(nb, xyz, zchi, rich, z, maxrad, model: FilterModel, wc: WcenModel) -> Centering:
    """wcen centring (redMaPPer's CenteringWcenZred) of a batch of clusters.

    Parameters
    ----------
    nb : neighbours [B, K] (zred, zred_e, refmag, pfree, valid).
    xyz : unit vectors of the neighbours [B, K, 3] (float32).
    zchi : zred chi^2 of the neighbours [B, K] (-1 where the zred fit failed).
    rich : richness at ``z`` with distances from the current position (lam, scaleval, r_lambda,
        p, pmem, r); clusters with lam <= 0 get no centre.
    z : cluster redshifts [B].
    maxrad : candidates also need r < maxrad [h^-1 Mpc] (1e3 in blind mode, scan_maxrad in scan
        mode).
    """
    fn = jax.vmap(center_wcen_one, in_axes=(0, 0, 0, 0, 0, None, None, None))
    return fn(nb, xyz, zchi, rich, z, maxrad, model, wc)


@partial(jax.jit, static_argnames=("maxcen",))
def as_centering(ic, found, maxcen: int = 5) -> Centering:
    """A single central per cluster (CenteringBCG) in the :class:`Centering` layout.

    index[:, 0] = ic and P_CEN = Q_CEN = 1 in slot 0 where ``found`` (NGOOD = 1, Q_MISS = 0);
    P_SAT, P_FG and P_C are 0 (as redMaPPer's CenteringBCG).
    """
    found = jnp.asarray(found, bool)
    first = found[:, None] & (jnp.arange(maxcen) == 0)[None, :]
    one = first.astype(jnp.float32)
    zero = jnp.zeros(first.shape, jnp.float32)
    nfound = found.astype(jnp.int32)
    return Centering(index=jnp.where(first, jnp.asarray(ic, jnp.int32)[:, None], -1),
                     p_cen=one, q_cen=one, p_sat=zero, p_fg=zero, p_c=zero,
                     q_miss=1.0 - found.astype(jnp.float32), ngood=nfound, ncand=nfound)


# --------------------------------------------------------------------------- LNCGLIKE
def lncglike_one(nb, rich, z, model: FilterModel, wc: WcenModel):
    """LNCGLIKE of one cluster (unbatched arrays [K]); see :func:`lncglike`."""
    lam, scaleval, rl = rich.lam, rich.scaleval, rich.r_lambda
    lam_ok = lam > 0
    lam_s = jnp.where(lam_ok, lam, 1.0)
    mstar = model.mstar(z)
    cen = nb.is_center & nb.valid
    ic = jnp.argmax(cen)
    # Central magnitude.
    mbar = mstar + wc.delta0 + wc.delta1 * jnp.log(lam_s / wc.pivot)
    ln_phi = -0.5 * ((nb.refmag[ic] - mbar) / wc.sigma_m) ** 2 - jnp.log(SQRT2PI * wc.sigma_m)
    # zred of the central against zrmod(z), the median zred of centrals at z (z itself without a
    # mapping), floored at 1e-10 (a failed zred gives the floor).
    zr = _interpol(wc.zrmod, wc.zrmod_z, z)
    ze_c = nb.zred_e[ic]
    ze = jnp.where(ze_c > 0, ze_c, 1.0)
    ln_g = -0.5 * ((nb.zred[ic] - zr) / ze) ** 2 - jnp.log(SQRT2PI * ze)
    ln_g = jnp.where(ze_c > 0, jnp.maximum(ln_g, LN_G_FLOOR), LN_G_FLOOR)
    # Connectivity around the position: pmem L weights, no r_lambda cut, always luminosity.
    lum = 10.0 ** ((mstar - nb.refmag) / 2.5)
    u = nb.valid & (rich.r > R_MIN_LIKE) & (rich.pmem > 0)
    wt = jnp.where(u, rich.pmem * lum, 0.0)
    num = jnp.sum(wt / jnp.sqrt(rich.r**2 + wc.rsoft**2))
    den = (1.0 / rl) * jnp.sum(wt)
    ok = jnp.any(cen) & lam_ok & (num > 0) & (den > 0)
    w = jnp.log(jnp.where(ok, num, 1.0) / jnp.where(ok, den, 1.0))
    ok &= w > 0
    sig = wc.lnw_cen_sigma / jnp.sqrt((jnp.minimum(lam_s, wc.maxlambda) / scaleval) / wc.pivot)
    ln_fw = (-0.5 * ((jnp.log(jnp.where(ok, w, 1.0)) - wc.lnw_cen_mean) / sig) ** 2
             - jnp.log(SQRT2PI * sig))
    return jnp.where(ok, ln_phi + ln_g + ln_fw, jnp.nan)


@jax.jit
def lncglike(nb, rich, z, model: FilterModel, wc: WcenModel):
    """LNCGLIKE [B]: redMaPPer's likelihood-pass central term (run_likelihoods.py), in log space.

    ln N(m_c; m* + Delta0 + Delta1 ln(lambda/pivot), sigma_m) + ln max(N(zred_c; zrmod(z),
    zred_e_c), 1e-10) + ln N(ln w; lnw_cen_mean, lnw_cen_sigma / sigscale) for the central c (the
    neighbour flagged ``is_center``, the seed), with w = ln[sum pmem L / sqrt(r^2 + rsoft^2) /
    ((1/r_lambda) sum pmem L)] over the members with r > 1e-5 from the position. zrmod is the
    z -> zred_uncorr table of ``wc`` (z itself without a mapping) at z, the first-pass z_lambda in
    blind mode. NaN when there is no central, w <= 0 or lambda <= 0.
    """
    return jax.vmap(lncglike_one, in_axes=(0, 0, 0, None, None))(nb, rich, z, model, wc)


# --------------------------------------------------------------------------- W column
def w_column(r, p, refmag, valid, mstar, r_lambda, rsoft: float = 0.05, uselum: bool = True) -> float:
    """Catalogue connectivity W of one cluster at its final centre (host, numpy).

    W = ln[sum p L / sqrt(r^2 + rsoft^2) / ((1/r_lambda) sum p L)], L = 10^(0.4 (m* - m))
    (1 without ``uselum``), over the valid neighbours with p > 0 and r < r_lambda, excluding the
    one nearest the centre (redMaPPer's RunPercolation). NaN without such neighbours.
    """
    r = np.asarray(r, np.float64)
    p = np.asarray(p, np.float64)
    refmag = np.asarray(refmag, np.float64)
    valid = np.asarray(valid, bool)
    if not valid.any():
        return float("nan")
    u = valid & (r > r[valid].min()) & (r < r_lambda) & (p > 0)
    if not u.any():
        return float("nan")
    wt = p[u] * (10.0 ** ((mstar - refmag[u]) / 2.5) if uselum else 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.log(np.sum(wt / np.sqrt(r[u] ** 2 + rsoft**2)) / ((1.0 / r_lambda) * np.sum(wt))))
