"""Blind cluster finding: seeds -> first pass -> likelihood -> percolation -> catalogue.

Follows redMaPPer's run mode:

1. **Seeds**: galaxies with zred in ``model.zrange``, zred chi^2 < ``seeds.chisq_max`` and
   m < m*(zred) + ``seeds.dmag_max``.
2. **First pass** (``richness.firstpass``: r0 = 0.5 h^-1 Mpc, beta = 0): z_lambda and lambda
   centred on each seed, starting from its zred; candidates with lambda >= ``minlambda`` and
   z_lambda in range are kept.
3. **Likelihood pass** (``richness.likelihoods``: r0 = 1, beta = 0.2): lambda at the first-pass
   z_lambda; LNLIKE = LNLAMLIKE + LNCGLIKE (LNCGLIKE = 0 until a wcen centring model exists).
4. **Percolation** (``richness.percolation``), in order of decreasing LNLIKE (ties by seed ID):
   a candidate whose seed galaxy has pfree < ``centering.pbcg_cut`` is skipped. Otherwise, as in
   redMaPPer's RunPercolation, lambda is computed at the seed with the current pfree (dropped if
   lambda/S < ``minlambda``), then z_lambda at the seed, the central galaxy is chosen
   (CenteringBCG), and z_lambda and lambda are recomputed at the new centre. The candidate is
   dropped if no central is found, if the central's pfree < ``pbcg_cut`` or if lambda/S <
   ``minlambda``. For an accepted cluster, every galaxy within
   R_MASK = max(r_lambda, rmask_0 (lambda/100)^rmask_beta ((1+z)/(1+zpivot))^rmask_gamma) and
   brighter than m*(z) - 2.5 log10(lmask) has its probability p added to its claimed fraction
   (pfree = 1 - claimed, clipped to [0, 1]). The result is that of the sequential loop,
   computed in batches of independent candidates (see :func:`percolate`).
5. **Consolidation**: clusters with lambda >= minlambda, lambda/scaleval >= minlambda,
   MASKFRAC < ``mask.max_maskfrac`` and centre inside the region's own box are kept; members
   are the galaxies with p pfree theta_i theta_r >= ``percolation.member_pmin``.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..core.centering import as_centering, center_bcg, center_wcen, lncglike, w_column
from ..core.richness import Neighbors, RadialQuad, Stage, richness
from ..core.zlambda import ZLambda, zlambda
from ..sky.neighbors import radec_from_unit, unit_vectors
from ..sky.regions import Box
from .common import Region, snr

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- seeds
def select_seeds(region: Region) -> np.ndarray:
    """Seed galaxies: zred in the redshift range with a red-sequence chi^2 below
    ``seeds.chisq_max``, or, with the photo-z filter, ZPHOT in the range with a calibrated width
    s / (1 + ZPHOT) below ``seeds.zphot_err_max`` (any colour); brighter than m*(z) + dmag_max."""
    cfg = region.cfg
    g = region.gal
    z = region.seed_z(slice(None))
    ok = (z >= cfg.model.zrange[0]) & (z <= cfg.model.zrange[1])
    if region.model.filter == "photoz":
        s = g["ZPHOT_E"]
        ok &= (s > 0) & (s < cfg.seeds.zphot_err_max * (1.0 + z))
    else:
        ok &= (g["ZRED_CHISQ"] >= 0) & (g["ZRED_CHISQ"] < cfg.seeds.chisq_max)
    ok &= g["REFMAG"] < region.mstar_np(np.clip(z, 0.01, 2.0)) + cfg.seeds.dmag_max
    return np.flatnonzero(ok)


# --------------------------------------------------------------------------- batched stages
def _bucket(n: int, k0: int, base: int) -> int:
    """Smallest k0 * base**i >= n: arrays come in a few sizes, each compiled once."""
    k = k0
    while k < n:
        k *= base
    return k


def _shape_bases() -> tuple[int, int]:
    """(neighbour-count base, batch-size base) for padding array shapes.

    Powers of 4 and 8 on GPUs, where padded entries are almost free and every extra shape costs
    a compilation; powers of 2 on CPUs, where padded entries cost as much as real ones.
    """
    return (4, 8) if jax.default_backend() == "gpu" else (2, 2)


def _pad_batch(arrs, B):
    n = arrs[0].shape[0]
    if n == B:
        return arrs, n
    return [np.concatenate([a, np.repeat(a[:1], B - n, axis=0)]) for a in arrs], n


@dataclass
class _Batch:
    """Device inputs of one sub-batch of seeds (see :func:`_iter_batches`)."""

    rows: np.ndarray          # positions of the seeds in ``gi``; the first ``n`` are real
    n: int
    nb: Neighbors
    frad: jnp.ndarray
    fgeo: jnp.ndarray
    z: np.ndarray             # [B] float64, the seeds' redshifts (padded)
    ids: np.ndarray           # [B] galaxy IDs of the seeds (padded)
    K: int
    B: int
    outer: int                # index of the query batch, and seeds queried so far
    done: int
    last: bool                # last sub-batch of its query batch


def _iter_batches(region: Region, gi: np.ndarray, z0: np.ndarray, stage: Stage, quad: RadialQuad,
                  kind: str, batch: int = 1024, budget: int | None = None, pfree_nb=None):
    """Padded neighbours and aperture completeness of seeds ``gi`` centred on themselves.

    Seeds are processed in redshift order; each query of ``batch`` seeds is split into
    sub-batches of B seeds with B x K <= ``budget`` (B a power of two), so device memory stays
    bounded. kind = "zlambda" reads neighbours over z0 +- 0.05, "richness" at z0.
    ``pfree_nb(rows, idx, valid)`` -> [B, K] gives the free fraction of each neighbour (galaxy
    indices ``idx``) of the seeds at ``rows`` of ``gi`` (default 1).
    """
    cfg = region.cfg
    g = region.gal
    if budget is None:
        budget = 2**19 if kind == "zlambda" else 2**21
    # XLA's GPU compile time grows quickly with the batch size (~13 s at 256, ~50 s at 1024);
    # the run time per cluster is already flat at 256.
    bmax = 256 if jax.default_backend() == "gpu" else batch
    zfloor = cfg.model.zrange[0]
    order = np.argsort(z0, kind="stable")
    for ib, b0 in enumerate(range(0, gi.size, batch)):
        sel = order[b0:b0 + batch]
        ra, dec, ids = g["RA"][gi[sel]], g["DEC"][gi[sel]], g["ID"][gi[sel]]
        zz = z0[sel].astype(np.float64)
        dzz = 0.05 if kind == "zlambda" else 0.0
        zlo = np.maximum(zz - dzz, zfloor)
        rad = region.radius_deg(float(stage.maxrad), zlo)
        # galaxies fainter than maxmag + 0.5 have theta_i ~ 0; each seed keeps its own limit
        mlim = region.maxmag(zz + dzz) + 0.5
        pad = region.query(ra, dec, rad, mag_max=float(mlim.max()))
        pad.valid &= g["REFMAG"][pad.idx] < mlim[:, None]
        cnt = pad.valid.sum(axis=1)
        K = _bucket(int(cnt.max(initial=1)), 128, _shape_bases()[0])
        pad.idx, pad.valid, _ = _crop(pad.idx, pad.valid, K)
        pad.theta = _theta_for(region, ra, dec, pad.idx, pad.valid)
        Bs = int(2 ** np.floor(np.log2(max(32, min(batch, bmax, budget // K)))))
        for s0 in range(0, sel.size, Bs):
            ss = slice(s0, s0 + Bs)
            (rows_p, ra_p, dec_p, ids_p, zz_p, pidx, pval, pth), n = _pad_batch(
                [sel[ss], ra[ss], dec[ss], ids[ss], zz[ss], pad.idx[ss], pad.valid[ss], pad.theta[ss]], Bs)
            sub = type(pad)(pidx, pval, pth, np.zeros(Bs, np.int64))
            pf = None if pfree_nb is None else pfree_nb(rows_p, pidx, pval)
            nb = region.neighbors(sub, center_ids=ids_p, pfree_nb=pf)
            fr, fg = region.completeness(ra_p, dec_p, zz_p, quad)
            yield _Batch(rows=rows_p, n=n, nb=nb, frad=fr, fgeo=fg, z=zz_p, ids=ids_p, K=K, B=Bs,
                         outer=ib, done=b0 + sel.size, last=s0 + Bs >= sel.size)


def _run_batched(region: Region, gi: np.ndarray, z0: np.ndarray, stage: Stage, quad: RadialQuad,
                 kind: str, batch: int = 1024, log_every: int = 20, budget: int | None = None,
                 calc_err: bool = True, pfree_nb=None):
    """Per-seed stage over galaxies ``gi`` centred on themselves, starting at ``z0``.

    kind = "zlambda" (first pass; Z_LAMBDA_E only with ``calc_err``) or "richness" (likelihood
    pass). Batching and ``pfree_nb``: see :func:`_iter_batches`.
    """
    zc = region.cfg.zlambda
    out = {k: np.full(gi.size, np.nan, np.float32) for k in
           ("LAMBDA", "LAMBDA_E", "Z_LAMBDA", "Z_LAMBDA_E", "R_LAMBDA", "SCALEVAL", "MASKFRAC",
            "LNLAMLIKE", "LNCGLIKE", "NINCUT")}
    t0 = time.time()
    for bt in _iter_batches(region, gi, z0, stage, quad, kind, batch, budget, pfree_nb):
        n, nb, tgt = bt.n, bt.nb, bt.rows[:bt.n]
        if kind == "zlambda":
            zl = zlambda(nb, jnp.asarray(bt.z, jnp.float32), bt.frad, bt.fgeo, quad, region.model,
                         stage, maxiter=zc.maxiter, tol=zc.tol, ngrid=zc.ngrid,
                         half_width=zc.half_width, npz=zc.npzbins, topfrac=zc.topfrac,
                         soft=zc.pcol_soft, calc_err=calc_err)
            rich, zres, zres_e = zl.rich, np.asarray(zl.z)[:n], np.asarray(zl.z_e)[:n]
        else:
            zj = jnp.asarray(bt.z, jnp.float32)
            rich = richness(nb, zj, bt.frad, bt.fgeo, quad, region.model, stage)
            zres, zres_e = bt.z[:n], np.full(n, np.nan)
            if region.wcen is not None:
                # Central-galaxy likelihood of the seed (redMaPPer's likelihood pass).
                out["LNCGLIKE"][tgt] = np.asarray(lncglike(nb, rich, zj, region.model, region.wcen))[:n]
            # Members other than the seed (redMaPPer rejects fewer than 3).
            out["NINCUT"][tgt] = np.sum((np.asarray(rich.pmem) > 0)
                                        & ~np.asarray(nb.is_center), axis=1)[:n]
        out["LAMBDA"][tgt] = np.asarray(rich.lam)[:n]
        out["LAMBDA_E"][tgt] = np.asarray(rich.lam_e)[:n]
        out["R_LAMBDA"][tgt] = np.asarray(rich.r_lambda)[:n]
        out["SCALEVAL"][tgt] = np.asarray(rich.scaleval)[:n]
        out["MASKFRAC"][tgt] = np.asarray(rich.maskfrac)[:n]
        out["LNLAMLIKE"][tgt] = np.asarray(rich.lnlamlike)[:n]
        out["Z_LAMBDA"][tgt] = zres
        out["Z_LAMBDA_E"][tgt] = zres_e
        if kind != "richness" or region.wcen is None:
            out["LNCGLIKE"][tgt] = 0.0
        if log_every and bt.last and bt.outer % log_every == 0:
            log.info("%s: %d/%d seeds (K=%d, B=%d), %.1fs", kind, bt.done, gi.size, bt.K, bt.B,
                     time.time() - t0)
    return out


# --------------------------------------------------------------------------- percolation step
@jax.jit
def _with_center(nb, gxyz, cxyz, ic):
    """Neighbours re-centred on cxyz [B, 3] (theta from unit vectors; is_center = index ic)."""
    d = jnp.linalg.norm(gxyz - cxyz[:, None, :], axis=-1)
    th = jnp.rad2deg(2.0 * jnp.arcsin(jnp.clip(d / 2.0, 0.0, 1.0)))
    isc = jnp.arange(gxyz.shape[1])[None, :] == ic[:, None]
    return dataclasses.replace(nb, theta=th.astype(nb.theta.dtype), is_center=isc)


def _fixed_z(nb, z0, frad, fgeo, quad, model, stage, npz):
    rich = richness(nb, z0, frad, fgeo, quad, model, stage)
    zero = jnp.zeros(z0.shape + (npz,), z0.dtype)
    return ZLambda(z=jnp.where(rich.lam > 0, z0, -1.0), z_e=jnp.zeros_like(z0),
                   niter=jnp.zeros(z0.shape, jnp.int32), pzbins=zero, pz=zero, rich=rich)


def _seed_frame(nb, gxyz, cxyz):
    """Neighbours relative to the seed position cxyz [B, 3], keeping the seed flag."""
    B = gxyz.shape[0]
    nb0 = _with_center(nb, gxyz, cxyz, jnp.full((B,), -1, jnp.int32))
    return dataclasses.replace(nb0, is_center=nb.is_center)


@jax.jit
def _random_point(cxyz, r_lambda, mpc_per_deg, u):
    """Points uniform in area within r_lambda [h^-1 Mpc] of cxyz [B, 3] (redMaPPer
    CenteringRandom: r = r_lambda sqrt(u0), position angle 2 pi u1)."""
    ang = jnp.deg2rad(r_lambda * jnp.sqrt(u[:, 0]) / mpc_per_deg)
    phi = 2.0 * jnp.pi * u[:, 1]
    x, y, zc = cxyz[:, 0], cxyz[:, 1], cxyz[:, 2]
    rho = jnp.maximum(jnp.sqrt(x * x + y * y), 1e-12)
    east = jnp.stack([-y / rho, x / rho, jnp.zeros_like(x)], axis=-1)
    north = jnp.stack([-zc * x / rho, -zc * y / rho, rho], axis=-1)
    step = jnp.cos(phi)[:, None] * east + jnp.sin(phi)[:, None] * north
    return jnp.cos(ang)[:, None] * cxyz + jnp.sin(ang)[:, None] * step


@jax.jit
def _draw_member(pmem, u):
    """Neighbour index drawn with probability proportional to pmem [B, K] (redMaPPer
    CenteringRandomSatellite); -1 when no neighbour has pmem > 0."""
    cdf = jnp.cumsum(pmem, axis=1)
    tot = cdf[:, -1]
    i = jnp.sum(cdf < (u * tot)[:, None], axis=1)
    return jnp.where(tot > 0, jnp.minimum(i, pmem.shape[1] - 1), -1).astype(jnp.int32)


def _perc_batch(nb0, gxyz, cxyz, z0, frad, fgeo, quad, model, stage, centering: str = "bcg",
                wc=None, zchi=None, urand=None, stage_seed=None, maxcen: int = 5, maxiter: int = 5,
                tol: float = 2e-4, ngrid: int = 21, half_width: float = 0.03, npz: int = 21,
                topfrac: float = 0.7, soft: float = 0.04, keepz: bool = False):
    """One percolation step for a batch of candidates (neighbours ``nb0`` in the seed frame).

    z_lambda (or lambda at fixed z with ``keepz``) at the seed without errors, centring, then
    z_lambda with errors and p(z) at the new centre, as in redMaPPer. ``centering``:

    - "bcg": CenteringBCG; "wcen": CenteringWcenZred (needs ``wc``, the WcenModel, and ``zchi``,
      the neighbours' zred chi^2);
    - "random" (a point uniform within r_lambda of the seed) and "randsat" (a member drawn with
      probability pmem), for the wcen calibration; ``urand`` [B, 2] uniform deviates.

    The seed step uses ``stage_seed`` (default ``stage``), whose maxrad sets how far from the
    seed the membership probabilities p entering the wcen connectivity are computed.

    Built from the jitted ``zlambda``/``richness``/centring programs. Returns (z_lambda at the
    centre, Centering, centre unit vector [B, 3]).
    """
    kw = dict(maxiter=maxiter, tol=tol, ngrid=ngrid, half_width=half_width, npz=npz,
              topfrac=topfrac, soft=soft)

    def zfun(nbx, zz, calc_err, st):
        if keepz:
            return _fixed_z(nbx, zz, frad, fgeo, quad, model, st, npz)
        return zlambda(nbx, zz, frad, fgeo, quad, model, st, calc_err=calc_err, **kw)

    zl0 = zfun(nb0, z0, False, stage if stage_seed is None else stage_seed)
    ok0 = zl0.rich.lam > 0
    rich0 = zl0.rich
    if centering == "wcen":
        cen = center_wcen(nb0, gxyz, zchi, rich0, zl0.z, jnp.asarray(1e3, zl0.z.dtype), model, wc)
    elif centering == "randsat":
        ic = _draw_member(rich0.pmem, urand[:, 0])
        cen = as_centering(ic, ok0 & (ic >= 0), maxcen)
    elif centering == "random":
        cen = as_centering(jnp.full(ok0.shape, -1, jnp.int32), ok0, maxcen)
    else:
        ic, found = center_bcg(nb0, rich0.pmem, rich0.r, rich0.r_lambda, zl0.z,
                               jnp.asarray(1e3, zl0.z.dtype), use_zphot=model.filter == "photoz")
        cen = as_centering(jnp.where(found, ic, -1).astype(jnp.int32), found & ok0, maxcen)
    ic = cen.index[:, 0]
    found = (cen.ngood > 0) & ok0
    if centering == "random":
        c1 = jnp.where(found[:, None],
                       _random_point(cxyz, rich0.r_lambda, model.mpc_per_deg(zl0.z), urand), cxyz)
    else:
        gal = jnp.take_along_axis(gxyz, jnp.maximum(ic, 0)[:, None, None], axis=1)[:, 0, :]
        c1 = jnp.where((found & (ic >= 0))[:, None], gal, cxyz)
    nb1 = _with_center(nb0, gxyz, c1, jnp.where(found, ic, -1))
    zl1 = zfun(nb1, jnp.where(ok0, zl0.z, z0), True, stage)
    return zl1, cen, c1


def _dependencies(xyz, rank, rad_read, rad_reach, factor: float = 1.5):
    """Pairs (i, j) such that candidate j must wait for candidate i.

    rank[i] < rank[j] and the seed separation is below rad_read[j] + rad_reach[i] (degrees).
    Candidates are binned by read radius (as j) and by reach (as i) in bins a factor ``factor``
    wide, and every pair of bins is matched with a dual-tree search bounded by the two bins'
    largest radii; no search uses the global largest reach (degrees at low redshift). Returns
    int32 arrays (i, j).
    """
    from scipy.spatial import cKDTree

    if rank.size == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)

    def bins(r):
        lr = np.log(np.maximum(r, 1e-6))
        k = np.floor((lr - lr.min()) / np.log(factor)).astype(np.int64)
        return [m for m in (np.flatnonzero(k == b) for b in np.unique(k))]

    bj, bi = bins(rad_read), bins(rad_reach)
    tj = [(m, cKDTree(xyz[m]), float(rad_read[m].max())) for m in bj]
    out_i, out_j = [], []
    for mi in bi:
        ti, ri = cKDTree(xyz[mi]), float(rad_reach[mi].max())
        for mj, tree_j, rj in tj:
            chord = 2.0 * np.sin(np.radians(min(ri + rj, 180.0)) / 2.0)
            pairs = ti.sparse_distance_matrix(tree_j, chord, output_type="ndarray")
            if pairs.size == 0:
                continue
            a, b = mi[pairs["i"]], mj[pairs["j"]]
            sep = np.degrees(2.0 * np.arcsin(np.clip(pairs["v"] / 2.0, 0.0, 1.0)))
            ok = (rank[a] < rank[b]) & (sep < rad_read[b] + rad_reach[a])
            out_i.append(a[ok].astype(np.int32))
            out_j.append(b[ok].astype(np.int32))
            del pairs, a, b, sep, ok
    if not out_i:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    return np.concatenate(out_i), np.concatenate(out_j)


def percolate(region: Region, cand: dict, stage: Stage, quad: RadialQuad, batch: int = 64,
              kmin: int = 512, log_every: int = 50, keepz: bool = False,
              centering: str | None = None, seed: int = 12345, info: dict | None = None):
    """Percolation of candidates (dict with seed galaxy index GI, Z_LAMBDA, LAMBDA, LNLIKE).

    Exact sequential semantics (rank by decreasing LNLIKE, ties by seed ID), computed on the
    dependency graph of the candidates: candidate j depends on every higher-ranked candidate i
    whose claims can reach j's read region (seed separation < read radius of j + reach of i,
    both bounded from 2 x the likelihood-pass lambda). A candidate is computed once all its
    dependencies are done; the ready candidates do not depend on each other, so they are
    computed together (up to ``batch`` per call, grouped by neighbour-count bucket) and their
    claims applied in any order. ``batch = 1`` is the plain sequential loop. Each round first
    evaluates lambda at the seeds (one richness call) and runs z_lambda only for the candidates
    that pass lambda/S >= minlambda.

    ``centering``: "bcg", "wcen" or "auto" (default: cfg.centering.method, see
    :meth:`Region.centering_method`), or "random" / "randsat" (random centres or satellites,
    drawn with ``seed``, for the wcen calibration). ``info``: a dict that receives the counts
    (dependencies, rounds, calls, clusters, skipped, dropped by reason) and the time.
    """
    cfg = region.cfg
    g = region.gal
    pc = cfg.percolation
    zc = cfg.zlambda
    pgal = np.zeros(g["RA"].size, np.float64)
    xyz_all = unit_vectors(g["RA"], g["DEC"])
    gi = cand["GI"]
    ncand = gi.size
    order = np.lexsort((g["ID"][gi], -cand["LNLIKE"]))
    rank = np.empty(ncand, np.int64)
    rank[order] = np.arange(ncand)
    # Read radius: the aperture plus room for re-centring within r_lambda, and at least the
    # masking radius; reach: how far from the seed claims can extend.
    lam_est = np.maximum(np.asarray(cand.get("LAMBDA", np.full(ncand, 30.0))), 3.0) * 2.0
    rl_est = float(stage.r0) * (lam_est / 100.0) ** float(stage.beta)
    rmask_est = pc.rmask_0 * (lam_est / 100.0) ** pc.rmask_beta
    zcand = np.asarray(cand["Z_LAMBDA"], np.float64)
    # Same redshift floor as the first pass: angular radii stay bounded at low z.
    zlo = np.maximum(zcand - 0.05, cfg.model.zrange[0])
    rad_read = region.radius_deg(np.maximum(float(stage.maxrad), rmask_est) + rl_est, zlo)
    rad_reach = region.radius_deg(rmask_est + rl_est, zlo)
    mag_lim = np.maximum(region.maxmag(zcand + 0.05) + 0.5,
                         region.mstar_np(zcand + 0.05) - 2.5 * np.log10(pc.lmask)) + 0.1
    seed_xyz = xyz_all[gi]

    # Dependency graph (CSR): blockers of j = higher-ranked i with sep < read_j + reach_i.
    bi, bj = _dependencies(seed_xyz, rank, rad_read, rad_reach)
    nblock = np.bincount(bj, minlength=ncand)
    order_i = np.argsort(bi, kind="stable")
    dep_j = bj[order_i]
    dep_ptr = np.zeros(ncand + 1, np.int64)
    np.cumsum(np.bincount(bi, minlength=ncand), out=dep_ptr[1:])
    ndep = int(bi.size)
    log.info("percolation: %d candidates, %d dependencies", ncand, ndep)
    del bi, bj, order_i

    method = centering if centering in ("random", "randsat") else region.centering_method(centering)
    maxcen = cfg.centering.maxcen
    kw = dict(maxiter=zc.maxiter, tol=zc.tol, ngrid=zc.ngrid, half_width=zc.half_width,
              npz=zc.npzbins, topfrac=zc.topfrac, soft=zc.pcol_soft, keepz=keepz,
              centering=method, wc=region.wcen if method == "wcen" else None, maxcen=maxcen)
    minlam, pbcg = cfg.richness.minlambda, cfg.centering.pbcg_cut
    # Before centring, p is needed out to about 2 r_lambda from the seed (the members within
    # r_lambda of each candidate centre). As redMaPPer, reach its neighbour radius
    # max(r0, rmask_0) 3^beta (1.87 h^-1 Mpc); the read radius already covers it.
    kw["stage_seed"] = Stage(stage.r0, stage.beta,
                             jnp.maximum(stage.maxrad, pc.rmask_0 * 3.0 ** pc.rmask_beta))
    # One pair of deviates per candidate (random centring), independent of the batching.
    urand = (np.random.default_rng(seed).uniform(size=(ncand, 2)).astype(np.float32)
             if method in ("random", "randsat") else None)
    import heapq

    ready = [(rank[j], j) for j in range(ncand) if nblock[j] == 0]
    heapq.heapify(ready)
    clusters, members = [], []
    t0 = time.time()
    nskip = nround = ncalls = 0
    drop = dict.fromkeys(("seed", "zlambda", "centre", "centre_pfree", "lambda", "w"), 0)

    kbase, bbase = _shape_bases()

    def batches(counts):
        """(K, rows) sub-batches, K from a few sizes (see _shape_bases)."""
        bk = np.array([_bucket(int(c), kmin, kbase) for c in counts], np.int64)
        for K in np.unique(bk):
            sel = np.flatnonzero(bk == K)
            for s0 in range(0, sel.size, batch):
                yield int(K), sel[s0:s0 + batch]

    def padded(n):
        return min(_bucket(n, 1, bbase), batch) if n <= batch else n

    def prepare(pad, ks, ss, K):
        """Device inputs for rows ``ss`` of a round, padded to one of a few batch sizes."""
        n = ss.size
        Bc = padded(n)
        sub_idx, sub_valid, sub_theta = _crop(pad.idx[ss], pad.valid[ss], K)
        pf = np.clip(1.0 - pgal[sub_idx], 0.0, 1.0)
        extra = []
        if method == "wcen":
            extra.append(np.where(sub_valid, g["ZRED_CHISQ"][sub_idx], -1.0))
        if urand is not None:
            extra.append(urand[ks[ss]])
        (pidx, pval, pth, pfp, sid, sx, z0, *ex), _ = _pad_batch(
            [sub_idx, sub_valid, sub_theta, pf, g["ID"][gi[ks[ss]]], seed_xyz[ks[ss]],
             zcand[ks[ss]], *extra], Bc)
        sub = type(pad)(pidx, pval, pth, np.zeros(Bc, np.int64))
        gx = jnp.asarray(xyz_all[pidx], jnp.float32)
        sx = jnp.asarray(sx, jnp.float32)
        nb0 = _seed_frame(region.neighbors(sub, pfree_nb=pfp, center_ids=sid), gx, sx)
        rows = ks[ss[np.arange(Bc) % n]]
        fr, fg = region.completeness(g["RA"][gi[rows]], g["DEC"][gi[rows]], z0, quad)
        zchi = jnp.asarray(ex.pop(0), jnp.float32) if method == "wcen" else None
        ur = jnp.asarray(ex.pop(0), jnp.float32) if urand is not None else None
        return dict(nb0=nb0, gx=gx, sx=sx, z0=jnp.asarray(z0, jnp.float32), fr=fr, fg=fg,
                    idx=sub_idx, valid=sub_valid, pf=pf, n=n, zchi=zchi, urand=ur)


    def finish(j):
        ds = dep_j[dep_ptr[j]:dep_ptr[j + 1]]
        if ds.size:
            nblock[ds] -= 1
            for d in ds[nblock[ds] == 0]:
                heapq.heappush(ready, (rank[d], d))

    while ready:
        take = [heapq.heappop(ready)[1] for _ in range(min(batch, len(ready)))]
        nround += 1
        todo = []
        for k in take:
            if 1.0 - pgal[gi[k]] < cfg.centering.pbcg_cut:
                nskip += 1
                finish(k)
            else:
                todo.append(k)
        if not todo:
            continue
        ks = np.array(todo)
        pad = region.query(g["RA"][gi[ks]], g["DEC"][gi[ks]], rad_read[ks],
                           mag_max=float(mag_lim[ks].max()), kmin=kmin)
        # Each candidate sees exactly its own neighbours (its own magnitude limit).
        pad.valid &= g["REFMAG"][pad.idx] < mag_lim[ks][:, None]
        counts = pad.valid.sum(axis=1)
        # 1. As redMaPPer: lambda at the seed and the candidate's redshift with the current
        #    pfree; lambda/S < minlambda drops the candidate before any z_lambda work.
        seed_ok = np.zeros(ks.size, bool)
        for K, ss in batches(counts):
            d = prepare(pad, ks, ss, K)
            r0 = richness(d["nb0"], d["z0"], d["fr"], d["fg"], quad, region.model, stage)
            seed_ok[ss] = np.asarray((r0.lam > 0) & (r0.lam >= minlam * r0.scaleval))[:d["n"]]
            ncalls += 1
        drop["seed"] += int((~seed_ok).sum())
        # 2. z_lambda at the seed, re-centring and z_lambda at the new centre.
        live = np.flatnonzero(seed_ok)
        results = {}
        for K, sl in batches(counts[live]):
            ss = live[sl]
            d = prepare(pad, ks, ss, K)
            zl, cen, c1 = _perc_batch(d["nb0"], d["gx"], d["sx"], d["z0"], d["fr"], d["fg"],
                                      quad, region.model, stage, zchi=d["zchi"],
                                      urand=d["urand"], **kw)
            ncalls += 1
            host = jax.tree_util.tree_map(np.asarray, (zl, cen, c1))
            for t, j in enumerate(ss):
                results[j] = (host, t, d["idx"][t], d["valid"][t], d["pf"][t])
        # Candidates of a round are independent: accept them in any order.
        for j in range(ks.size):
            k = ks[j]
            if j not in results:
                finish(k)
                continue
            (zlh, cenh, c1h), t, idx, valid, pf_used = results[j]
            lam, z = float(zlh.rich.lam[t]), float(zlh.z[t])
            ngood, ic = int(cenh.ngood[t]), int(cenh.index[t, 0])
            # As redMaPPer: a centre found (a central galaxy still free, pfree >= pbcg_cut,
            # unless the centre is a random point), lambda/S >= minlambda at the new centre and
            # a defined connectivity W; otherwise the candidate claims nothing.
            galaxy_centre = method != "random"
            W = np.nan
            why = ("zlambda" if not (lam > 0 and z > 0)
                   else "centre" if ngood == 0 or (galaxy_centre and ic < 0)
                   else "centre_pfree" if galaxy_centre and pf_used[ic] < pbcg
                   else "lambda" if lam < minlam * float(zlh.rich.scaleval[t]) else None)
            if why is None:
                W = w_column(zlh.rich.r[t], zlh.rich.p[t], g["REFMAG"][idx], valid,
                             float(region.mstar_np(z)), float(zlh.rich.r_lambda[t]),
                             cfg.centering.wcen_rsoft, cfg.centering.wcen_uselum)
                if not np.isfinite(W):
                    why = "w"
            if why is None:
                rl = float(zlh.rich.r_lambda[t])
                rmask = max(rl, pc.rmask_0 * (lam / 100.0) ** pc.rmask_beta
                            * ((1.0 + z) / (1.0 + pc.rmask_zpivot)) ** pc.rmask_gamma)
                r = zlh.rich.r[t]
                p = zlh.rich.p[t]
                mlim = float(region.mstar_np(z)) - 2.5 * np.log10(pc.lmask)
                claim = valid & (r < rmask) & (g["REFMAG"][idx] < mlim)
                np.add.at(pgal, idx[claim], p[claim])
                _record(clusters, members, region, zlh, t, k, gi[k], idx, valid, pf_used, rmask,
                        cand, cenh, c1h[t], W, galaxy_centre)
            else:
                drop[why] += 1
            finish(k)
        if log_every and nround % log_every == 0:
            log.info("percolation: round %d, %d calls, %d clusters, %d skipped, %d dropped, "
                     "%d ready, %.1fs", nround, ncalls, len(clusters), nskip, sum(drop.values()),
                     len(ready), time.time() - t0)
    log.info("percolation: %d candidates, %d rounds, %d calls, %d clusters, %d skipped, "
             "%d dropped (%s), %.1fs", ncand, nround, ncalls, len(clusters), nskip,
             sum(drop.values()), ", ".join(f"{k} {v}" for k, v in drop.items()), time.time() - t0)
    if info is not None:
        info.update(n_dependencies=ndep, rounds=nround, calls=ncalls,
                    n_percolated=len(clusters), skipped=nskip, dropped=dict(drop),
                    t_percolation=time.time() - t0)
    # Rank order (the order of a sequential run).
    if clusters:
        srt = np.argsort([rank[c["_K"]] for c in clusters], kind="stable")
        clusters = [clusters[i] for i in srt]
        old = {c["RANK"]: new for new, c in enumerate(clusters)}
        for new, c in enumerate(clusters):
            c["RANK"] = new
            c.pop("_K")
        for m in members:
            m["RANK"] = np.full(m["RANK"].size, old[int(m["RANK"][0])] if m["RANK"].size else 0)
    cat = {k: np.array([c[k] for c in clusters]) for k in clusters[0]} if clusters else {}
    mem = {k: np.concatenate([m[k] for m in members]) for k in members[0]} if members else {}
    return cat, mem


def _theta_for(region: Region, ra, dec, idx, valid):
    """Angular separations [deg] between centres (ra, dec) and neighbours idx (0 where invalid)."""
    xc = unit_vectors(ra, dec)
    xg = unit_vectors(region.gal["RA"][idx], region.gal["DEC"][idx])
    d = np.linalg.norm(xg - xc[:, None, :], axis=-1)
    return np.where(valid, np.degrees(2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))), 0.0)


def _crop(idx, valid, K):
    """Keep the valid neighbours of each row first, cropped to K columns."""
    order = np.argsort(~valid, axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, axis=1)[:, :K]
    val = np.take_along_axis(valid, order, axis=1)[:, :K]
    return np.where(val, idx, 0), val, np.zeros(idx.shape)


def _record(clusters, members, region, zl, j, k, s, idx, valid, pf_used, rmask, cand, cen, c1,
            W, galaxy_centre: bool = True):
    """Append one accepted cluster (element j of the host arrays ``zl`` and ``cen``) and its members.

    ``s`` is the seed galaxy and ``idx``/``valid`` its neighbours (galaxy-table rows), ``cen`` the
    centring, ``c1`` the centre's unit vector (used for a random centre, which is no galaxy), and
    ``W`` the connectivity of the centre. Members are the galaxies with pmem >= member_pmin plus
    the centre candidates (CENT_RANK 0..maxcen-1), as in redMaPPer.
    """
    g = region.gal
    pc = region.cfg.percolation
    rich = zl.rich
    take = lambda a: np.asarray(a)[j]
    lam, z = float(take(rich.lam)), float(take(zl.z))
    ci = np.asarray(cen.index[j])                         # [maxcen] neighbour slots, -1 unused
    used = ci >= 0
    rows = np.where(used, idx[np.maximum(ci, 0)], 0)       # galaxy-table rows of the candidates
    ic = int(ci[0])
    cg = int(rows[0]) if galaxy_centre and ic >= 0 else s  # central galaxy (seed for a random centre)
    if galaxy_centre:
        ra, dec = float(g["RA"][cg]), float(g["DEC"][cg])
    else:
        ra, dec = radec_from_unit(c1)
    clusters.append({
        "SEED_ID": g["ID"][s], "ID_CENT": np.where(used, g["ID"][rows], -1).astype(np.int64),
        "RA": ra, "DEC": dec,
        "RA_CENT": np.where(used, g["RA"][rows], -400.0), "DEC_CENT": np.where(used, g["DEC"][rows], -400.0),
        "RA_SEED": g["RA"][s], "DEC_SEED": g["DEC"][s], "Z_INIT": float(region.seed_z(s)),
        "LAMBDA": lam, "LAMBDA_E": float(take(rich.lam_e)), "Z_LAMBDA": z,
        "Z_LAMBDA_E": float(take(zl.z_e)), "Z_LAMBDA_NITER": int(take(zl.niter)),
        "R_LAMBDA": float(take(rich.r_lambda)), "R_MASK": rmask,
        "SCALEVAL": float(take(rich.scaleval)), "MASKFRAC": float(take(rich.maskfrac)),
        "LNLAMLIKE": float(take(rich.lnlamlike)),
        "LNCGLIKE": float(cand["LNCGLIKE"][k]) if "LNCGLIKE" in cand else 0.0,
        "LNLIKE": float(cand["LNLIKE"][k]),
        "PZBINS": take(zl.pzbins).astype(np.float32), "PZ": take(zl.pz).astype(np.float32),
        "REFMAG": float(g["REFMAG"][cg]), "REFMAG_ERR": float(g["REFMAG_ERR"][cg]),
        "ZRED": float(g["ZRED"][cg]), "ZRED_E": float(g["ZRED_E"][cg]),
        "ZRED_CHISQ": float(g["ZRED_CHISQ"][cg]),
        **{k: float(v) for k, v in region.photoz_columns(cg).items()},
        "CHISQ": float(take(rich.chisq)[ic]) if galaxy_centre and ic >= 0 else np.nan,
        "NCENT_GOOD": np.int16(take(cen.ngood)), "P_CEN": take(cen.p_cen).astype(np.float32),
        "Q_CEN": take(cen.q_cen).astype(np.float32), "P_SAT": take(cen.p_sat).astype(np.float32),
        "P_FG": take(cen.p_fg).astype(np.float32), "P_C": take(cen.p_c).astype(np.float32),
        "Q_MISS": float(take(cen.q_miss)), "W": float(W),
        "RANK": len(clusters), "_K": k,
    })
    pmem = take(rich.pmem)
    cent_rank = np.full(idx.size, -1, np.int16)
    cent_rank[ci[used]] = np.arange(int(used.sum()), dtype=np.int16)
    msel = valid & ((pmem >= pc.member_pmin) | (cent_rank >= 0))
    mi = idx[msel]
    members.append({
        "RANK": np.full(mi.size, len(clusters) - 1), "ID": g["ID"][mi], "RA": g["RA"][mi],
        "DEC": g["DEC"][mi], "Z": np.full(mi.size, z, np.float32),
        "R": take(rich.r)[msel].astype(np.float32), "P": take(rich.p)[msel].astype(np.float32),
        "PFREE": pf_used[msel].astype(np.float32), "PCOL": take(rich.pcol)[msel].astype(np.float32),
        "THETA_I": take(rich.theta_i)[msel].astype(np.float32),
        "THETA_R": take(rich.theta_r)[msel].astype(np.float32), "PMEM": pmem[msel].astype(np.float32),
        "CHISQ": take(rich.chisq)[msel].astype(np.float32),
        "REFMAG": g["REFMAG"][mi], "REFMAG_ERR": g["REFMAG_ERR"][mi], "ZRED": g["ZRED"][mi],
        "ZRED_E": g["ZRED_E"][mi], "ZSPEC": g["ZSPEC"][mi] if "ZSPEC" in g else np.full(mi.size, -1.0),
        **region.photoz_columns(mi), "FLUX": g["FLUX"][mi], "FLUX_IVAR": g["FLUX_IVAR"][mi], "CG": g["ID"][mi] == g["ID"][cg],
        "CENT_RANK": cent_rank[msel],
    })


# --------------------------------------------------------------------------- driver
class _Checkpoint:
    """Per-stage results of a blind run (npz files), so an interrupted run resumes.

    Every file carries a key computed from the inputs: the galaxy IDs (all, in table order) and
    their zred and zred chi^2 (which carry the zred correction; and their photo-z with the photo-z
    filter), the seeds, the red-sequence model, the backgrounds, the wcen model, the footprint,
    the configuration and the rema version. A file with another key (other inputs) is ignored and overwritten.
    """

    def __init__(self, directory, key: str):
        from pathlib import Path

        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.key = key

    @classmethod
    def for_run(cls, directory, region: Region, seeds: np.ndarray) -> "_Checkpoint":
        import hashlib

        from .. import __version__

        h = hashlib.sha1()
        cols = ("ID", "ZRED", "ZRED_CHISQ")
        if region.model.filter == "photoz":
            cols += ("ZPHOT", "ZPHOT_E")
        for col in cols:
            h.update(np.ascontiguousarray(region.gal[col]).tobytes())
        h.update(np.ascontiguousarray(region.gal["ID"][seeds]).tobytes())
        for leaf in jax.tree_util.tree_leaves((region.rs, region.model.bkg, region.model.pzbkg,
                                               region.wcen)):
            h.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())
        h.update((region.footprint.digest() if region.footprint is not None else "none").encode())
        h.update(region.cfg.to_yaml().encode())
        h.update(__version__.encode())
        return cls(directory, h.hexdigest())

    def load(self, name: str) -> dict | None:
        f = self.dir / f"{name}.npz"
        if not f.exists():
            return None
        with np.load(f) as d:
            if str(d["_KEY"]) != self.key:
                log.info("%s was computed from other inputs; recomputing", f)
                return None
            log.info("%s: read from %s", name, f)
            return {k: d[k] for k in d.files if k != "_KEY"}

    def save(self, name: str, arrays: dict) -> None:
        f = self.dir / f"{name}.npz"
        tmp = self.dir / f"{name}.tmp.npz"
        np.savez(tmp, _KEY=np.array(self.key), **arrays)
        tmp.replace(f)


def run_blind(region: Region, own: Box | None = None, region_id: int = 0, batch: int = 1024,
              checkpoint=None, info: dict | None = None):
    """Blind run over the region; returns (catalogue, members) dicts.

    ``checkpoint``: directory for the first-pass and likelihood results; a rerun with the same
    inputs resumes from them. ``info``: a dict that receives the run statistics (galaxies,
    seeds, candidates, dependencies, clusters, the time of each stage and the peak memory).
    """
    cfg = region.cfg
    rc = cfg.richness
    t0 = time.time()
    seeds = select_seeds(region)
    log.info("seeds: %d", seeds.size)
    stats = {"n_galaxies": int(region.gal["ID"].size), "n_seeds": int(seeds.size)}
    if seeds.size == 0:
        log.warning("no seed galaxy: empty catalogue")
        if info is not None:
            info.update(stats, n_clusters=0, t_total=time.time() - t0)
        return {}, {}
    ck = _Checkpoint.for_run(checkpoint, region, seeds) if checkpoint else None
    st_fp = Stage.make(rc.firstpass.r0, rc.firstpass.beta, rc.maxrad_factor)
    st_lk = Stage.make(rc.likelihoods.r0, rc.likelihoods.beta, rc.maxrad_factor)
    st_pc = Stage.make(rc.percolation.r0, rc.percolation.beta, rc.maxrad_factor)
    q_fp = RadialQuad.make(rmax=float(st_fp.maxrad) + 5 * cfg.model.rsig)
    q_pc = RadialQuad.make(rmax=float(rc.percolation.r0) * 20.0 ** rc.percolation.beta + 5 * cfg.model.rsig)

    cand = ck.load("likelihood") if ck else None
    if cand is None:
        fp = ck.load("firstpass") if ck else None
        if fp is None:
            z0 = region.seed_z(seeds)
            fp = _run_batched(region, seeds, z0, st_fp, q_fp, "zlambda", batch, calc_err=False)
            for _ in range(max(0, rc.firstpass_niter - 1)):
                fp = _run_batched(region, seeds, np.where(fp["Z_LAMBDA"] > 0, fp["Z_LAMBDA"], z0),
                                  st_fp, q_fp, "zlambda", batch, calc_err=False)
            if ck:
                ck.save("firstpass", fp)
        zr = cfg.model.zrange
        keep = ((fp["LAMBDA"] >= rc.minlambda) & (fp["Z_LAMBDA"] >= zr[0]) & (fp["Z_LAMBDA"] <= zr[1]))
        if rc.min_lnlamlike is not None:
            keep &= fp["LNLAMLIKE"] >= rc.min_lnlamlike
        gi = seeds[keep]
        zfp = fp["Z_LAMBDA"][keep]
        log.info("first pass: %d candidates (%.1fs)", gi.size, time.time() - t0)
        stats.update(n_firstpass=int(gi.size), t_firstpass=time.time() - t0)

        if gi.size:
            lk = _run_batched(region, gi, zfp, st_lk, q_pc, "richness", batch)
            # LNLIKE = LNLAMLIKE + LNCGLIKE (0 without a wcen model); non-finite rows are dropped.
            lnlike = lk["LNLAMLIKE"].astype(np.float64) + lk["LNCGLIKE"].astype(np.float64)
            keep = (lk["LAMBDA"] >= rc.minlambda) & np.isfinite(lnlike) & (lk["NINCUT"] >= 3)
            if rc.min_lnlamlike is not None:
                keep &= lk["LNLAMLIKE"] >= rc.min_lnlamlike
            cand = {"GI": gi[keep], "Z_LAMBDA": zfp[keep], "LAMBDA": lk["LAMBDA"][keep],
                    "LNLIKE": lnlike[keep], "LNCGLIKE": lk["LNCGLIKE"][keep]}
        else:
            cand = {"GI": gi, "Z_LAMBDA": zfp, "LAMBDA": np.zeros(0, np.float32),
                    "LNLIKE": np.zeros(0), "LNCGLIKE": np.zeros(0, np.float32)}
        if ck:
            ck.save("likelihood", cand)
    log.info("likelihood: %d candidates (%.1fs)", cand["GI"].size, time.time() - t0)
    stats.update(n_candidates=int(cand["GI"].size), t_likelihood=time.time() - t0)

    cat, mem = percolate(region, cand, st_pc, q_pc, info=stats) if cand["GI"].size else ({}, {})
    log.info("percolation: %d clusters (%.1fs)", len(cat.get("LAMBDA", [])), time.time() - t0)
    out, mem = consolidate(region, cat, mem, own, region_id)
    if info is not None:
        import resource

        stats.update(n_clusters=len(out.get("LAMBDA", [])), t_total=time.time() - t0,
                     peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        info.update(stats)
    return out, mem


def consolidate(region: Region, cat: dict, mem: dict, own: Box | None = None, region_id: int = 0):
    cfg = region.cfg
    if not cat:
        return {}, {}
    lam, sv = cat["LAMBDA"], cat["SCALEVAL"]
    keep = (lam >= cfg.richness.minlambda) & (lam / np.maximum(sv, 1e-6) >= cfg.richness.minlambda)
    keep &= cat["MASKFRAC"] < cfg.mask.max_maskfrac
    if own is not None:
        keep &= own.contains(cat["RA"], cat["DEC"])
    rank = cat["RANK"][keep]
    out = {k: v[keep] for k, v in cat.items()}
    out["Z_LAMBDA_RAW"] = out["Z_LAMBDA"].copy()
    out["Z_LAMBDA_E_RAW"] = out["Z_LAMBDA_E"].copy()
    out["Z_LAMBDA"], out["Z_LAMBDA_E"] = region.correct_zlambda(out["Z_LAMBDA"], out["Z_LAMBDA_E"],
                                                                out["LAMBDA"])
    out["SNR"] = snr(out["LNLAMLIKE"])
    srt = np.argsort(-out["LAMBDA"], kind="stable")
    out = {k: v[srt] for k, v in out.items()}
    mmid = (np.int64(region_id) << 32) | np.arange(srt.size, dtype=np.int64)
    out["MEM_MATCH_ID"] = mmid
    lookup = dict(zip(out["RANK"], mmid))
    msel = np.isin(mem["RANK"], rank)
    m = {k: v[msel] for k, v in mem.items()}
    m["MEM_MATCH_ID"] = np.array([lookup[r] for r in m["RANK"]], dtype=np.int64)
    if region.model.filter == "photoz" and m["ID"].size:
        # colour of the members: red-sequence chi^2 at the cluster redshift
        m["CHISQ_RS"] = region.rs_chisq(m["FLUX"], m["FLUX_IVAR"], m["Z"])
    m.pop("RANK")
    out.pop("RANK")
    return out, m
