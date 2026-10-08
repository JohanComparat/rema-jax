"""Scan mode: richness and redshift at given positions (redMaPPer "zscan").

For every input position (e.g. X-ray or SZ cluster candidates):

1. lambda(z) and the likelihood LNLAMLIKE(z) = -lambda/S - sum ln(1 - pmem) are computed on a
   regular redshift grid (``scan.zrange``, ``scan.zstep``) in a fixed aperture
   (``richness.zscan``: r0 = 0.5 h^-1 Mpc, beta = 0) centred on the input position;
2. ZMAX is the grid redshift of maximum likelihood (ZMAX_EDGE flags a maximum on the grid edge);
3. starting from ZMAX, z_lambda and lambda are refined with the percolation aperture
   (r0 = 1, beta = 0.2) at the input position;
4. an optical centre is chosen within ``centering.scan_maxrad`` h^-1 Mpc of the input position
   (wcen centring when calibrated, else BCG; the up to ``maxcen`` candidates and their
   probabilities are recorded) and lambda, z_lambda are recomputed there (LAMBDA_OPT,
   Z_LAMBDA_OPT). A position without an optical centre is kept (RA_OPT = RA).

Neighbours are gathered per redshift chunk of the scan, keeping only galaxies brighter than the
chunk's faintest relevant magnitude, so low-redshift steps (large apertures) stay cheap.
"""

from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..core.centering import as_centering, center_bcg, center_wcen
from ..core.richness import RadialQuad, Stage, richness_one
from ..core.zlambda import zlambda
from ..sky.neighbors import next_pow2, unit_vectors
from .common import Region, snr

log = logging.getLogger(__name__)


@partial(jax.jit, static_argnames=())
def _scan_kernel(nb, zsteps, frad, fgeo, quad, model, stage):
    """lambda and LNLAMLIKE for B clusters x S redshift steps (frad, fgeo: [B, S, n])."""
    def one_cluster(nb1, fr, fg):
        def one_step(z, fr1, fg1):
            r = richness_one(nb1, z, fr1, fg1, quad, model, stage)
            return r.lam, r.lnlamlike
        return jax.vmap(one_step)(zsteps, fr, fg)

    return jax.vmap(one_cluster)(nb, frad, fgeo)


def _z_chunks(zsteps: np.ndarray, zstep: float, width: float = 0.1) -> list[np.ndarray]:
    """Consecutive runs of the scan grid, ``width`` in redshift each; every step exactly once.

    >>> z = np.round(np.arange(0.05, 1.0 + 0.0025, 0.005), 6)
    >>> c = _z_chunks(z, 0.005)
    >>> [x.size for x in c], bool(np.array_equal(np.concatenate(c), z))
    ([20, 20, 20, 20, 20, 20, 20, 20, 20, 11], True)
    """
    n = max(1, int(round(width / zstep)))
    return [zsteps[i:i + n] for i in range(0, zsteps.size, n)]


def _completeness_grid(region: Region, ra, dec, zsteps, quad):
    B, S = np.size(ra), np.size(zsteps)
    rr = np.repeat(np.asarray(ra), S)
    dd = np.repeat(np.asarray(dec), S)
    zz = np.tile(np.asarray(zsteps), B)
    fr, fg = region.completeness(rr, dd, zz, quad)
    n = quad.r.shape[0]
    return jnp.reshape(fr, (B, S, n)), jnp.reshape(fg, (B, S, n))


def _refine(region: Region, ra, dec, z0, stage: Stage, quad: RadialQuad):
    """z_lambda iteration around (ra, dec) starting at z0; returns (zl, pad, nb).

    Every position keeps its own aperture and magnitude limit (galaxies fainter than
    maxmag + 0.5 have theta_i ~ 0), and the neighbour arrays are cropped to the largest count:
    one low-redshift position (wide aperture) must not inflate the arrays of the whole batch
    with the faint limit of a high-redshift one.
    """
    cfg = region.cfg
    z0 = np.asarray(z0, np.float64)
    zlo = np.maximum(z0 - 0.05, cfg.scan.zrange[0])
    rad = region.radius_deg(float(stage.maxrad), zlo)
    mlim = region.maxmag(z0 + 0.05) + 0.5
    pad = region.query(ra, dec, rad, mag_max=float(np.max(mlim)))
    pad.valid &= region.gal["REFMAG"][pad.idx] < mlim[:, None]
    pad = _crop_padded(pad)
    nb = region.neighbors(pad)
    fr, fg = region.completeness(ra, dec, z0, quad)
    zc = cfg.zlambda
    zl = zlambda(nb, jnp.asarray(z0, jnp.float32), fr, fg, quad, region.model, stage,
                 maxiter=zc.maxiter, tol=zc.tol, ngrid=zc.ngrid, half_width=zc.half_width,
                 npz=zc.npzbins, topfrac=zc.topfrac, soft=zc.pcol_soft)
    return zl, pad, nb


def _crop_padded(pad):
    """Valid neighbours first in every row, cropped to a power of two >= the largest count."""
    order = np.argsort(~pad.valid, axis=1, kind="stable")
    counts = pad.valid.sum(axis=1)
    K = next_pow2(int(counts.max(initial=1)))
    take = lambda a: np.take_along_axis(a, order, axis=1)[:, :K]
    valid = take(pad.valid)
    return type(pad)(np.where(valid, take(pad.idx), 0), valid, np.where(valid, take(pad.theta), 0.0),
                     counts)


def run_scan(region: Region, ra, dec, ids=None, batch: int = 128):
    """Scan the given positions. Returns (catalogue, members) as dicts of numpy arrays."""
    cfg = region.cfg
    ra = np.atleast_1d(np.asarray(ra, np.float64))
    dec = np.atleast_1d(np.asarray(dec, np.float64))
    ids = np.arange(ra.size) if ids is None else np.atleast_1d(np.asarray(ids))
    sc = cfg.scan
    zsteps = np.round(np.arange(sc.zrange[0], sc.zrange[1] + sc.zstep / 2, sc.zstep), 6)
    st_scan = Stage.make(cfg.richness.zscan.r0, cfg.richness.zscan.beta, cfg.richness.maxrad_factor)
    st_perc = Stage.make(cfg.richness.percolation.r0, cfg.richness.percolation.beta,
                         cfg.richness.maxrad_factor)
    st_cen = Stage(st_perc.r0, st_perc.beta, st_perc.maxrad + cfg.centering.scan_maxrad)
    q_scan = RadialQuad.make(rmax=float(st_scan.maxrad) + 5 * cfg.model.rsig)
    q_perc = RadialQuad.make(rmax=float(st_perc.r0) * 20.0 ** float(st_perc.beta) + 5 * cfg.model.rsig)
    chunks = _z_chunks(zsteps, sc.zstep)

    cats, mems = [], []
    for b0 in range(0, ra.size, batch):
        sl = slice(b0, min(ra.size, b0 + batch))
        bra, bdec, bid = ra[sl], dec[sl], ids[sl]
        lam_steps, like_steps = [], []
        for zc in chunks:
            rad = region.radius_deg(float(st_scan.maxrad), zc.min())
            pad = region.query(bra, bdec, rad, mag_max=float(region.maxmag(zc.max())) + 1.0)
            nb = region.neighbors(pad)
            fr, fg = _completeness_grid(region, bra, bdec, zc, q_scan)
            lam, like = _scan_kernel(nb, jnp.asarray(zc, jnp.float32), fr, fg, q_scan, region.model, st_scan)
            lam_steps.append(np.asarray(lam))
            like_steps.append(np.asarray(like))
        lam_steps = np.concatenate(lam_steps, axis=1)
        like_steps = np.concatenate(like_steps, axis=1)
        like_ok = np.where(lam_steps > 0, like_steps, -np.inf)
        imax = np.argmax(like_ok, axis=1)
        has = np.isfinite(like_ok[np.arange(imax.size), imax])
        zmax = np.where(has, zsteps[imax], -1.0)
        zmax_edge = has & ((imax == 0) | (imax == zsteps.size - 1))

        # Refine at the input position. Its membership probabilities feed the centring, whose
        # candidates lie up to scan_maxrad away: p is computed that much further out.
        z0 = np.where(has, zmax, 0.3)
        zl, pad, nb = _refine(region, bra, bdec, z0, st_cen, q_perc)
        rich = zl.rich
        z_l = np.where(has, np.asarray(zl.z), -1.0)

        # Optical centre within scan_maxrad of the input position.
        cen = _centre(region, pad, nb, rich, zl.z)
        ic = cen.index[:, 0]
        found = (cen.ngood > 0) & (ic >= 0) & has
        g = region.gal
        rows = np.where(cen.index >= 0, np.take_along_axis(pad.idx, np.maximum(cen.index, 0), axis=1), 0)
        ra_opt = np.where(found, g["RA"][rows[:, 0]], bra)
        dec_opt = np.where(found, g["DEC"][rows[:, 0]], bdec)
        cslot = (cen.index >= 0) & has[:, None]
        zl_o, _, _ = _refine(region, ra_opt, dec_opt, np.where(z_l > 0, z_l, z0), st_perc, q_perc)

        n = bra.size
        z_l_raw, z_e_raw = z_l.copy(), np.asarray(zl.z_e, np.float64)
        z_l, z_e = region.correct_zlambda(z_l_raw, z_e_raw, np.asarray(rich.lam))
        zo_raw, zoe_raw = np.asarray(zl_o.z, np.float64), np.asarray(zl_o.z_e, np.float64)
        zo, zoe = region.correct_zlambda(zo_raw, zoe_raw, np.asarray(zl_o.rich.lam))
        cat = {
            "MEM_MATCH_ID": bid, "RA": bra, "DEC": bdec,
            "LAMBDA": np.where(has, np.asarray(rich.lam), -1.0).astype(np.float32),
            "LAMBDA_E": np.asarray(rich.lam_e, np.float32),
            "Z_LAMBDA": z_l.astype(np.float32), "Z_LAMBDA_E": z_e.astype(np.float32),
            "Z_LAMBDA_RAW": z_l_raw.astype(np.float32), "Z_LAMBDA_E_RAW": z_e_raw.astype(np.float32),
            "R_LAMBDA": np.asarray(rich.r_lambda, np.float32),
            "SCALEVAL": np.asarray(rich.scaleval, np.float32),
            "MASKFRAC": np.asarray(rich.maskfrac, np.float32),
            "LNLAMLIKE": np.asarray(rich.lnlamlike, np.float32),
            "SNR": snr(rich.lnlamlike),
            "PZBINS": np.asarray(zl.pzbins, np.float32), "PZ": np.asarray(zl.pz, np.float32),
            "Z_STEPS": np.broadcast_to(zsteps.astype(np.float32), (n, zsteps.size)).copy(),
            "LAMBDA_STEPS": lam_steps.astype(np.float32),
            "LIKELIHOOD_STEPS": like_steps.astype(np.float32),
            "ZMAX": zmax.astype(np.float32), "MAX_IND": imax.astype(np.int32),
            "LMAX": lam_steps[np.arange(n), imax].astype(np.float32), "ZMAX_EDGE": zmax_edge,
            "IN_CALIB_ZRANGE": (z_l >= cfg.model.zrange[0]) & (z_l <= cfg.model.zrange[1]),
            "ID_CENT": np.where(cslot, g["ID"][rows], -1).astype(np.int64),
            "RA_CENT": np.where(cslot, g["RA"][rows], -400.0),
            "DEC_CENT": np.where(cslot, g["DEC"][rows], -400.0),
            "NCENT_GOOD": np.where(has, cen.ngood, 0).astype(np.int16),
            "P_CEN": np.where(has[:, None], cen.p_cen, 0).astype(np.float32),
            "Q_CEN": np.where(has[:, None], cen.q_cen, 0).astype(np.float32),
            "P_SAT": np.where(has[:, None], cen.p_sat, 0).astype(np.float32),
            "P_FG": np.where(has[:, None], cen.p_fg, 0).astype(np.float32),
            "P_C": np.where(has[:, None], cen.p_c, 0).astype(np.float32),
            "Q_MISS": np.where(has, cen.q_miss, 1.0).astype(np.float32),
            "RA_OPT": ra_opt, "DEC_OPT": dec_opt,
            "LAMBDA_OPT": np.asarray(zl_o.rich.lam, np.float32),
            "LAMBDA_OPT_E": np.asarray(zl_o.rich.lam_e, np.float32),
            "Z_LAMBDA_OPT": zo.astype(np.float32), "Z_LAMBDA_OPT_E": zoe.astype(np.float32),
            "Z_LAMBDA_OPT_RAW": zo_raw.astype(np.float32),
        }
        cats.append(cat)
        mems.append(member_table(region, bid, pad, rich, np.asarray(zl.z), cfg.percolation.member_pmin))
    return _concat(cats), _concat(mems)


def _centre(region: Region, pad, nb, rich, z):
    """Centring (host arrays) of a batch of scan positions, within scan_maxrad of each position."""
    cfg = region.cfg
    maxrad = jnp.float32(cfg.centering.scan_maxrad)
    if region.centering_method() == "wcen":
        g = region.gal
        xyz = jnp.asarray(unit_vectors(g["RA"][pad.idx], g["DEC"][pad.idx]), jnp.float32)
        zchi = jnp.asarray(np.where(pad.valid, g["ZRED_CHISQ"][pad.idx], -1.0), jnp.float32)
        cen = center_wcen(nb, xyz, zchi, rich, z, maxrad, region.model, region.wcen)
    else:
        ic, found = center_bcg(nb, rich.pmem, rich.r, rich.r_lambda, z, maxrad,
                               use_zphot=region.model.filter == "photoz")
        cen = as_centering(jnp.where(found, ic, -1).astype(jnp.int32), found, cfg.centering.maxcen)
    return jax.tree_util.tree_map(np.asarray, cen)


def member_table(region: Region, match_ids, pad, rich, z, pmin: float = 0.01) -> dict:
    """Members (pmem >= pmin) of a batch of clusters."""
    pmem = np.asarray(rich.pmem)
    sel = pad.valid & (pmem >= pmin)
    b, k = np.nonzero(sel)
    gi = pad.idx[b, k]
    g = region.gal
    out = {"MEM_MATCH_ID": np.asarray(match_ids)[b], "ID": g["ID"][gi], "RA": g["RA"][gi],
           "DEC": g["DEC"][gi], "Z": np.asarray(z)[b].astype(np.float32),
           "R": np.asarray(rich.r)[b, k].astype(np.float32),
           "P": np.asarray(rich.p)[b, k].astype(np.float32),
           "PMEM": pmem[b, k].astype(np.float32),
           "PCOL": np.asarray(rich.pcol)[b, k].astype(np.float32),
           "THETA_I": np.asarray(rich.theta_i)[b, k].astype(np.float32),
           "THETA_R": np.asarray(rich.theta_r)[b, k].astype(np.float32),
           "CHISQ": np.asarray(rich.chisq)[b, k].astype(np.float32),
           "REFMAG": g["REFMAG"][gi], "REFMAG_ERR": g["REFMAG_ERR"][gi],
           "ZRED": g["ZRED"][gi], "ZRED_E": g["ZRED_E"][gi]}
    if "ZSPEC" in g:
        out["ZSPEC"] = g["ZSPEC"][gi]
    out.update(region.photoz_columns(gi))
    out["FLUX"] = g["FLUX"][gi]
    out["FLUX_IVAR"] = g["FLUX_IVAR"][gi]
    return out


def _concat(parts):
    parts = [p for p in parts if p]
    if not parts:
        return {}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
