"""Spectroscopic post-processing: cluster redshift and velocity dispersion from members.

The procedure is the velocity clipping of Clerc et al. (2016), vectorised over clusters (members
with a spectroscopic redshift padded to a common length):

1. clusters need >= ``min_members`` (3) members with ZSPEC > 0;
2. z0 = biweight location (c = 6) of their redshifts; members with
   |v| = c |z - z0| / (1 + z0) <= ``vmax_init`` (5000 km/s) are kept;
3. ``niter`` times (20): z_c = biweight location of the kept members, v = c (z - z_c)/(1 + z_c),
   sigma = gapper estimator (fewer than ``gapper_nmax`` = 15 members) or biweight scale (c = 9),
   and the members are re-selected among *all* spectroscopic members by |v| <= 3 sigma (a member
   clipped earlier can come back). Fewer than 3 kept members is a failure. The values of the last
   iteration are returned (no convergence test), with VDISP_FLAG = 3 sigma > 5000 km/s;
4. ``nboot`` bootstrap resamples of the spectroscopic members repeat 1-3: SPEC_Z_BOOT and
   VDISP_BOOT are the means, SPEC_ZERR_BOOT and VDISP_ERR_BOOT the standard deviations over the
   successful resamples (seeded, hence reproducible);
5. BEST_Z = SPEC_Z_BOOT if available and not flagged, else the central galaxy's ZSPEC
   (CG_SPEC_Z), else Z_LAMBDA (BEST_Z_TYPE says which).

Members get VEL (rest-frame velocity) and ISMEMBER_SPEC (kept by the final clipping).
"""

from __future__ import annotations

import warnings
from functools import wraps

import numpy as np


def _quiet(fn):
    """Silence numpy's all-NaN / empty-slice warnings (clusters without enough spectra)."""
    @wraps(fn)
    def wrapper(*a, **k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return fn(*a, **k)
    return wrapper

C_KMS = 299792.458

BEST_Z_TYPES = ("spec_z_boot", "cg_spec_z", "photo_z", "none")


# --------------------------------------------------------------------------- estimators (masked rows)
def _masked(x, mask):
    return np.where(mask, x, np.nan)


@_quiet
def biweight_location(x, mask, c: float = 6.0):
    """Row-wise biweight location of x[mask] (astropy convention: M = median, MAD scale)."""
    xm = _masked(x, mask)
    with np.errstate(all="ignore"):
        M = np.nanmedian(xm, axis=-1)
        d = xm - M[..., None]
        mad = np.nanmedian(np.abs(d), axis=-1)
        u = d / (c * mad[..., None])
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        w = np.where(mask, w, 0.0)
        num = np.nansum(w * np.nan_to_num(d), axis=-1)
        den = np.sum(w, axis=-1)
        loc = M + num / den
    return np.where(mad > 0, loc, M)


@_quiet
def biweight_scale(x, mask, c: float = 9.0):
    """Row-wise biweight scale of x[mask] (astropy ``biweight_scale``, modify_sample_size=False)."""
    xm = _masked(x, mask)
    n = mask.sum(axis=-1)
    with np.errstate(all="ignore"):
        M = np.nanmedian(xm, axis=-1)
        d = xm - M[..., None]
        mad = np.nanmedian(np.abs(d), axis=-1)
        u = d / (c * mad[..., None])
        inside = mask & (np.abs(u) < 1)
        d2 = np.where(inside, np.nan_to_num(d) ** 2 * (1 - u**2) ** 4, 0.0)
        num = n * np.sum(d2, axis=-1)
        den = np.sum(np.where(inside, (1 - u**2) * (1 - 5 * u**2), 0.0), axis=-1) ** 2
        scale = np.sqrt(num / den)
    return np.where(mad > 0, scale, 0.0)


@_quiet
def gapper(x, mask):
    """Row-wise gapper scale (Wainer & Thissen 1976): sqrt(pi)/(n(n-1)) sum i(n-i) g_i."""
    xs = np.sort(_masked(x, mask), axis=-1)            # NaNs last
    n = mask.sum(axis=-1)
    g = np.diff(xs, axis=-1)                           # NaN beyond the valid range
    i = np.arange(1, xs.shape[-1])
    w = i[None, :] * (n[:, None] - i[None, :]) if xs.ndim == 2 else i * (n - i)
    with np.errstate(all="ignore"):
        s = np.nansum(np.where(np.isfinite(g), w * g, 0.0), axis=-1)
        return np.sqrt(np.pi) / (n * (n - 1)) * s


def velocities(z, zc):
    return C_KMS * (z - zc[..., None]) / (1 + zc[..., None])


# --------------------------------------------------------------------------- clipping
def clip_velocity_batch(z, valid, *, nsig: float = 3.0, vmax_init: float = 5000.0,
                        niter: int = 20, gapper_nmax: int = 15, c_location: float = 6.0,
                        c_scale: float = 9.0, min_members: int = 3):
    """Velocity clipping for C clusters with padded member redshifts ``z`` [C, M].

    Returns a dict of arrays [C] (ok, zspec, vdisp, vclip, vflag, gapper_used, n_members) and
    ``v``, ``kept`` [C, M].
    """
    z = np.asarray(z, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(z)
    n0 = valid.sum(axis=1)
    ok = n0 >= min_members
    z0 = biweight_location(z, valid, c_location)
    v = velocities(z, z0)
    kept = valid & (np.abs(v) <= vmax_init)
    ok &= kept.sum(axis=1) > 2
    zc = z0
    vdisp = np.zeros(z.shape[0])
    vclip = np.full(z.shape[0], vmax_init)
    gap = np.zeros(z.shape[0], bool)
    for depth in range(1, niter + 1):
        nk = kept.sum(axis=1)
        zc = biweight_location(z, kept, c_location)
        v = velocities(z, zc)
        gap = nk < gapper_nmax
        vdisp = np.where(gap, gapper(v, kept), biweight_scale(v, kept, c_scale))
        vclip = nsig * vdisp
        if depth == niter:
            break
        new = valid & (np.abs(v) <= np.abs(vclip)[:, None])
        ok &= new.sum(axis=1) > 2
        kept = np.where(ok[:, None], new, kept)
    nkept = kept.sum(axis=1)
    return {"ok": ok & np.isfinite(zc) & np.isfinite(vdisp), "zspec": zc, "vdisp": vdisp,
            "vclip": vclip, "vflag": vclip > vmax_init, "gapper_used": gap,
            "n_members": nkept, "v": v, "kept": kept & ok[:, None]}


@_quiet
def bootstrap_clip(z, valid, nboot: int = 64, seed: int = 12345, **kw):
    """Bootstrap means and standard deviations of (zspec, vdisp, vclip, vflag)."""
    rng = np.random.default_rng(seed)
    C, M = z.shape
    n = valid.sum(axis=1)
    order = np.argsort(~valid, axis=1, kind="stable")          # valid entries first
    zs = np.take_along_axis(z, order, axis=1)
    acc = {k: np.zeros((nboot, C)) for k in ("zspec", "vdisp", "vclip", "vflag")}
    okb = np.zeros((nboot, C), bool)
    for b in range(nboot):
        u = rng.random((C, M))
        pick = np.minimum((u * np.maximum(n, 1)[:, None]).astype(np.int64), np.maximum(n - 1, 0)[:, None])
        zb = np.take_along_axis(zs, pick, axis=1)
        vb = np.arange(M)[None, :] < n[:, None]
        r = clip_velocity_batch(zb, vb, **kw)
        okb[b] = r["ok"]
        for k in acc:
            acc[k][b] = r[k]
    out = {}
    cnt = okb.sum(axis=0)
    for k in ("zspec", "vdisp", "vclip", "vflag"):
        a = np.where(okb, acc[k], np.nan)
        with np.errstate(all="ignore"):
            out[f"{k}_boot"] = np.where(cnt > 0, np.nanmean(a, axis=0), np.nan)
            out[f"{k}_boot_std"] = np.where(cnt > 1, np.nanstd(a, axis=0), np.nan)
    out["nboot_ok"] = cnt
    return out


# --------------------------------------------------------------------------- catalogue level
@_quiet
def process(cat: dict, mem: dict, *, min_members: int = 3, vmax_init: float = 5000.0,
            nsigma_clip: float = 3.0, niter: int = 20, gapper_nmax: int = 15,
            c_location: float = 6.0, c_scale: float = 9.0, nboot: int = 64,
            seed: int = 12345, zlambda_col: str = "Z_LAMBDA", center_id_col: str = "ID_CENT"):
    """Add spectroscopic columns to a cluster catalogue and its members (returns new dicts)."""
    cat = dict(cat)
    mem = dict(mem)
    ids = np.asarray(cat["MEM_MATCH_ID"])
    C = ids.size
    mid = np.asarray(mem["MEM_MATCH_ID"])
    zsp = np.asarray(mem.get("ZSPEC", np.full(mid.size, -1.0)), np.float64)
    has = zsp > 0
    # Pad spectroscopic members per cluster.
    pos = {v: i for i, v in enumerate(ids)}
    row = np.array([pos.get(v, -1) for v in mid])
    use = has & (row >= 0)
    counts = np.bincount(row[use], minlength=C)
    M = max(int(counts.max(initial=0)), 1)
    Z = np.full((C, M), np.nan)
    V = np.zeros((C, M), bool)
    back = np.full((C, M), -1, np.int64)
    fill = np.zeros(C, np.int64)
    for j in np.flatnonzero(use):
        r = row[j]
        Z[r, fill[r]] = zsp[j]
        V[r, fill[r]] = True
        back[r, fill[r]] = j
        fill[r] += 1
    kw = dict(nsig=nsigma_clip, vmax_init=vmax_init, niter=niter, gapper_nmax=gapper_nmax,
              c_location=c_location, c_scale=c_scale, min_members=min_members)
    res = clip_velocity_batch(Z, V, **kw)
    boot = bootstrap_clip(Z, V, nboot=nboot, seed=seed, **kw)
    ok = res["ok"]
    nan = np.full(C, np.nan)
    vdisp = np.where(ok, res["vdisp"], nan)
    zspec = np.where(ok, res["zspec"], nan)
    nmem = np.where(ok, res["n_members"], 0)
    with np.errstate(all="ignore"):
        ruel = vdisp / C_KMS * (1 + zspec) / np.sqrt(nmem)
    zboot = np.where(ok, boot["zspec_boot"], nan)
    cat.update({
        "NSPEC": counts.astype(np.int32), "N_MEMBERS": nmem.astype(np.int32),
        "SPEC_Z": zspec, "SPEC_ZERR_RUEL": ruel, "SPEC_Z_BOOT": zboot,
        "SPEC_ZERR_BOOT": np.where(ok, boot["zspec_boot_std"], nan),
        "VDISP": vdisp, "VDISP_ERR": np.where(ok, boot["vdisp_boot_std"], nan),
        "VDISP_BOOT": np.where(ok, boot["vdisp_boot"], nan),
        "VDISP_ERR_BOOT": np.where(ok, boot["vdisp_boot_std"], nan),
        "VDISP_CLIP": np.where(ok, res["vclip"], nan),
        "VDISP_FLAG": np.where(ok, res["vflag"], False),
        "VDISP_FLAG_BOOT": np.where(ok, boot["vflag_boot"], nan),
        "VDISP_TYPE": np.where(ok, np.where(res["gapper_used"], "gapper", "biweight"), ""),
        "BOOTNUM": np.full(C, nboot, np.int32),
    })
    # Central galaxy's spectroscopic redshift.
    cg = np.full(C, -1.0)
    if center_id_col in cat and "ID" in mem:
        lookup = dict(zip(np.asarray(mem["ID"])[has], zsp[has]))
        ids = np.asarray(cat[center_id_col])
        ids = ids[:, 0] if ids.ndim == 2 else ids          # [N, maxcen] candidates: the central
        cg = np.array([lookup.get(i, -1.0) for i in ids])
    cat["CG_SPEC_Z"] = cg
    zl = np.asarray(cat.get(zlambda_col, np.full(C, -1.0)), np.float64)
    zl_e = np.asarray(cat.get(f"{zlambda_col}_E", np.full(C, np.nan)), np.float64)
    use_spec = ok & np.isfinite(zboot) & (zboot > 0) & ~res["vflag"]
    use_cg = ~use_spec & (cg > 0)
    use_ph = ~use_spec & ~use_cg & (zl > 0)
    cat["BEST_Z"] = np.where(use_spec, zboot, np.where(use_cg, cg, np.where(use_ph, zl, -1.0)))
    cat["BEST_ZERR"] = np.where(use_spec, cat["SPEC_ZERR_BOOT"],
                                np.where(use_cg, 1e-4, np.where(use_ph, zl_e, np.nan)))
    cat["BEST_Z_TYPE"] = np.where(use_spec, "spec_z_boot", np.where(use_cg, "cg_spec_z",
                                  np.where(use_ph, "photo_z", "none")))
    # Members: velocities and spectroscopic membership.
    vel = np.full(mid.size, np.nan)
    kept = np.zeros(mid.size, bool)
    sel = back >= 0
    vel[back[sel]] = np.where(ok[:, None], res["v"], np.nan)[sel]
    kept[back[sel]] = res["kept"][sel]
    mem["VEL"] = vel
    mem["ISMEMBER_SPEC"] = kept
    return cat, mem
