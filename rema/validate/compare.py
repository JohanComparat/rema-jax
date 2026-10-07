"""Comparison of two cluster catalogues (e.g. rema vs redMaPPer DR10, or vs ACT/DES).

``match_catalogs`` pairs clusters one to one: candidate pairs within ``radius_arcmin`` and
|dz| < ``dz_max`` (1 + z) are taken in order of increasing separation. ``summary`` then reports,
for a richness threshold applied to each side, the fraction of the reference catalogue recovered
and the fraction of the test catalogue confirmed, the median and robust scatter of ln(lambda
ratio), and the median and NMAD of the redshift difference.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..sky.neighbors import unit_vectors


def match_catalogs(ra1, dec1, z1, ra2, dec2, z2, radius_arcmin: float = 1.0, dz_max: float = 0.05):
    """One-to-one matches between catalogue 1 and 2. Returns (i1, i2, separation_arcmin)."""
    x1, x2 = unit_vectors(ra1, dec1), unit_vectors(ra2, dec2)
    chord = 2 * np.sin(np.radians(radius_arcmin / 60.0) / 2)
    pairs = cKDTree(x2).query_ball_point(x1, chord)
    cand = []
    z1, z2 = np.asarray(z1, float), np.asarray(z2, float)
    for i, js in enumerate(pairs):
        for j in js:
            if abs(z1[i] - z2[j]) < dz_max * (1 + z2[j]):
                sep = np.degrees(2 * np.arcsin(np.linalg.norm(x1[i] - x2[j]) / 2)) * 60
                cand.append((sep, i, j))
    cand.sort()
    used1, used2 = set(), set()
    out = []
    for sep, i, j in cand:
        if i in used1 or j in used2:
            continue
        used1.add(i)
        used2.add(j)
        out.append((i, j, sep))
    if not out:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    a = np.array(out)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2]


def summary(test: dict, ref: dict, lam_min: float = 20.0, radius_arcmin: float = 1.0,
            dz_max: float = 0.05, lam_col_test: str = "LAMBDA", lam_col_ref: str = "LAMBDA",
            z_col_test: str = "Z_LAMBDA", z_col_ref: str = "Z_LAMBDA") -> dict:
    """Recovery and agreement statistics between a test and a reference catalogue."""
    i1, i2, sep = match_catalogs(test["RA"], test["DEC"], test[z_col_test], ref["RA"], ref["DEC"],
                                 ref[z_col_ref], radius_arcmin, dz_max)
    lt, lr = np.asarray(test[lam_col_test], float), np.asarray(ref[lam_col_ref], float)
    zt, zr = np.asarray(test[z_col_test], float), np.asarray(ref[z_col_ref], float)
    big_r = lr >= lam_min
    big_t = lt >= lam_min
    rec = np.isin(np.flatnonzero(big_r), i2).mean() if big_r.any() else np.nan
    conf = np.isin(np.flatnonzero(big_t), i1).mean() if big_t.any() else np.nan
    both = big_r[i2] | big_t[i1]
    lnr = np.log(lt[i1][both] / lr[i2][both])
    dz = (zt[i1][both] - zr[i2][both]) / (1 + zr[i2][both])

    def nmad(x):
        return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size else np.nan

    return {"n_ref": int(big_r.sum()), "n_test": int(big_t.sum()), "n_matched": int(both.sum()),
            "recovered": float(rec), "confirmed": float(conf),
            "median_lnlam_ratio": float(np.median(lnr)) if lnr.size else np.nan,
            "nmad_lnlam_ratio": float(nmad(lnr)), "median_dz": float(np.median(dz)) if dz.size else np.nan,
            "nmad_dz": float(nmad(dz)), "median_sep_arcmin": float(np.median(sep)) if sep.size else np.nan}


# --------------------------------------------------------------------------- two runs of rema
def _nmad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def _central(cat):
    ids = np.asarray(cat["ID_CENT"])
    return ids[:, 0] if ids.ndim == 2 else ids


def _unique_rows(keys, rows):
    """Keys that occur once among ``rows``, and their rows."""
    u, first, n = np.unique(keys[rows], return_index=True, return_counts=True)
    return u[n == 1], rows[first[n == 1]]


def match_by_central(a: dict, b: dict, radius_arcmin: float = 1.0, dz_max: float = 0.02):
    """One-to-one matches between two runs of the finder on the same galaxies (e.g. in two
    cosmologies): the same central galaxy (ID_CENT[:, 0]), then the same seed (SEED_ID), then
    the nearest within ``radius_arcmin`` and |dz| < dz_max (1 + z). Returns (ia, ib, how) with
    how = 0, 1, 2 for the three rules."""
    ia, ib, how = [], [], []
    free_a = np.ones(len(a["RA"]), bool)
    free_b = np.ones(len(b["RA"]), bool)
    for rule, key in ((0, _central), (1, lambda c: np.asarray(c["SEED_ID"]) if "SEED_ID" in c else None)):
        ka, kb = key(a), key(b)
        if ka is None or kb is None:
            continue
        ua, ra_ = _unique_rows(ka, np.flatnonzero(free_a & (ka >= 0)))
        ub, rb_ = _unique_rows(kb, np.flatnonzero(free_b & (kb >= 0)))
        _, xa, xb = np.intersect1d(ua, ub, return_indices=True)
        i, j = ra_[xa], rb_[xb]
        ia.append(i), ib.append(j), how.append(np.full(i.size, rule))
        free_a[i], free_b[j] = False, False
    ra, rb = np.flatnonzero(free_a), np.flatnonzero(free_b)
    if ra.size and rb.size:
        i1, i2, _ = match_catalogs(np.asarray(a["RA"])[ra], np.asarray(a["DEC"])[ra],
                                   np.asarray(a["Z_LAMBDA"])[ra], np.asarray(b["RA"])[rb],
                                   np.asarray(b["DEC"])[rb], np.asarray(b["Z_LAMBDA"])[rb],
                                   radius_arcmin, dz_max)
        ia.append(ra[i1]), ib.append(rb[i2]), how.append(np.full(i1.size, 2))
    cat = lambda xs: np.concatenate(xs).astype(int) if xs else np.zeros(0, int)
    return cat(ia), cat(ib), cat(how)


def member_overlap(mem_a: dict, mem_b: dict, mmid_a, mmid_b, col: str = "PMEM") -> np.ndarray:
    """For matched clusters (MEM_MATCH_ID pairs), sum_i min(p_a, p_b) / sum_i max(p_a, p_b) over
    the union of their members (1: the same membership)."""
    key = lambda m, mm: (np.asarray(m["MEM_MATCH_ID"], np.int64), np.asarray(m["ID"], np.int64),
                         np.asarray(m[col], float))
    ma, ia_, pa = key(mem_a, mmid_a)
    mb, ib_, pb = key(mem_b, mmid_b)
    pair_of_b = dict(zip(np.asarray(mmid_b, np.int64), np.asarray(mmid_a, np.int64)))
    sel_a = np.isin(ma, mmid_a)
    sel_b = np.isin(mb, mmid_b)
    # rows keyed by (cluster of a, galaxy)
    ka = list(zip(ma[sel_a], ia_[sel_a]))
    kb = list(zip([pair_of_b[x] for x in mb[sel_b]], ib_[sel_b]))
    da, db = dict(zip(ka, pa[sel_a])), dict(zip(kb, pb[sel_b]))
    num, den = {}, {}
    for k in set(da) | set(db):
        x, y = da.get(k, 0.0), db.get(k, 0.0)
        num[k[0]] = num.get(k[0], 0.0) + min(x, y)
        den[k[0]] = den.get(k[0], 0.0) + max(x, y)
    return np.array([num.get(m, 0.0) / den[m] if den.get(m, 0) > 0 else np.nan for m in np.asarray(mmid_a, np.int64)])


def rerun_summary(a: dict, b: dict, lam_edges, z_edges, lam_min: float = 20.0, mem_a=None,
                  mem_b=None) -> dict:
    """Changes between a reference run ``a`` and a re-run ``b`` of the same region.

    Matched clusters with lambda_a >= lam_min: median and NMAD of d ln(lambda), d ln(R_lambda),
    dz_lambda and dP_CEN[0], the fraction whose centre changed, the member overlap; the clusters
    above lam_min without a match on the other side (lost, gained); and the ratio
    N_b(> lambda) / N_a(> lambda) in bins of z_lambda at each lambda edge.
    """
    ia, ib, how = match_by_central(a, b)
    la, lb = np.asarray(a["LAMBDA"], float), np.asarray(b["LAMBDA"], float)
    big = la[ia] >= lam_min
    out = {"n_a": int(np.sum(la >= lam_min)), "n_b": int(np.sum(lb >= lam_min)),
           "n_matched": int(big.sum()), "match_rule_counts": np.bincount(how[big], minlength=3).tolist()}
    i, j = ia[big], ib[big]
    dl = np.log(lb[j] / la[i])
    out.update(dlnlam_median=float(np.median(dl)) if dl.size else np.nan, dlnlam_nmad=_nmad(dl))
    if "R_LAMBDA" in a and "R_LAMBDA" in b:
        dr = np.log(np.asarray(b["R_LAMBDA"], float)[j] / np.asarray(a["R_LAMBDA"], float)[i])
        out["dlnr_median"] = float(np.median(dr)) if dr.size else np.nan
    dz = np.asarray(b["Z_LAMBDA"], float)[j] - np.asarray(a["Z_LAMBDA"], float)[i]
    out.update(dz_median=float(np.median(dz)) if dz.size else np.nan, dz_nmad=_nmad(dz))
    out["centre_changed"] = float(np.mean(_central(a)[i] != _central(b)[j])) if i.size else np.nan
    if "P_CEN" in a and "P_CEN" in b:
        pa = np.asarray(a["P_CEN"], float)
        pb = np.asarray(b["P_CEN"], float)
        pa, pb = (p[:, 0] if p.ndim == 2 else p for p in (pa, pb))
        out["dpcen_median"] = float(np.median(pb[j] - pa[i])) if i.size else np.nan
    if mem_a is not None and mem_b is not None and i.size:
        ov = member_overlap(mem_a, mem_b, np.asarray(a["MEM_MATCH_ID"])[i], np.asarray(b["MEM_MATCH_ID"])[j])
        out["member_overlap_median"] = float(np.nanmedian(ov))
    matched_a = np.zeros(la.size, bool)
    matched_a[ia] = True
    matched_b = np.zeros(lb.size, bool)
    matched_b[ib] = True
    out["lost"] = int(np.sum((la >= lam_min) & ~matched_a))
    out["gained"] = int(np.sum((lb >= lam_min) & ~matched_b))
    ze = np.asarray(z_edges, float)
    za, zb = np.asarray(a["Z_LAMBDA"], float), np.asarray(b["Z_LAMBDA"], float)
    ratio = np.full((len(lam_edges), ze.size - 1), np.nan)
    for k, lmin in enumerate(lam_edges):
        for m in range(ze.size - 1):
            na = np.sum((la >= lmin) & (za >= ze[m]) & (za < ze[m + 1]))
            nb = np.sum((lb >= lmin) & (zb >= ze[m]) & (zb < ze[m + 1]))
            ratio[k, m] = nb / na if na else np.nan
    out["ncum_ratio"] = ratio.tolist()
    return out
