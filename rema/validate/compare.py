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
