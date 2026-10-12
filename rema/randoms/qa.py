"""Checks of a sample against its randoms: the density ratio data / randoms against any quantity
of the sky (depth, E(B-V), seam distance, |b|, exposure, noise) should be flat.

>>> rng = np.random.default_rng(0)
>>> x = rng.uniform(size=20000)
>>> t = ratio_vs(x[:2000], np.ones(2000), x, np.ones(20000), nbins=4)
>>> bool(np.all(np.abs(t["RATIO"] - 1) < 4 * t["ERR"])), t["NDOF"]
(True, 4)
"""

from __future__ import annotations

import numpy as np


def ratio_vs(values_d, w_d, values_r, w_r, edges=None, nbins: int = 8) -> dict:
    """Normalised density ratio in bins of a quantity: (sum of data weights in the bin / their
    total) / (sum of random weights in the bin / their total), with the Poisson error of the data
    and of the randoms' effective number, and the chi^2 of the ratios against 1.

    ``edges``: bin edges (default: ``nbins`` quantiles of the random values)."""
    vd, vr = np.asarray(values_d, np.float64), np.asarray(values_r, np.float64)
    wd, wr = np.asarray(w_d, np.float64), np.asarray(w_r, np.float64)
    okd, okr = np.isfinite(vd), np.isfinite(vr)
    vd, wd, vr, wr = vd[okd], wd[okd], vr[okr], wr[okr]
    if edges is None:
        edges = np.unique(np.quantile(vr, np.linspace(0, 1, nbins + 1)))
    edges = np.asarray(edges, np.float64)
    hd = np.histogram(vd, edges, weights=wd)[0]
    nd = np.histogram(vd, edges)[0]
    hr = np.histogram(vr, edges, weights=wr)[0]
    hr2 = np.histogram(vr, edges, weights=wr * wr)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (hd / wd.sum()) / (hr / wr.sum())
        neff_r = np.where(hr2 > 0, hr**2 / hr2, 0.0)
        err = ratio * np.sqrt(1.0 / np.maximum(nd, 1) + 1.0 / np.maximum(neff_r, 1))
        good = (nd > 0) & (hr > 0)
        chi2 = float(np.sum(((ratio - 1) / err)[good] ** 2))
    return {"EDGES": edges, "CENTRE": 0.5 * (edges[1:] + edges[:-1]), "N_DATA": nd, "W_RAND": hr,
            "RATIO": ratio, "ERR": err, "CHI2": chi2, "NDOF": int(good.sum())}


def sample_qa(data: dict, rand: dict, quantities: dict, z_bins=None, nbins: int = 8) -> dict:
    """:func:`ratio_vs` of every quantity, for the whole sample and per redshift bin.

    ``quantities``: name -> function(ra, dec) giving the quantity at positions. ``data`` and
    ``rand``: sample columns (RA, DEC, WEIGHT, BIN_Z)."""
    z_bins = sorted(set(np.asarray(data["BIN_Z"]).tolist())) if z_bins is None else z_bins
    out = {}
    for name, fn in quantities.items():
        vd, vr = fn(data["RA"], data["DEC"]), fn(rand["RA"], rand["DEC"])
        out[name] = {"all": ratio_vs(vd, data["WEIGHT"], vr, rand["WEIGHT"], nbins=nbins)}
        edges = out[name]["all"]["EDGES"]
        for k in z_bins:
            md, mr = data["BIN_Z"] == k, rand["BIN_Z"] == k
            out[name][f"z{k}"] = ratio_vs(vd[md], data["WEIGHT"][md], vr[mr], rand["WEIGHT"][mr], edges=edges)
    return out
