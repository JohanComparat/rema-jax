"""Weights of the cluster copies (redMaPPer's RandomWeigher, with an external selection).

Copy j of cluster i, at position x_j, would be in the sample with probability

    q_ij = P_ext(obs_i, z_i, x_j) * 1[MASKFRAC_j < max_maskfrac] * 1[LAMBDA_i / SCALEVAL_j >= lambda_det],

(P_ext = 1 for a sample selected in richness alone). The copies of each cluster share its
weight:

    w_ij = n_samp,i q_ij / sum_k q_ik,

so the weights of a cluster's copies add up to n_samp,i: the weighted randoms have exactly the
redshift, richness and observable distribution of the sample, and the angular distribution of
its selection, p(x | obs, z, selected) proportional to P_ext q over D(z). With q in {0, 1} this
is redMaPPer's n_samp / n_keep. A cluster none of whose copies passes (sum q = 0) gets no
randoms (flagged); this happens only when n_samp is too small for its tiny selected area.

>>> src = np.array([0, 0, 0, 0, 1, 1])
>>> q = np.array([1.0, 0.0, 1.0, 1.0, 0.5, 0.25])
>>> w, d = copy_weights(src, q, np.array([4, 2]))
>>> w.round(3).tolist(), d["SUMQ"].tolist()
([1.333, 0.0, 1.333, 1.333, 1.333, 0.667], [3.0, 0.75])
"""

from __future__ import annotations

import numpy as np


def copy_weights(src, q, n_samp) -> tuple[np.ndarray, dict]:
    """Weights of copies with source index ``src`` and pass probability ``q``; ``n_samp``
    [n_sources] the copies made of each source. Diagnostics per source: SUMQ (sum of q),
    NEFF ((sum q)^2 / sum q^2, the effective number of copies) and NPASS (copies with q > 0)."""
    src = np.asarray(src, np.int64)
    q = np.asarray(q, np.float64)
    n_samp = np.asarray(n_samp, np.float64)
    ns = n_samp.size
    sumq = np.bincount(src, weights=q, minlength=ns)
    sumq2 = np.bincount(src, weights=q * q, minlength=ns)
    npass = np.bincount(src, weights=(q > 0).astype(float), minlength=ns)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(sumq[src] > 0, n_samp[src] * q / sumq[src], 0.0)
        neff = np.where(sumq2 > 0, sumq**2 / sumq2, 0.0)
    return w, {"SUMQ": sumq, "NEFF": neff, "NPASS": npass.astype(np.int64)}


def detection_q(lam, scaleval, maskfrac, lambda_det, max_maskfrac: float) -> np.ndarray:
    """1 where a copy passes the catalogue cuts at its position (``lambda_det`` a number or one
    threshold per row), else 0."""
    lam, scaleval, maskfrac = (np.asarray(a, np.float64) for a in (lam, scaleval, maskfrac))
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (maskfrac < max_maskfrac) & np.isfinite(scaleval) & (lam / scaleval >= lambda_det)
    return ok.astype(np.float64)


def effective_area(z_edges, rand_z, q, area_above) -> np.ndarray:
    """Area of each redshift bin over which the sample is complete (redMaPPer's area(z)):
    A(z_vlim > z_hi) times the fraction of the copies in the bin that pass (q averaged per
    copy), ``area_above`` the function z -> area with z_vlim above z (e.g.
    :meth:`rema.abundance.area.ZvlimMap.area_above`)."""
    z_edges = np.asarray(z_edges, np.float64)
    zb = np.digitize(np.asarray(rand_z, np.float64), z_edges) - 1
    nb = z_edges.size - 1
    ok = (zb >= 0) & (zb < nb)
    num = np.bincount(zb[ok], weights=np.asarray(q, np.float64)[ok], minlength=nb)
    den = np.bincount(zb[ok], minlength=nb).astype(np.float64)
    frac = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return frac * np.asarray(area_above(z_edges[1:]), np.float64)
