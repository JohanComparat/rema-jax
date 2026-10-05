"""Numpy test oracles for the wcen centring of :mod:`rema.core.centering`.

Float64 ports of redMaPPer (erykoff/redmapper; ``file:line`` refer to its source), statement by
statement where possible:

- :func:`find_center`: ``CenteringWcenZred.find_center`` (centering.py:132-358) for one cluster;
- :func:`lnbcglike`: the central term of ``RunLikelihoods._process_cluster``
  (run_likelihoods.py:203-235), rema's LNCGLIKE;
- :func:`w_catalog`: the W column of ``RunPercolation._process_cluster``
  (run_percolation.py:393-407);
- the z -> zred_uncorr mapping of the z_lambda correction: :func:`medz_fit` (``MedZFitter``,
  fitters.py:14-91, as called in zlambdacal.py:338-343), :func:`zlambda_corr_zred_uncorr`
  (``ZlambdaCorrectionPar``, zlambda.py:621-622, 662-663) and their helpers :func:`interpol`
  and :func:`cubic_spline` (utilities.py:557-583, 328-432).

Deliberate deviations (rema's conventions; everything else follows the redMaPPer code):

1. Q_MISS = q0 / (q0 + sum Pcen_unnorm) with q0 = prod_good (1 - Pcen_basic) is returned;
   redMaPPer computes it as a local and leaves ``self.q_miss`` at 0.
2. No centre (no candidate at all, or no Pcen_basic > 0): ngood = 0, Q_MISS = 1 and every slot
   has index -1 and zero probabilities. redMaPPer fills slot 0 with the candidate nearest the
   position (P_CEN = 0) and raises on an empty candidate list.
3. Unused slots keep index -1 and zero probabilities (redMaPPer's initial values).
4. Neighbour arrays are padded: entries with ``valid`` False are ignored. A cluster with
   lambda <= 0 (failed richness) has no candidate.
5. A failed zred (zred_e <= 0 or zred chi^2 < 0) is never a candidate. redMaPPer drops it
   implicitly (negative Gaussian normalisation, then the 1e-10 floor and an infinite
   background).
6. Candidates also need r < ``maxrad`` (rema's scan-mode radius; infinite by default).
7. gz is centred on the cluster redshift: no zlambda_corr, as in redMaPPer's percolation and
   zscan calls (centering.py:175-185 is dead code there).
8. With ``ncand``, at most ``ncand`` candidates keep Pcen_basic > 0: among those with
   phi_cen gz / (sqrt(2 pi) sig) >= 1e-10 (the others cannot reach ucen >= 1e-10), the largest
   pfree phi_cen gz, ties to the lower index. ``ncand=None`` is redMaPPer (no cap).
9. Ties in Pcen_basic go to the lower neighbour index (redMaPPer's reversed quicksort argsort
   has no defined tie order).
10. Quantities redMaPPer computes from its own tables are inputs: the positions as unit vectors
    (pair separations are exact great-circle angles, as from the HTM match), m*, mpc_scale,
    the phi_sat normalisation ``lumnorm`` (rema: Gauss-Legendre integral; redMaPPer: 0.01-mag
    Riemann sum) and ``sigma_g_lookup`` (rema: bilinear ZredBkg; redMaPPer: floor-index lookup
    of a resampled table). No spec-z substitution (centering_use_zspec = False).
11. LNCGLIKE: the central is given (rema flags the seed; redMaPPer's argmin(r) is the seed),
    and a zero or negative product gives -inf / NaN instead of raising. zrmod is z, or
    interpolated in the ``zlambda_corr`` table when one is given (as redMaPPer with a
    zlambdafile); ``cl["z"]`` plays both cluster.redshift and cluster.z_lambda, which are equal in
    the likelihood pass without keepz.
12. W: only valid entries count, and an empty member set gives NaN (redMaPPer resets the
    cluster).
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
from scipy.linalg import solve_banded

SQRT2PI = np.sqrt(2.0 * np.pi)


def gauss_function(x, *p):
    """Gaussian A exp(-(x - mu)^2 / (2 sigma^2)) (utilities.py:69-90)."""
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2. / (2. * sigma ** 2))


def schechter_pdf(x, alpha=-1.0, mstar=0.0):
    """Unnormalised Schechter function in magnitudes (utilities.py:92-111)."""
    return 10. ** (0.4 * (alpha + 1.0) * (mstar - x)) * np.exp(-10. ** (0.4 * (mstar - x)))


def sep_deg(xyz1, xyz2):
    """Great-circle separation [deg] between unit vectors (broadcast over leading axes)."""
    d = np.sqrt(np.sum((np.asarray(xyz1, np.float64) - np.asarray(xyz2, np.float64)) ** 2, axis=-1))
    return np.degrees(2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0)))


def _f8(a):
    return np.asarray(a, np.float64)


def find_center(nb: dict, cl: dict, par: dict, cfg: dict, sigma_g_lookup, *,
                maxrad: float = np.inf, ncand: int | None = None) -> dict:
    """CenteringWcenZred.find_center for one cluster.

    Parameters
    ----------
    nb : neighbour arrays [N]: xyz [N, 3] (unit vectors), refmag, zred, zred_e, zred_chisq, r
        (h^-1 Mpc from the position), p, pmem, pfree, valid.
    cl : cluster scalars: z, Lambda, scaleval, r_lambda, mstar, mpc_scale (h^-1 Mpc per deg),
        alpha, lumnorm (normalisation of phi_sat at m* - 2.5 log10(lval_reference)).
    par : wcen parameters DELTA0, DELTA1, SIGMA_M, LNW_{CEN,SAT,FG}_{MEAN,SIGMA}.
    cfg : pbcg_cut, zred_chisq_max, rsoft, maxlambda, pivot, uselum, maxcen.
    sigma_g_lookup : Sigma_g(zred, refmag) of the zred background [1/(deg^2 mag zred)].

    Returns
    -------
    dict with index [maxcen] (-1 unused), p_cen, q_cen, p_sat, p_fg, p_c [maxcen], q_miss, ngood,
    ncand (candidates before the cap) and ``dbg`` (intermediate arrays over the candidates).
    """
    xyz, refmag, r, p, pmem, pfree = (_f8(nb[k]) for k in ("xyz", "refmag", "r", "p", "pmem", "pfree"))
    valid = np.asarray(nb["valid"], bool)
    redshift, Lambda, scaleval = float(cl["z"]), float(cl["Lambda"]), float(cl["scaleval"])
    r_lambda, mstar, mpc_scale = float(cl["r_lambda"]), float(cl["mstar"]), float(cl["mpc_scale"])
    maxcen = int(cfg["maxcen"])
    out = dict(index=np.full(maxcen, -1, np.int64), p_cen=np.zeros(maxcen), q_cen=np.zeros(maxcen),
               p_sat=np.zeros(maxcen), p_fg=np.zeros(maxcen), p_c=np.zeros(maxcen), q_miss=0.0,
               ngood=0, ncand=0, dbg={})

    # (152-154) views of the neighbour columns; no spec-z substitution.
    z_neighbors, z_neighbors_e, chisq_neighbors = _f8(nb["zred"]), _f8(nb["zred_e"]), _f8(nb["zred_chisq"])

    # Candidate centrals (162-166), with deviations 4-6.
    use, = np.where(valid & (Lambda > 0) & (r < maxrad)
                    & (z_neighbors_e > 0) & (chisq_neighbors >= 0)
                    & (r < r_lambda)
                    & (pfree >= cfg["pbcg_cut"])
                    & (chisq_neighbors < cfg["zred_chisq_max"])
                    & ((pmem > 0.0) | (np.abs(redshift - z_neighbors) < 5.0 * z_neighbors_e)))
    out["ncand"] = int(use.size)
    out["dbg"]["use"] = use
    if use.size == 0:                                                   # deviation 2
        out["q_miss"] = 1.0
        return out

    with np.errstate(all="ignore"):
        # phi_cen filter (169-173)
        mbar = mstar + par["DELTA0"] + par["DELTA1"] * np.log(Lambda / cfg["pivot"])
        phi_cen = gauss_function(refmag[use], 1. / (np.sqrt(2. * np.pi) * par["SIGMA_M"]), mbar,
                                 par["SIGMA_M"])
        # zred filter, zlambda_corr is None (182-185)
        gz = gauss_function(z_neighbors[use], 1. / (np.sqrt(2. * np.pi) * z_neighbors_e[use]),
                            redshift, z_neighbors_e[use])

        # w filter (191-230)
        u, = np.where(valid & (p > 0.0))
        maxrad_deg = 1.1 * r_lambda / mpc_scale
        # esutil HTM match of the members (u) against the candidates (use), every pair within
        # maxrad_deg: i2 indexes u, i1 indexes use.
        dist_all = sep_deg(xyz[u][:, None, :], xyz[use][None, :, :])
        i2, i1 = np.nonzero(dist_all <= maxrad_deg)
        dist = dist_all[i2, i1]
        subdifferent, = np.where(~(use[i1] == u[i2]))
        i1 = i1[subdifferent]
        i2 = i2[subdifferent]
        pdis = dist[subdifferent] * mpc_scale
        pdis = np.sqrt(pdis**2. + cfg["rsoft"]**2.)
        lum = 10.**((mstar - refmag) / (2.5))
        w = np.zeros(use.size) + 1e-3
        for i in range(use.size):
            subgal, = np.where(i1 == i)
            if subgal.size > 0:
                inside, = np.where(pdis[subgal] < r_lambda)
                if inside.size > 0:
                    indices = u[i2[subgal[inside]]]
                    if cfg["uselum"]:
                        w[i] = np.log(np.sum(p[indices] * lum[indices] / pdis[subgal[inside]])
                                      / ((1. / r_lambda) * np.sum(p[indices] * lum[indices])))
                    else:
                        w[i] = np.log(np.sum(p[indices] / pdis[subgal[inside]])
                                      / ((1. / r_lambda) * np.sum(p[indices])))

        # central term (232-245)
        sigscale = np.sqrt((np.clip(Lambda, None, cfg["maxlambda"]) / scaleval) / cfg["pivot"])
        sig = par["LNW_CEN_SIGMA"] / sigscale
        fw = gauss_function(np.log(w), 1. / (np.sqrt(2. * np.pi) * sig), par["LNW_CEN_MEAN"], sig)
        ucen = phi_cen * gz * fw
        ucen_raw = ucen.copy()
        lo, = np.where(ucen < 1e-10)
        ucen[lo] = 0.0

        # satellite term (248-260); cluster._calc_luminosity(maxmag) = schechter_pdf / lumnorm
        phi_sat = schechter_pdf(refmag[use], alpha=cl["alpha"], mstar=mstar) / cl["lumnorm"]
        satsig = par["LNW_SAT_SIGMA"] / sigscale
        fsat = gauss_function(np.log(w), 1. / (np.sqrt(2. * np.pi) * satsig), par["LNW_SAT_MEAN"],
                              satsig)
        usat = phi_sat * gz * fsat
        usat_raw = usat.copy()
        lo, = np.where(usat < 1e-10)
        usat[lo] = 0.0

        # foreground/background term (262-275); calc_zred_bkg_density (cluster.py:367-390)
        fgsig = par["LNW_FG_SIGMA"] / sigscale
        ffg = gauss_function(np.log(w), 1. / (np.sqrt(2. * np.pi) * fgsig), par["LNW_FG_MEAN"], fgsig)
        rtest = np.zeros(use.size) + 0.1
        zred_bkg_density = (2. * np.pi * rtest
                            * (_f8(sigma_g_lookup(z_neighbors[use], refmag[use])) / mpc_scale**2.))
        bcounts = ffg * (zred_bkg_density / (2. * np.pi * rtest)) * np.pi * r_lambda**2.

        # Pcen_basic (277-282)
        pcen_raw = pfree[use] * (ucen / (ucen + (Lambda / scaleval - 1.0) * usat + bcounts))
        Pcen_basic = np.clip(pcen_raw, None, 0.99999)
        bad, = np.where(~np.isfinite(Pcen_basic))
        Pcen_basic[bad] = 0.0

        # candidate cap (deviation 8)
        s = phi_cen * gz
        fwmax = 1. / (np.sqrt(2. * np.pi) * sig)
        pre = np.isfinite(s) & (s * fwmax >= 1e-10)
        score = np.where(pre, pfree[use] * s, -1.0)
        capsel = np.ones(use.size, bool)
        if ncand is not None:
            order = np.lexsort((np.arange(use.size), -score))
            capsel = np.zeros(use.size, bool)
            capsel[order[:ncand]] = True
            capsel &= pre
            Pcen_basic[~capsel] = 0.0

    out["dbg"].update(w=w, s=s, fwmax=fwmax, ucen_raw=ucen_raw, usat_raw=usat_raw, bcounts=bcounts,
                      pcen_raw=pcen_raw, Pcen_basic=Pcen_basic.copy(), pdis=pdis, score=score,
                      pre=pre, capsel=capsel)

    okay, = np.where(Pcen_basic > 0.0)
    if okay.size == 0:                                                  # deviation 2
        out["q_miss"] = 1.0
        return out

    # renormalisation (299-334)
    with np.errstate(all="ignore"):
        Pcen_unnorm = np.zeros(use.size)
        ok, = np.where(Pcen_basic > 0)
        st = np.lexsort((ok, -Pcen_basic[ok]))                          # deviation 9
        if st.size < maxcen:
            good = ok[st]
        else:
            good = ok[st[0: maxcen]]
        ngood = good.size
        for i in range(ngood):
            Pcen0 = Pcen_basic[good[i]]
            Pcen_basic[good[i]] = 0.0
            Pcen_unnorm[good[i]] = Pcen0 * np.prod(1.0 - Pcen_basic[good])
            Pcen_basic[good[i]] = Pcen0
        Qmiss = np.prod(1.0 - Pcen_basic[good])
        KQ = 1. / (Qmiss + np.sum(Pcen_unnorm))
        KP = 1. / np.sum(Pcen_unnorm)
        Pcen = KP * Pcen_unnorm
        Qcen = KQ * Pcen_unnorm

        # satellite and foreground split (336-346)
        Pfg_basic = bcounts[good] / ((Lambda - 1.0) * usat[good] + bcounts[good])
        inf, = np.where(~np.isfinite(Pfg_basic))
        Pfg_basic[inf] = 0.0
        Pfg = (1.0 - Pcen[good]) * Pfg_basic
        Psat_basic = (Lambda - 1.0) * usat[good] / ((Lambda - 1.0) * usat[good] + bcounts[good])
        inf, = np.where(~np.isfinite(Psat_basic))
        Psat_basic[inf] = 0.0
        Psat = (1.0 - Pcen[good]) * Psat_basic

    # outputs (348-356)
    n = good.size
    out["index"][0:n] = use[good]
    out["p_cen"][0:n] = Pcen[good]
    out["q_cen"][0:n] = Qcen[good]
    out["p_fg"][0:n] = Pfg
    out["p_sat"][0:n] = Psat
    out["p_c"][0:n] = Pcen_basic[good]
    out["ngood"] = int(ngood)
    out["q_miss"] = float(KQ * Qmiss)                                   # deviation 1
    return out


def lnbcglike(nb: dict, cl: dict, par: dict, cfg: dict, icen: int, zlambda_corr=None) -> float:
    """Central likelihood of the likelihood pass (run_likelihoods.py:203-235) for one cluster.

    ``nb``, ``cl``, ``par`` and ``cfg`` as :func:`find_center` (``nb`` needs refmag, zred,
    zred_e, r, pmem, valid; ``cl`` z, Lambda, scaleval, r_lambda, mstar); ``icen`` is the index
    of the central (redMaPPer: argmin of r, the seed). ``zlambda_corr``: the (z, zred_uncorr)
    arrays of a ZlambdaCorrectionPar (:func:`zlambda_corr_zred_uncorr`), or None (no
    zlambdafile). NaN for lambda <= 0.
    """
    refmag, zred, zred_e, r, pmem = (_f8(nb[k]) for k in ("refmag", "zred", "zred_e", "r", "pmem"))
    valid = np.asarray(nb["valid"], bool)
    Lambda, mstar = float(cl["Lambda"]), float(cl["mstar"])
    if not Lambda > 0:
        return float("nan")
    with np.errstate(all="ignore"):
        mbar = (mstar + par["DELTA0"] + par["DELTA1"] * np.log(Lambda / cfg["pivot"]))
        phi_cen = ((1. / (np.sqrt(2. * np.pi) * par["SIGMA_M"]))
                   * np.exp(-0.5 * (refmag[icen] - mbar)**2. / par["SIGMA_M"]**2.))
        # (213-216)
        if zlambda_corr is not None:
            zrmod = float(interpol(_f8(zlambda_corr[1]), _f8(zlambda_corr[0]), float(cl["z"])))
        else:
            zrmod = float(cl["z"])
        g = ((1. / (np.sqrt(2. * np.pi) * zred_e[icen]))
             * np.exp(-0.5 * (zred[icen] - zrmod)**2. / zred_e[icen]**2.))
        lum = 10.**((mstar - refmag) / 2.5)
        u, = np.where(valid & (r > 1e-5) & (pmem > 0.0))
        w = np.log(np.sum(pmem[u] * lum[u] / np.sqrt(r[u]**2. + cfg["rsoft"]**2.))
                   / ((1. / cl["r_lambda"]) * np.sum(pmem[u] * lum[u])))
        sig = par["LNW_CEN_SIGMA"] / np.sqrt(((np.clip(Lambda, None, cfg["maxlambda"]))
                                              / cl["scaleval"]) / cfg["pivot"])
        fw = (1. / (np.sqrt(2. * np.pi) * sig)) * np.exp(-0.5 * (np.log(w) - par["LNW_CEN_MEAN"])**2.
                                                       / (sig**2.))
        return float(np.log(phi_cen * np.clip(g, 1e-10, None) * fw))


def w_catalog(r, p, refmag, mstar, r_lambda, rsoft=0.05, uselum=True, valid=None) -> float:
    """Connectivity W at the final centre (run_percolation.py:393-407); NaN without members."""
    r, p, refmag = _f8(r), _f8(p), _f8(refmag)
    if valid is not None:
        keep = np.asarray(valid, bool)
        r, p, refmag = r[keep], p[keep], refmag[keep]
    if r.size == 0:
        return float("nan")
    minind = np.argmin(r)
    u, = np.where((r > r[minind]) & (r < r_lambda) & (p > 0.0))
    if u.size == 0:
        return float("nan")
    lum = 10.**((mstar - refmag[u]) / 2.5)
    if uselum:
        return float(np.log(np.sum(p[u] * lum / np.sqrt(r[u]**2. + rsoft**2.))
                            / ((1. / r_lambda) * np.sum(p[u] * lum))))
    return float(np.log(np.sum(p[u] / np.sqrt(r[u]**2. + rsoft**2.)) / ((1. / r_lambda) * np.sum(p[u]))))


# --------------------------------------------------------------------------- z -> zred_uncorr
def interpol(v, x, xout):
    """Linear interpolation, extrapolating the end segments (utilities.py:557-583)."""
    v, x = _f8(v), _f8(x)
    m = v.size
    s = np.clip(np.searchsorted(x, xout) - 1, 0, m - 2)
    diff = v[s + 1] - v[s]
    return (xout - x[s]) * diff / (x[s + 1] - x[s]) + v[s]


def cubic_spline(x, y, xout):
    """Natural cubic spline through (x, y) at ``xout``, extrapolating the end cubics
    (``CubicSpline(x, y)`` with yp = None and fixextrap = False, utilities.py:328-432)."""
    x, y = _f8(x), _f8(y)
    npts = len(x)
    mat = np.zeros((3, npts))
    mat[1, 1:-1] = (x[2:] - x[0:-2]) / 3.
    mat[2, 0:-2] = (x[1:-1] - x[0:-2]) / 6.
    mat[0, 2:] = (x[2:] - x[1:-1]) / 6.
    bb = np.zeros(npts)
    bb[1:-1] = ((y[2:] - y[1:-1]) / (x[2:] - x[1:-1]) - (y[1:-1] - y[0:-2]) / (x[1:-1] - x[0:-2]))
    mat[1, 0] = 1.
    mat[1, -1] = 1.
    bb[0] = 0.
    bb[-1] = 0.
    y2 = solve_banded((1, 1), mat, bb)
    lo = np.clip(np.searchsorted(x, xout) - 1, 0, npts - 2)
    hi = lo + 1
    dx = x[hi] - x[lo]
    a = (x[hi] - xout) / dx
    b = (xout - x[lo]) / dx
    return a * y[lo] + b * y[hi] + ((a**3 - a) * y2[lo] + (b**3 - b) * y2[hi]) * dx**2. / 6.


def zlambda_corr_zred_uncorr(offset_z, zred_uncorr, zrange, zbinsize=0.002):
    """(z, zred_uncorr) arrays of a ZlambdaCorrectionPar (zlambda.py:621-622, 662-663); ``zrange``
    is the file header's [ZRANGE0, ZRANGE1] = [zrange0 - 0.02, zrange1 + 0.07]
    (zlambdacal.py:347-350)."""
    nbins = np.round((zrange[1] - zrange[0]) / zbinsize).astype(np.int32)
    z = zbinsize * np.arange(nbins) + zrange[0]
    return z, cubic_spline(offset_z, zred_uncorr, z)


class MedZFitter:
    """Spline fit to the median value as a function of redshift (fitters.py:14-91)."""

    def __init__(self, z_nodes, redshifts, values):
        self._z_nodes = _f8(z_nodes)
        self._redshifts = _f8(redshifts)
        self._values = _f8(values)

    def fit(self, p0, min_val=-np.inf, max_val=np.inf):
        bounds = [[min_val, max_val] for _ in range(len(p0))]
        res = scipy.optimize.minimize(self, p0, method='L-BFGS-B', bounds=bounds, jac=False,
                                      options={'maxfun': 2000, 'maxiter': 2000, 'maxcor': 20,
                                               'eps': 1e-5, 'gtol': 1e-8}, callback=None)
        return res.x

    def __call__(self, pars):
        m = cubic_spline(self._z_nodes, pars, self._redshifts)
        absdev = np.abs(self._values - m)
        return np.sum(absdev.astype(np.float64))


def medz_fit(nodes, z_lambda, zred) -> np.ndarray:
    """zred_uncorr node values: ``MedZFitter(nodes, cat.z_lambda, cat.zred).fit(nodes)``
    (zlambdacal.py:338-343)."""
    return MedZFitter(nodes, z_lambda, zred).fit(_f8(nodes))
