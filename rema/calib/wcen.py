"""Calibration of the wcen centring model (redMaPPer ``WcenCalibrator``, calibration/centeringcal.py).

The wcen model (:mod:`rema.core.centering`) scores central candidates with their magnitude, zred
and connectivity W = ln[sum p L / sqrt(r^2 + rsoft^2) / ((1/r_lambda) sum p L)]. W is already a
logarithm, and every ``LNW_*`` parameter models ln W. The parameters are fitted on the
spectroscopically seeded calibration clusters, at z = the seed's ZSPEC:

- :func:`select_training`: wcen_minlambda < lambda/S < wcen_maxlambda, W > 0, z inside
  wcen_cal_zrange and MASKFRAC < max_maskfrac, all strict.
- :func:`fit_lnw` (``WcenFgFitter``): ln W ~ N(mean, sigma / sqrt((lambda/S)/pivot)) of the
  clusters re-centred on random points within r_lambda (LNW_FG_*, :func:`random_offsets`,
  ``CenteringRandom``) and on members drawn with probability pmem (LNW_SAT_*,
  :func:`random_satellite`, ``CenteringRandomSatellite``).
- :func:`phi1_model`: the brightest of lambda - 1 Schechter-distributed satellites, a Gaussian
  whose mean (relative to m*) and width are straight lines in ln(lambda/pivot) (PHI1_*).
- :func:`fit_central_mag` (``WcenCFitter``): DELTA0, DELTA1, SIGMA_M of the central magnitude,
  Gaussian around m* + Delta0 + Delta1 ln((lambda/S)/pivot), in a mixture with the brightest
  satellite (phi1) and the background counts, weighted by P_CEN[0] and P_SAT[0].
- :func:`fit_lnw_cen` (``WcenCwFitter``): LNW_CEN_* from the same mixture in ln W, with the
  satellite and foreground Gaussians fixed.

With BCG centring (redMaPPer's first calibration iteration, the only one in the DR10 run) P_CEN[0] = 1
and P_SAT[0] = 0: the magnitude fit is a Gaussian maximum likelihood in which the 1e-5 floor on
the likelihood acts as a chi^2-dependent outlier clip, and the ln W fit is a single Gaussian.

Formulas, constants, floors and Nelder-Mead settings (xtol = ftol = 1e-5) are redMaPPer's,
including its mixed richness conventions: ln(lambda/pivot) in the least-squares start and in
phi1, ln((lambda/S)/pivot) in the fitted mean magnitude, widths scaled by 1/sqrt((lambda/S)/pivot).
Deviations:

- phi1: magnitudes come from the continuous truncated Schechter function by inverse CDF.
  redMaPPer's ``sample_from_pdf`` quantises the CDF on its 0.002 mag grid, which puts the whole
  bright tail m* - 2.5 to m* - 1.87 (mass 1/2124) exactly on m* - 2.5, where 1-5% of the trials
  then find their brightest galaxy. The histogram fits hardly see that spike: in an emulation
  over 40 seeds the PHI1 values (-0.93, -0.31, 0.39, -0.08 for alpha = -1) move by less than
  their seed-to-seed scatter of 0.006-0.009, while the plain mean of m1 would shift (slope -0.335
  instead of -0.318). Fitted widths are taken in absolute value (the Gaussian is even in sigma).
- ln W fits: sigma > 0 is enforced (cost +inf otherwise); redMaPPer's +1000 penalty does not act
  there, since a negative width makes its cost NaN. ``WcenFgFitter``'s maximum is computed in
  closed form. ``WcenCwFitter``'s mixture is summed in log space with its weights clipped at 0
  (float32 rounding makes 1 - P_CEN - P_SAT about -6e-8), so a component with zero weight drops
  out and needs no parameters.
- Random satellites are drawn exactly in proportion to pmem, not with redMaPPer's CDF quantised
  in steps of 1/N over the N neighbours. Random centres are wrapped into 0 <= RA < 360. The
  random streams differ from redMaPPer's.
- Too few usable rows give NaN parameters and a warning (redMaPPer fails or returns NaN), so the
  centring model counts as uncalibrated.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import scipy.optimize
from scipy import special

log = logging.getLogger(__name__)

PHI1_KEYS = ("PHI1_MMSTAR_M", "PHI1_MMSTAR_SLOPE", "PHI1_MSIG_M", "PHI1_MSIG_SLOPE")
PHI1_LUM_RANGE = (10.0, 0.2)        # L/L* of the phi1 Monte Carlo: m* - 2.5 to m* + 1.7474
RHO_FLOOR = 1e-5                    # WcenCFitter: floor on each cluster's likelihood
SIGMA_PENALTY = 1000.0              # WcenCFitter: added to the cost when sigma_m < 0
FMIN_TOL = 1e-5                     # xtol = ftol of the Nelder-Mead fits
CDF_NODES = 20001                   # nodes of the tabulated Schechter CDF (0.0002 mag)
_LN_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _fmin(cost, p0) -> np.ndarray:
    """``scipy.optimize.fmin`` with redMaPPer's settings; warns when it stops at a limit."""
    p, _, _, _, flag = scipy.optimize.fmin(cost, np.asarray(p0, np.float64), xtol=FMIN_TOL,
                                           ftol=FMIN_TOL, disp=False, full_output=True)
    if flag:
        log.warning("wcen fit: Nelder-Mead stopped without converging (warnflag %d)", flag)
    return p


def _ln_gauss(x, mu, sig):
    return -0.5 * ((x - mu) / sig) ** 2 - np.log(sig) - _LN_SQRT_2PI


def _as_f8(*arrays):
    return np.broadcast_arrays(*(np.asarray(a, np.float64) for a in arrays))


def _drop_bad(ok: np.ndarray, what: str) -> None:
    if not ok.all():
        log.warning("%s: %d of %d rows with non-finite input or lambda/S <= 0 ignored",
                    what, int((~ok).sum()), ok.size)


# --------------------------------------------------------------------------- inputs and sample
def chisq_pdf(chisq, k):
    """chi^2 pdf with ``k`` degrees of freedom (redMaPPer ``chisq_pdf``).

    The weight ``cwt`` of :func:`fit_central_mag`, with ``k`` = the number of colours. A negative
    chi^2 (a failed fit) gives NaN for every k, which that fit's floor excludes.

    >>> float(chisq_pdf(2.0, 2))
    0.18393972058572117
    """
    chisq = np.asarray(chisq, np.float64)
    with np.errstate(invalid="ignore"):
        pdf = chisq ** (k / 2.0 - 1.0) * np.exp(-chisq / 2.0) / (2.0 ** (k / 2.0) * special.gamma(k / 2.0))
    return np.where(chisq >= 0.0, pdf, np.nan)


def background_counts(sigma_g, mpc_per_deg, r_lambda):
    """``bcounts`` of :func:`fit_central_mag`: Sigma_g / mpc_per_deg^2 x pi r_lambda^2.

    Sigma_g [deg^-2 mag^-1 per unit chi^2] is the chi^2 background at the central's (z, chi^2, m),
    ``mpc_per_deg`` the h^-1 Mpc per degree at z (redMaPPer's ``mpc_scale``): background galaxies
    per mag and unit chi^2 within r_lambda. An infinite Sigma_g (outside the table) stays infinite.
    """
    sigma_g, mpc_per_deg, r_lambda = _as_f8(sigma_g, mpc_per_deg, r_lambda)
    return (sigma_g / mpc_per_deg**2) * np.pi * r_lambda**2


def select_training(lam, scaleval, w, z, maskfrac, cfg) -> np.ndarray:
    """Clusters that enter the wcen fits (``WcenCalibrator.run``, centeringcal.py:328-333).

    wcen_minlambda < lambda/S < wcen_maxlambda, W > 0, wcen_cal_zrange[0] < z < wcen_cal_zrange[1]
    and MASKFRAC < max_maskfrac, from ``cfg.centering`` and ``cfg.mask``. Every inequality is
    strict; NaN fails every test. ``z`` is the seed's ZSPEC. The same selection applies to the
    random-centre and random-satellite catalogues, with their own lambda, S, W and MASKFRAC.
    """
    c = cfg.centering
    lam, scaleval, w, z, maskfrac = _as_f8(lam, scaleval, w, z, maskfrac)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_s = lam / scaleval
    zlo, zhi = c.wcen_cal_zrange
    return ((lam_s > c.wcen_minlambda) & (lam_s < c.wcen_maxlambda) & (w > 0.0)
            & (z > zlo) & (z < zhi) & (maskfrac < cfg.mask.max_maskfrac))


def random_offsets(rng, r_lambda, mpc_per_deg, ra, dec) -> tuple[np.ndarray, np.ndarray]:
    """Random centres uniform in area within r_lambda (``CenteringRandom``, centering.py:385-392).

    r = r_lambda sqrt(U1), phi = 2 pi U2, and the flat-sky offset x = r cos(phi), y = r sin(phi)
    [h^-1 Mpc] becomes RA + x / (mpc_per_deg cos(Dec)), Dec + y / mpc_per_deg [deg]. Inputs
    broadcast (repeat them for several centres per cluster); RA is wrapped into [0, 360).
    ``rng`` is a ``numpy.random.Generator`` or a seed. Returns (ra, dec).
    """
    rng = np.random.default_rng(rng)
    r_lambda, mpc_per_deg, ra, dec = _as_f8(r_lambda, mpc_per_deg, ra, dec)
    r = r_lambda * np.sqrt(rng.random(r_lambda.shape))
    phi = 2.0 * np.pi * rng.random(r_lambda.shape)
    x = r * np.cos(phi) / mpc_per_deg
    y = r * np.sin(phi) / mpc_per_deg
    ra_c = np.mod(ra + x / np.cos(np.radians(dec)), 360.0)
    return np.where(ra_c >= 360.0, 0.0, ra_c), dec + y      # mod of -1e-17 rounds to 360


def random_satellite(rng, pmem_rows) -> np.ndarray:
    """One neighbour per row, drawn with probability pmem / sum(pmem) (``CenteringRandomSatellite``).

    ``pmem_rows``: a [N, K] array (numpy or JAX, padded with zeros) or a sequence of N 1-D arrays
    (any iterable, ragged rows allowed), the memberships of each cluster's neighbours at its seed
    position. A 1-D array, or a flat sequence of numbers, is one row; an empty sequence has no
    rows. The draw is exact: redMaPPer quantises the CDF in steps of 1/K (centering.py:434-443),
    so that galaxies with pmem below sum(pmem)/K are picked with probability 0 or 1/K. The chosen
    galaxy can be the central itself. Non-finite or negative pmem count as 0; a row without
    positive pmem gives -1. Returns int64 indices into each row, one per row.
    """
    rng = np.random.default_rng(rng)
    if hasattr(pmem_rows, "ndim"):                           # an array: [N, K], or 1-D = one row
        pmem_rows = np.asarray(pmem_rows)
    else:                                                    # a sequence of rows
        pmem_rows = list(pmem_rows)
        if not pmem_rows:                                    # no rows, not one row without members
            return np.zeros(0, np.int64)
        try:
            pmem_rows = np.asarray(pmem_rows, np.float64)
        except ValueError:                                   # ragged rows
            pmem_rows = [np.asarray(r, np.float64) for r in pmem_rows]
    if isinstance(pmem_rows, np.ndarray) and pmem_rows.ndim == 1:
        pmem_rows = pmem_rows[None]
    rows = list(pmem_rows)
    u = rng.random(len(rows))
    out = np.full(len(rows), -1, np.int64)
    for i, row in enumerate(rows):
        p = np.asarray(row, np.float64)
        p = np.where(np.isfinite(p) & (p > 0.0), p, 0.0)
        cdf = np.cumsum(p)
        if cdf.size == 0 or not cdf[-1] > 0.0:
            continue
        j = int(np.searchsorted(cdf, u[i] * cdf[-1], side="right"))
        out[i] = min(j, int(np.flatnonzero(p)[-1]))       # u cdf[-1] rounding up to cdf[-1]
    return out


# --------------------------------------------------------------------------- phi1 Monte Carlo
def _schechter(m, alpha, mstar=0.0):
    """redMaPPer's Schechter function in magnitudes, unnormalised (centeringcal.py:486-487)."""
    return 10.0 ** (0.4 * (alpha + 1.0) * (mstar - m)) * np.exp(-(10.0 ** (0.4 * (mstar - m))))


def _schechter_draws(rng, n: int, alpha: float, mrange) -> np.ndarray:
    """n magnitudes (m* = 0) from the Schechter function truncated to ``mrange``, by inverse CDF.

    The CDF is integrated with the trapezoid rule on CDF_NODES nodes (0.0002 mag apart) and
    inverted by linear interpolation, so the draws follow the density averaged over those cells:
    continuous, without quantisation.
    """
    m = np.linspace(mrange[0], mrange[1], CDF_NODES)
    pdf = _schechter(m, alpha)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(m))])
    return np.interp(rng.random(n), cdf / cdf[-1], m)


def _gauss(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def _histo_gauss(x: np.ndarray) -> tuple[float, float, float]:
    """(A, mu, sigma) of a least-squares Gaussian fit to the histogram of x (redMaPPer histoGauss).

    Bin size 2 IQR n^(-1/3); bins start at min(x) and are fitted at their centres (esutil
    ``histogram`` convention). Starts from (n, median, std), which is also the fallback if the
    fit fails or is not finite.
    """
    x = np.asarray(x, np.float64)
    q25, q75 = np.percentile(x, [25, 75])
    binsize = 2.0 * (q75 - q25) * x.size ** (-1.0 / 3.0)
    p0 = (float(x.size), float(np.median(x)), float(np.std(x)))
    if not binsize > 0.0:
        return p0
    nbin = int((x.max() - x.min()) / binsize) + 1
    ibin = np.clip(np.floor((x - x.min()) / binsize).astype(np.int64), 0, nbin - 1)
    hist = np.bincount(ibin, minlength=nbin).astype(np.float64)
    centre = x.min() + (np.arange(nbin) + 0.5) * binsize
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeff, _ = scipy.optimize.curve_fit(_gauss, centre, hist, p0=p0)
    except Exception:                    # as redMaPPer: any failure falls back to the start
        return p0
    if not np.all(np.isfinite(coeff)):
        return p0
    return float(coeff[0]), float(coeff[1]), abs(float(coeff[2]))


def phi1_model(alpha=-1.0, pivot=30.0, rng=None, nmag=100_000, ntrial=5000,
               lambdas=(20, 30, 40, 50, 60, 70, 80, 90, 100)) -> dict:
    """Brightest-satellite magnitude model PHI1_* (``_schechter_montecarlo_calib``, centeringcal.py:448-539).

    A pool of ``nmag`` magnitudes (relative to m*) is drawn from the Schechter function with slope
    ``alpha`` between 10 L* and 0.2 L* (m* - 2.5 to m* + 1.7474), by exact inverse CDF (see the
    module notes for redMaPPer's quantised sampler). Each of ``ntrial`` trials draws max(lambdas)
    - 1 pool members without replacement in random order; for every lambda its first lambda - 1
    give the brightest satellite m1, as redMaPPer's prefixes of one permutation per trial. Per
    lambda, a Gaussian fitted to the histogram of m1 (histoGauss) gives its mean and width, and
    unweighted straight lines in ln(lambda/pivot) give

        PHI1_MMSTAR_M + PHI1_MMSTAR_SLOPE ln(lambda/pivot)   (mean of m1 - m*),
        PHI1_MSIG_M + PHI1_MSIG_SLOPE ln(lambda/pivot)       (its width).

    ``rng``: a ``numpy.random.Generator`` or a seed (None: fresh entropy, not reproducible). The
    defaults are redMaPPer's (its testing mode is nmag=1000, ntrial=100, lambdas=(20, 60, 100))
    and take about 0.1 s.
    """
    rng = np.random.default_rng(rng)
    lambdas = np.asarray(lambdas)
    if (lambdas.ndim != 1 or np.unique(lambdas).size < 2 or np.any(lambdas != np.round(lambdas))
            or lambdas.min() < 2):
        raise ValueError(f"lambdas must hold at least two distinct integers >= 2, got {lambdas}")
    lambdas = lambdas.astype(np.int64)
    ndraw = int(lambdas.max()) - 1
    if ndraw > nmag or ntrial < 1:
        raise ValueError(f"need nmag >= max(lambdas) - 1 = {ndraw} and ntrial >= 1, got {nmag}, {ntrial}")
    pivot = float(pivot)
    pool = _schechter_draws(rng, nmag, float(alpha), -2.5 * np.log10(np.asarray(PHI1_LUM_RANGE)))
    idx = np.empty((ntrial, ndraw), np.int64)
    for i in range(ntrial):
        idx[i] = rng.choice(nmag, size=ndraw, replace=False)
    m1 = np.minimum.accumulate(pool[idx], axis=1)[:, lambdas - 2].T     # [nlambda, ntrial]
    coeff = np.array([_histo_gauss(m) for m in m1])
    lnl = np.log(lambdas / pivot)
    mean_slope, mean_m = np.polyfit(lnl, coeff[:, 1], 1)
    sig_slope, sig_m = np.polyfit(lnl, coeff[:, 2], 1)
    return {"PHI1_MMSTAR_M": float(mean_m), "PHI1_MMSTAR_SLOPE": float(mean_slope),
            "PHI1_MSIG_M": float(sig_m), "PHI1_MSIG_SLOPE": float(sig_slope)}


# --------------------------------------------------------------------------- fits
def fit_lnw(lnw, lam_s, pivot=30.0) -> tuple[float, float]:
    """(mean, sigma) of ln W ~ N(mean, sigma / sqrt(lam_s/pivot)) (``WcenFgFitter``).

    ``lnw`` = ln(W) of the clusters re-centred on random points (LNW_FG_*) or on random
    satellites (LNW_SAT_*), ``lam_s`` their lambda/S. The maximum likelihood is in closed form:
    with v = lam_s/pivot, mean = sum v lnw / sum v and sigma^2 = sum v (lnw - mean)^2 / n.
    redMaPPer reaches the same point by Nelder-Mead. Non-finite rows and lam_s <= 0 are ignored;
    fewer than two rows, or sigma = 0, give (nan, nan).
    """
    x, ls = _as_f8(lnw, lam_s)
    ok = np.isfinite(x) & np.isfinite(ls) & (ls > 0.0)
    _drop_bad(ok, "fit_lnw")
    x, v = x[ok], ls[ok] / float(pivot)
    if x.size < 2:
        log.warning("fit_lnw: %d usable rows, need at least 2", x.size)
        return np.nan, np.nan
    mean = np.sum(v * x) / np.sum(v)
    sigma = np.sqrt(np.mean(v * (x - mean) ** 2))
    if not sigma > 0.0:
        log.warning("fit_lnw: zero width")
        return np.nan, np.nan
    return float(mean), float(sigma)


def fit_lnw_cen(lnw, lam_s, pcen, psat, lnw_sat, lnw_fg, pivot=30.0) -> tuple[float, float]:
    """(LNW_CEN_MEAN, LNW_CEN_SIGMA) from the ln W mixture of the centrals (``WcenCwFitter``).

    Minimises -sum ln(pcen fcen + psat fsat + (1 - pcen - psat) ffg), with every component
    N(lnw; mean, sigma / sqrt(lam_s/pivot)): fcen with the fitted (mean, sigma), fsat and ffg
    with the fixed ``lnw_sat`` and ``lnw_fg`` = (mean, sigma) from :func:`fit_lnw`. ``pcen`` and
    ``psat`` are P_CEN[:, 0] and P_SAT[:, 0] (scalars broadcast). Nelder-Mead (xtol = ftol =
    1e-5) from [mean(lnw), std(lnw)]; sigma <= 0 has infinite cost. The mixture is summed in log
    space with weights clipped at 0; a component with zero weight everywhere needs no
    parameters, so with pcen = 1 (BCG centring) ``lnw_sat`` and ``lnw_fg`` may be None or NaN.
    Non-finite rows and lam_s <= 0 are ignored; fewer than two rows give (nan, nan).
    """
    lnw_sat = (np.nan, np.nan) if lnw_sat is None else lnw_sat
    lnw_fg = (np.nan, np.nan) if lnw_fg is None else lnw_fg
    x, ls, pc, ps = _as_f8(lnw, lam_s, pcen, psat)
    ok = np.isfinite(x) & np.isfinite(ls) & (ls > 0.0) & np.isfinite(pc) & np.isfinite(ps)
    _drop_bad(ok, "fit_lnw_cen")
    x, pc, ps = x[ok], pc[ok], ps[ok]
    scale = 1.0 / np.sqrt(ls[ok] / float(pivot))
    if x.size < 2 or not np.std(x) > 0.0:
        log.warning("fit_lnw_cen: %d usable rows, need at least 2 distinct values", x.size)
        return np.nan, np.nan
    # The fixed satellite and foreground components, ln(psat fsat + pfg ffg).
    ln_rest = np.full(x.size, -np.inf)
    for name, weight, (mu, sig) in (("lnw_sat", ps, lnw_sat), ("lnw_fg", 1.0 - pc - ps, lnw_fg)):
        use = weight > 0.0
        if not use.any():
            continue
        mu, sig = float(mu), float(sig)
        if not (np.isfinite(mu) and sig > 0.0):
            raise ValueError(f"fit_lnw_cen: {name} = ({mu}, {sig}) is needed where its weight is > 0")
        ln_rest[use] = np.logaddexp(ln_rest[use],
                                    np.log(weight[use]) + _ln_gauss(x[use], mu, sig * scale[use]))
    with np.errstate(divide="ignore"):
        ln_pc = np.log(np.maximum(pc, 0.0))

    def cost(p):
        if not p[1] > 0.0:
            return np.inf
        return -np.sum(np.logaddexp(ln_pc + _ln_gauss(x, p[0], p[1] * scale), ln_rest))

    p = _fmin(cost, [np.mean(x), np.std(x)])
    return float(p[0]), float(p[1])


def fit_central_mag(refmag, mstar, lam, lam_s, pcen, psat, cwt, bcounts, phi1: dict | None,
                    pivot=30.0) -> tuple[float, float, float]:
    """(DELTA0, DELTA1, SIGMA_M) of the central magnitude (``WcenCFitter``, centeringcal.py:367-404).

    Per cluster: the chosen central's ``refmag``, ``mstar`` = m*(z), ``lam`` = LAMBDA, ``lam_s`` =
    LAMBDA/SCALEVAL, ``pcen``/``psat`` = P_CEN[:, 0]/P_SAT[:, 0], ``cwt`` = :func:`chisq_pdf` of the
    central's chi^2 at z, ``bcounts`` = :func:`background_counts` of its chi^2 background Sigma_g
    (inf outside the table), ``phi1`` = :func:`phi1_model`'s dict (None if psat = 0 everywhere).

    1. Start: a soft_l1 least-squares line of refmag - m* against ln(lam/pivot) gives Delta0 and
       Delta1, and sigma_m = std of its residuals.
    2. Nelder-Mead (xtol = ftol = 1e-5) on -sum ln(rho), with
       rho = pcen phicen cwt + psat phi1 cwt + (1 - pcen - psat) bcounts,
       phicen = N(refmag; m* + Delta0 + Delta1 ln(lam_s/pivot), sigma_m), phi1 = N(refmag;
       m* + PHI1_MMSTAR_M + PHI1_MMSTAR_SLOPE ln(lam/pivot), PHI1_MSIG_M + PHI1_MSIG_SLOPE
       ln(lam/pivot)). rho < 1e-5 or non-finite (as 0 x inf) is set to 1e-5, so those clusters
       drop out; +1000 when sigma_m < 0.

    bcounts is an unnormalised count density mixed with normalised pdfs, as in redMaPPer. Rows
    with non-finite refmag - m* or ln(lam/pivot) are left out of the start only; fewer than three
    such rows give (nan, nan, nan).
    """
    refmag, mstar, lam, lam_s, pcen, psat, cwt, bcounts = _as_f8(
        refmag, mstar, lam, lam_s, pcen, psat, cwt, bcounts)
    pivot = float(pivot)
    with np.errstate(divide="ignore", invalid="ignore"):
        lnlam = np.log(lam / pivot)          # LAMBDA: start and phi1
        lscale = np.log(lam_s / pivot)       # LAMBDA/SCALEVAL: the fitted mean magnitude
    dm = refmag - mstar
    ok = np.isfinite(dm) & np.isfinite(lnlam)
    if ok.sum() < 3:
        log.warning("fit_central_mag: %d usable rows, need at least 3", int(ok.sum()))
        return np.nan, np.nan, np.nan
    fit = scipy.optimize.least_squares(lambda p, x, y: (p[1] + p[0] * x) - y, [0.0, 0.0],
                                       loss="soft_l1", args=(lnlam[ok], dm[ok]))
    delta0, delta1 = fit.x[1], fit.x[0]
    sigma_m = np.std(dm[ok] - (delta0 + delta1 * lnlam[ok]))

    if phi1 is None:
        if np.any(psat > 0.0):
            raise ValueError("fit_central_mag: phi1 is needed when psat > 0")
        phi1v = np.zeros_like(refmag)
    else:
        p1 = {k: float(phi1[k]) for k in PHI1_KEYS}
        mmstar1 = mstar + p1["PHI1_MMSTAR_M"] + p1["PHI1_MMSTAR_SLOPE"] * lnlam
        phisig1 = p1["PHI1_MSIG_M"] + p1["PHI1_MSIG_SLOPE"] * lnlam
        with np.errstate(all="ignore"):
            phi1v = (1.0 / (np.sqrt(2.0 * np.pi) * phisig1)) * np.exp(
                -0.5 * (refmag - mmstar1) ** 2 / phisig1**2)
    with np.errstate(all="ignore"):
        sat = psat * phi1v * cwt
        fg = (1.0 - pcen - psat) * bcounts

    def cost(p):
        mbar = mstar + p[0] + p[1] * lscale
        with np.errstate(all="ignore"):
            phicen = (1.0 / (np.sqrt(2.0 * np.pi) * p[2])) * np.exp(-0.5 * (refmag - mbar) ** 2 / p[2] ** 2)
            rho = pcen * phicen * cwt + sat + fg
        rho = np.where((rho < RHO_FLOOR) | ~np.isfinite(rho), RHO_FLOOR, rho)
        t = -np.sum(np.log(rho))
        return t + SIGMA_PENALTY if p[2] < 0.0 else t

    p = _fmin(cost, [delta0, delta1, sigma_m])
    return float(p[0]), float(p[1]), float(p[2])
