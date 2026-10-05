"""wcen calibration (rema.calib.wcen): phi1 Monte Carlo, ln W and magnitude fits, training
selection, random centres and satellites."""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize
import scipy.special
import scipy.stats

from rema.calib import wcen
from rema.config import RemaConfig

PIVOT = 30.0
D0, D1, SIGMA_M = -1.5, -0.4, 0.3
PHI1 = {"PHI1_MMSTAR_M": -0.94, "PHI1_MMSTAR_SLOPE": -0.31, "PHI1_MSIG_M": 0.385,
        "PHI1_MSIG_SLOPE": -0.077}


def _lam_s(rng, n):
    """lambda/S log-uniform over the training range 10-100."""
    return np.exp(rng.uniform(np.log(10.0), np.log(100.0), n))


def _mixture_labels(rng, pcen, psat):
    """0 = central, 1 = satellite, 2 = foreground, drawn with probabilities (pcen, psat, rest)."""
    u = rng.random(pcen.size)
    return np.where(u < pcen, 0, np.where(u < pcen + psat, 1, 2))


# redMaPPer's procedures, transcribed from calibration/centeringcal.py (WcenCalibrator.run with
# WcenCFitter and WcenCwFitter) as references for the port.
def _redmapper_central_mag(refmag, mstar, lam, lam_s, pcen, psat, cwt, bcounts, phi1, pivot):
    fit = scipy.optimize.least_squares(lambda p, x, y: (p[1] + p[0] * x) - y, [0.0, 0.0],
                                       loss="soft_l1", args=(np.log(lam / pivot), refmag - mstar))
    delta0, delta1 = fit.x[1], fit.x[0]
    sigma_m = np.std((refmag - mstar) - (delta0 + delta1 * np.log(lam / pivot)))
    mmstar1 = mstar + phi1["PHI1_MMSTAR_M"] + phi1["PHI1_MMSTAR_SLOPE"] * np.log(lam / pivot)
    phisig1 = phi1["PHI1_MSIG_M"] + phi1["PHI1_MSIG_SLOPE"] * np.log(lam / pivot)
    phi1v = ((1.0 / (np.sqrt(2.0 * np.pi) * phisig1))
             * np.exp(-0.5 * (refmag - mmstar1) ** 2.0 / (phisig1**2.0)))
    lscale = np.log(lam_s / pivot)

    def cost(pars):
        mbar = mstar + pars[0] + pars[1] * lscale
        with np.errstate(all="ignore"):
            phicen = ((1.0 / (np.sqrt(2.0 * np.pi) * pars[2]))
                      * np.exp(-0.5 * (refmag - mbar) ** 2.0 / (pars[2] ** 2.0)))
            rho = pcen * phicen * cwt + psat * phi1v * cwt + (1.0 - pcen - psat) * bcounts
            (bad,) = np.where((rho < 1e-5) | (~np.isfinite(rho)))
            rho[bad] = 1e-5
        t = -np.sum(np.log(rho))
        return t + 1000 if pars[2] < 0.0 else t

    return scipy.optimize.fmin(cost, np.array([delta0, delta1, sigma_m]), disp=False, xtol=1e-5, ftol=1e-5)


def _redmapper_lnw_cen(w, lam_s, pcen, psat, lnw_sat, lnw_fg, pivot):
    fgsig = lnw_fg[1] / np.sqrt(lam_s / pivot)
    ffg = ((1.0 / (np.sqrt(2.0 * np.pi) * fgsig))
           * np.exp(-0.5 * (np.log(w) - lnw_fg[0]) ** 2.0 / (fgsig**2.0)))
    satsig = lnw_sat[1] / np.sqrt(lam_s / pivot)
    fsat = ((1.0 / (np.sqrt(2.0 * np.pi) * satsig))
            * np.exp(-0.5 * (np.log(w) - lnw_sat[0]) ** 2.0 / (satsig**2.0)))
    lscale = 1.0 / np.sqrt(lam_s / pivot)

    def cost(pars):
        sig = pars[1] * lscale
        fcen = (1.0 / (np.sqrt(2.0 * np.pi) * sig)) * np.exp(-0.5 * (np.log(w) - pars[0]) ** 2.0 / (sig**2.0))
        t = -np.sum(np.log(pcen * fcen + psat * fsat + (1.0 - pcen - psat) * ffg))
        return t + 1000.0 if pars[1] < 0.0 else t

    p0 = np.array([np.mean(np.log(w)), np.std(np.log(w))])
    return scipy.optimize.fmin(cost, p0, disp=False, xtol=1e-5, ftol=1e-5)


# --------------------------------------------------------------------------- phi1
@pytest.mark.parametrize("nmag, ntrial, tol", [(20_000, 1000, 0.08), (100_000, 5000, 0.035)])
def test_phi1_model_reference_values(nmag, ntrial, tol):
    # Reference values of the Monte Carlo with an exact sampler: -0.94, -0.31, 0.385, -0.077. The
    # seed-to-seed scatter of single runs is 0.016-0.021 at the reduced size, 0.006-0.009 at redMaPPer's.
    p = wcen.phi1_model(rng=1, nmag=nmag, ntrial=ntrial)
    assert tuple(p) == wcen.PHI1_KEYS
    assert p["PHI1_MMSTAR_M"] == pytest.approx(-0.94, abs=tol)
    assert p["PHI1_MMSTAR_SLOPE"] == pytest.approx(-0.31, abs=tol)
    assert p["PHI1_MSIG_M"] == pytest.approx(0.385, abs=0.7 * tol)
    assert p["PHI1_MSIG_SLOPE"] == pytest.approx(-0.077, abs=tol)
    assert wcen.phi1_model(rng=1, nmag=nmag, ntrial=ntrial) == p


def test_phi1_model_testing_sizes_and_checks():
    p = wcen.phi1_model(rng=2, nmag=1000, ntrial=100, lambdas=(20, 60, 100))   # redMaPPer testing mode
    assert np.all(np.isfinite(list(p.values()))) and p["PHI1_MSIG_M"] > 0
    with pytest.raises(ValueError):
        wcen.phi1_model(lambdas=(20,))
    with pytest.raises(ValueError):
        wcen.phi1_model(nmag=50)


def test_schechter_draws_are_continuous():
    rng = np.random.default_rng(3)
    mrange = -2.5 * np.log10(np.asarray(wcen.PHI1_LUM_RANGE))
    m = wcen._schechter_draws(rng, 200_000, -1.0, mrange)
    assert m.min() >= mrange[0] and m.max() <= mrange[1]
    # alpha = -1: dN/dx = exp(-x)/x in x = L/L*, so P(M < m) = (E1(x(m)) - E1(10)) / (E1(0.2) - E1(10)).
    e1 = scipy.special.exp1
    cdf = lambda mm: (e1(10.0 ** (-0.4 * mm)) - e1(10.0)) / (e1(0.2) - e1(10.0))
    assert scipy.stats.kstest(m, cdf).pvalue > 0.01
    # The tail that redMaPPer's quantised sampler collapses onto m* - 2.5 is drawn continuously.
    tail = m < -1.868
    assert tail.mean() == pytest.approx(cdf(-1.868), rel=0.35)
    assert np.unique(m[tail]).size == tail.sum()


# --------------------------------------------------------------------------- ln W fits
def test_fit_lnw_recovers_mean_and_width():
    rng = np.random.default_rng(10)
    lam_s = _lam_s(rng, 5000)
    lnw = 0.12 + 0.28 / np.sqrt(lam_s / PIVOT) * rng.normal(size=lam_s.size)
    mean, sigma = wcen.fit_lnw(lnw, lam_s, PIVOT)
    assert mean == pytest.approx(0.12, abs=0.012)        # statistical errors 0.003
    assert sigma == pytest.approx(0.28, abs=0.012)

    # The same maximum as redMaPPer's WcenFgFitter: Nelder-Mead on its cost.
    lscale = 1.0 / np.sqrt(lam_s / PIVOT)

    def cost(p):
        sig = p[1] * lscale
        f = (1.0 / (np.sqrt(2.0 * np.pi) * sig)) * np.exp(-0.5 * (lnw - p[0]) ** 2 / sig**2)
        return -np.sum(np.log(f)) + (1000.0 if p[1] < 0 else 0.0)

    p = scipy.optimize.fmin(cost, [lnw.mean(), lnw.std()], xtol=1e-5, ftol=1e-5, disp=False)
    np.testing.assert_allclose([mean, sigma], p, atol=2e-4)
    assert cost([mean, sigma]) <= cost(p) + 1e-9

    # Non-finite rows and lambda/S <= 0 are ignored; too few rows give NaN.
    bad = wcen.fit_lnw(np.r_[lnw, np.nan, 0.3], np.r_[lam_s, 30.0, 0.0], PIVOT)
    assert bad == (mean, sigma)
    assert np.isnan(wcen.fit_lnw([0.1], [30.0])).all()
    assert np.isnan(wcen.fit_lnw([0.1, 0.1], [30.0, 50.0])).all()


def test_fit_lnw_cen_recovers_central_component():
    rng = np.random.default_rng(11)
    n = 6000
    lam_s = _lam_s(rng, n)
    pcen = rng.uniform(0.5, 1.0, n)
    psat = (1.0 - pcen) * rng.uniform(0.2, 0.8, n)
    pars = np.array([(0.15, 0.22), (-0.35, 0.40), (-0.9, 0.6)])     # central, satellite, foreground
    comp = _mixture_labels(rng, pcen, psat)
    lnw = pars[comp, 0] + pars[comp, 1] / np.sqrt(lam_s / PIVOT) * rng.normal(size=n)
    mean, sigma = wcen.fit_lnw_cen(lnw, lam_s, pcen, psat, tuple(pars[1]), tuple(pars[2]), PIVOT)
    assert mean == pytest.approx(0.15, abs=0.015)        # statistical errors 0.003
    assert sigma == pytest.approx(0.22, abs=0.015)
    dev = wcen.fit_lnw_cen(lnw, lam_s, pcen, psat, jnp.asarray(pars[1]), jnp.asarray(pars[2]), PIVOT)
    assert dev == (mean, sigma)
    # redMaPPer's WcenCwFitter (linear-space mixture) finds the same point.
    ref = _redmapper_lnw_cen(np.exp(lnw), lam_s, pcen, psat, pars[1], pars[2], PIVOT)
    np.testing.assert_allclose((mean, sigma), ref, atol=1e-4)
    # A single Gaussian through the contaminated sample is far off.
    mean0, sigma0 = wcen.fit_lnw(lnw, lam_s, PIVOT)
    assert mean0 < 0.05 and sigma0 > 0.4


def test_fit_lnw_cen_bcg_case_is_one_gaussian():
    rng = np.random.default_rng(12)
    lam_s = _lam_s(rng, 3000)
    lnw = 0.1 + 0.3 / np.sqrt(lam_s / PIVOT) * rng.normal(size=lam_s.size)
    # P_CEN = 1, P_SAT = 0: the satellite and foreground Gaussians drop out, even uncalibrated.
    got = wcen.fit_lnw_cen(lnw, lam_s, 1.0, 0.0, None, (np.nan, np.nan), PIVOT)
    np.testing.assert_allclose(got, wcen.fit_lnw(lnw, lam_s, PIVOT), atol=2e-4)
    # 1 - P_CEN - P_SAT slightly negative (float32 rounding) is clipped to 0, not a NaN.
    got32 = wcen.fit_lnw_cen(lnw, lam_s, 1.0 + 1e-7, 0.0, None, None, PIVOT)
    np.testing.assert_allclose(got32, got, atol=2e-4)
    with pytest.raises(ValueError, match="lnw_sat"):
        wcen.fit_lnw_cen(lnw, lam_s, 0.9, 0.05, None, (0.0, 0.5), PIVOT)
    assert np.isnan(wcen.fit_lnw_cen([0.1], [30.0], 1.0, 0.0, None, None)).all()


# --------------------------------------------------------------------------- central magnitudes
def _clusters(rng, n, scaleval=1.15):
    """lambda (= S lambda/S, so the ln(lambda) and ln(lambda/S) conventions differ), lambda/S, m*."""
    lam_s = _lam_s(rng, n)
    return scaleval * lam_s, lam_s, rng.uniform(17.0, 21.0, n)


def test_fit_central_mag_bcg_case_with_outliers():
    rng = np.random.default_rng(13)
    n = 2000
    lam, lam_s, mstar = _clusters(rng, n)
    refmag = mstar + D0 + D1 * np.log(lam_s / PIVOT) + SIGMA_M * rng.normal(size=n)
    out = rng.random(n) < 0.04                           # wrong centrals, 2-4 mag off
    refmag[out] += rng.choice([-1.0, 1.0], out.sum()) * rng.uniform(2.0, 4.0, out.sum())
    cwt = wcen.chisq_pdf(rng.chisquare(3, n), 3)
    binf = rng.random(n) < 0.05                          # outside the chi^2 background table
    bcounts = np.where(binf, np.inf, rng.uniform(0.1, 5.0, n))
    # P_CEN = 1, P_SAT = 0 (BCG-centred calibration): rho = phicen cwt, floored at 1e-5.
    d0, d1, sm = wcen.fit_central_mag(refmag, mstar, lam, lam_s, 1.0, 0.0, cwt, bcounts, PHI1, PIVOT)
    assert d0 == pytest.approx(D0, abs=0.03)             # statistical errors 0.007, 0.014, 0.007
    assert d1 == pytest.approx(D1, abs=0.05)
    assert sm == pytest.approx(SIGMA_M, abs=0.025)
    ref = _redmapper_central_mag(refmag, mstar, lam, lam_s, np.ones(n), np.zeros(n), cwt, bcounts,
                                 PHI1, PIVOT)
    np.testing.assert_allclose((d0, d1, sm), ref, rtol=0, atol=1e-12)
    # Without the floor (a plain Gaussian fit) the outliers dominate the width.
    x = np.log(lam_s / PIVOT)
    resid = refmag - mstar - np.polyval(np.polyfit(x, refmag - mstar, 1), x)
    assert np.std(resid) > 0.5
    # 0 x inf = NaN is floored too: clusters with infinite background counts drop out.
    keep = ~binf
    alone = wcen.fit_central_mag(refmag[keep], mstar[keep], lam[keep], lam_s[keep], 1.0, 0.0,
                                 cwt[keep], bcounts[keep], PHI1, PIVOT)
    np.testing.assert_allclose((d0, d1, sm), alone, atol=2e-3)
    # phi1 is not needed when P_SAT = 0.
    np.testing.assert_allclose(
        wcen.fit_central_mag(refmag, mstar, lam, lam_s, 1.0, 0.0, cwt, bcounts, None, PIVOT),
        (d0, d1, sm), atol=1e-12)
    # Device scalars (e.g. read back from a WcenModel) do not turn the fit into float32 JAX.
    phi1_dev = {k: jnp.asarray(v) for k, v in PHI1.items()}
    got = wcen.fit_central_mag(refmag, mstar, lam, lam_s, 1.0, 0.0, cwt, bcounts, phi1_dev,
                               jnp.asarray(PIVOT))
    assert got == (d0, d1, sm) and all(type(v) is float for v in got)


def test_fit_central_mag_mixture():
    rng = np.random.default_rng(14)
    n = 4000
    lam, lam_s, mstar = _clusters(rng, n)
    pcen = rng.uniform(0.6, 1.0, n)
    psat = (1.0 - pcen) * rng.uniform(0.3, 0.7, n)
    comp = _mixture_labels(rng, pcen, psat)
    refmag = mstar + D0 + D1 * np.log(lam_s / PIVOT) + SIGMA_M * rng.normal(size=n)
    chisq = rng.chisquare(3, n)
    # Satellites: the brightest-satellite Gaussian, in ln(lambda/pivot) as phi1 in the fit.
    lnlam = np.log(lam / PIVOT)
    sat = comp == 1
    m_sat = (mstar + PHI1["PHI1_MMSTAR_M"] + PHI1["PHI1_MMSTAR_SLOPE"] * lnlam
             + (PHI1["PHI1_MSIG_M"] + PHI1["PHI1_MSIG_SLOPE"] * lnlam) * rng.normal(size=n))
    refmag[sat] = m_sat[sat]
    # Foreground: uniform in m* - 3..m* + 2 and chi^2 0..20, i.e. a density of 1/100 per mag per unit
    # chi^2, which makes rho a proper mixture density in (m, chi^2).
    fg = comp == 2
    refmag[fg] = mstar[fg] + rng.uniform(-3.0, 2.0, fg.sum())
    chisq[fg] = rng.uniform(0.0, 20.0, fg.sum())
    cwt, bcounts = wcen.chisq_pdf(chisq, 3), np.full(n, 0.01)
    d0, d1, sm = wcen.fit_central_mag(refmag, mstar, lam, lam_s, pcen, psat, cwt, bcounts, PHI1, PIVOT)
    assert d0 == pytest.approx(D0, abs=0.025)            # statistical errors 0.005, 0.007, 0.004
    assert d1 == pytest.approx(D1, abs=0.035)
    assert sm == pytest.approx(SIGMA_M, abs=0.02)
    ref = _redmapper_central_mag(refmag, mstar, lam, lam_s, pcen, psat, cwt, bcounts, PHI1, PIVOT)
    np.testing.assert_allclose((d0, d1, sm), ref, rtol=0, atol=1e-12)
    # Treating every chosen galaxy as a central biases the fit towards the satellites.
    d0n, _, smn = wcen.fit_central_mag(refmag, mstar, lam, lam_s, 1.0, 0.0, cwt, bcounts, PHI1, PIVOT)
    assert d0n > D0 + 0.03 and smn > SIGMA_M + 0.05
    with pytest.raises(ValueError, match="phi1"):
        wcen.fit_central_mag(refmag, mstar, lam, lam_s, pcen, psat, cwt, bcounts, None, PIVOT)


def test_chisq_pdf_and_background_counts():
    x = np.linspace(0.05, 40.0, 60)
    for k in (2, 3, 4):
        np.testing.assert_allclose(wcen.chisq_pdf(x, k), scipy.stats.chi2.pdf(x, k), rtol=1e-12)
        assert np.isnan(wcen.chisq_pdf(-1.0, k))         # failed chi^2: floored in the fit
    b = wcen.background_counts(np.array([2.0, np.inf]), 20.0, 1.0)
    assert b[0] == pytest.approx(2.0 / 20.0**2 * np.pi) and np.isinf(b[1])


# --------------------------------------------------------------------------- sample and randoms
def test_select_training_strict_inequalities():
    cfg = RemaConfig().replace(
        centering={"wcen_minlambda": 10.0, "wcen_maxlambda": 100.0, "wcen_cal_zrange": (0.05, 0.6)},
        mask={"max_maskfrac": 0.2})
    base = {"lam": 30.0, "scaleval": 1.0, "w": 0.5, "z": 0.3, "maskfrac": 0.0}
    cases = [
        ({}, True),
        ({"lam": 10.0}, False), ({"lam": 10.001}, True),
        ({"lam": 20.0, "scaleval": 2.0}, False),            # lambda/S = 10
        ({"lam": 100.0}, False), ({"lam": 99.99}, True),
        ({"lam": 150.0, "scaleval": 1.6}, True),            # the cut is on lambda/S = 93.75
        ({"w": 0.0}, False), ({"w": -0.2}, False), ({"w": 1e-6}, True),
        ({"z": 0.05}, False), ({"z": 0.0501}, True), ({"z": 0.6}, False), ({"z": 0.5999}, True),
        ({"maskfrac": 0.2}, False), ({"maskfrac": 0.1999}, True),
        ({"lam": np.nan}, False), ({"w": np.nan}, False), ({"z": np.nan}, False),
        ({"maskfrac": np.nan}, False),
    ]
    cols = {k: np.array([{**base, **change}[k] for change, _ in cases]) for k in base}
    sel = wcen.select_training(cols["lam"], cols["scaleval"], cols["w"], cols["z"], cols["maskfrac"], cfg)
    assert sel.dtype == bool
    np.testing.assert_array_equal(sel, [expect for _, expect in cases])


def test_random_offsets_uniform_in_area():
    rng = np.random.default_rng(15)
    n = 20_000
    r_lambda = rng.uniform(0.6, 1.2, n)
    mpc_per_deg = rng.uniform(5.0, 60.0, n)
    ra = rng.uniform(-0.5, 0.5, n) % 360.0                # straddles RA = 0
    dec = rng.uniform(-60.0, 10.0, n)
    ra_c, dec_c = wcen.random_offsets(np.random.default_rng(16), r_lambda, mpc_per_deg, ra, dec)
    assert np.all((ra_c >= 0.0) & (ra_c < 360.0))
    # Back to the flat-sky offsets in h^-1 Mpc.
    x = ((ra_c - ra + 180.0) % 360.0 - 180.0) * np.cos(np.radians(dec)) * mpc_per_deg
    y = (dec_c - dec) * mpc_per_deg
    q = np.hypot(x, y) / r_lambda
    assert q.max() <= 1.0 + 1e-9 and q.max() > 0.999
    # Uniform in area: P(r < sqrt(f) r_lambda) = f, and a uniform position angle.
    for f in (0.05, 0.25, 0.5, 0.75, 0.95):
        assert np.mean(q < np.sqrt(f)) == pytest.approx(f, abs=0.012)
    assert scipy.stats.kstest(q**2, "uniform").pvalue > 0.01
    assert scipy.stats.kstest((np.arctan2(y, x) + np.pi) / (2.0 * np.pi), "uniform").pvalue > 0.01
    # Reproducible; scalar inputs broadcast.
    again = wcen.random_offsets(np.random.default_rng(16), r_lambda, mpc_per_deg, ra, dec)
    np.testing.assert_array_equal(again[0], ra_c)
    ra1, dec1 = wcen.random_offsets(4, 0.5, 30.0, np.full(3, 10.0), -5.0)
    assert ra1.shape == dec1.shape == (3,)


def test_random_satellite_frequencies_follow_pmem():
    pmem = np.array([0.95, 0.6, 0.0, 0.3, 0.02, 0.0, 0.13, 0.0])
    n = 100_000
    idx = wcen.random_satellite(np.random.default_rng(17), np.tile(pmem, (n, 1)))
    freq = np.bincount(idx, minlength=pmem.size) / n
    expect = pmem / pmem.sum()
    assert np.all(np.abs(freq - expect) <= 4.0 * np.sqrt(expect * (1.0 - expect) / n))
    assert np.all(freq[pmem == 0.0] == 0.0)
    assert freq[4] > 0.005                               # pmem = 0.02: 1%, as pmem/sum(pmem)
    # Ragged rows; NaN and negative pmem count as 0; rows without members give -1.
    rows = [np.array([0.0, 0.0]), np.array([0.0, 0.7, 0.0]), np.array([]),
            np.array([np.nan, 0.4]), np.array([-1.0, 0.0, 0.2])]
    np.testing.assert_array_equal(wcen.random_satellite(5, rows), [-1, 1, -1, 1, 2])
    # One index per row whatever the container: no rows give an empty result, while a 1-D array
    # (numpy or JAX) or a flat list is one row, even without neighbours.
    for none in ([], (), iter([]), np.zeros((0, 4)), jnp.zeros((0, 4))):
        got = wcen.random_satellite(5, none)
        assert got.dtype == np.int64 and got.shape == (0,)
    for one in (np.array([0.0, 1.0]), jnp.asarray([0.0, 1.0]), [0.0, 1.0]):
        assert wcen.random_satellite(5, one).tolist() == [1]
    for empty_row in ([[]], np.zeros(0), jnp.zeros(0)):
        assert wcen.random_satellite(5, empty_row).tolist() == [-1]
    assert wcen.random_satellite(5, jnp.asarray([[0.0, 1.0], [2.0, 0.0]])).tolist() == [1, 0]
    assert wcen.random_satellite(5, iter(rows)).tolist() == [-1, 1, -1, 1, 2]
