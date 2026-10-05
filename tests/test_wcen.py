"""wcen centring (rema.core.centering) against numpy ports of redMaPPer (tests/wcen_reference.py)."""

import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import wcen_reference as ref
from rema.calibration import ZRMOD_PAD, ZlambdaCorrection
from rema.config import RemaConfig
from rema.core import centering as C
from rema.core.context import FilterModel
from rema.core.richness import Neighbors, RadialQuad, Richness, Stage, richness
from rema.core.zred import ZredGrid, compute_zred
from rema.model import profiles as prof
from rema.model.background import build_chisq_bkg, build_zred_bkg
from rema.model.cosmo import CosmoTable
from rema.model.profiles import MStar
from rema.model.redsequence import RSModel
from rema.modes import blind as B
from rema.modes.common import Region
from rema.sky.neighbors import NeighborIndex, unit_vectors
from rema.sky.regions import Box
from rema.validate import mocks

# Plausible calibrated values (DR10-like ln W of centrals, satellites and random points).
PARAMS = {"DELTA0": -1.5, "DELTA1": 0.0, "SIGMA_M": 0.3, "LNW_CEN_MEAN": 0.3, "LNW_CEN_SIGMA": 0.25,
          "LNW_SAT_MEAN": -0.2, "LNW_SAT_SIGMA": 0.35, "LNW_FG_MEAN": -0.6, "LNW_FG_SIGMA": 0.45}
OUT = ("p_cen", "q_cen", "p_sat", "p_fg", "p_c")


def _zlcorr(shift=0.02, amp=0.03):
    """A z_lambda correction whose z -> zred_uncorr mapping is well off the identity
    (zred_uncorr - z between shift - amp and shift + amp), on rema's z_lambda-correction nodes."""
    nodes = np.arange(0.05, 0.94, 0.08)
    zero = jnp.zeros(nodes.size)
    return ZlambdaCorrection(jnp.asarray(nodes), zero, zero, zero,
                             zred_uncorr=jnp.asarray(nodes + shift + amp * np.sin(6.0 * nodes)))


def _ref_table(zl):
    """redMaPPer's ZlambdaCorrectionPar table of the same mapping (float32 nodes, as stored)."""
    nodes = np.asarray(zl.z, np.float64)
    return ref.zlambda_corr_zred_uncorr(nodes, np.asarray(zl.zred_uncorr, np.float64),
                                        [nodes[0] - ZRMOD_PAD[0], nodes[-1] + ZRMOD_PAD[1]])


@pytest.fixture(scope="module")
def env():
    """Config, a filter model without red sequence (centring needs m*, D and alpha only) and a
    zred background from a synthetic field."""
    cfg = RemaConfig()
    ms = MStar(cfg.model.mstar)
    model = FilterModel(rs=None, bkg=None, cosmo=CosmoTable.create(), mstar_z=jnp.asarray(ms.z),
                        mstar_m=jnp.asarray(ms.m))
    rng = np.random.default_rng(3)
    n = 200_000
    a, b = 10 ** (0.4 * 13.0), 10 ** (0.4 * 24.5)
    mag = np.log10(a + rng.uniform(size=n) * (b - a)) / 0.4
    zbkg = build_zred_bkg(rng.uniform(0.0, 1.1, n), rng.uniform(0.0, 120.0, n), mag, lambda m: 10.0)
    return cfg, model, zbkg


def _params(rng):
    return {"DELTA0": rng.uniform(-2.0, -1.0), "DELTA1": rng.uniform(-0.3, 0.1),
            "SIGMA_M": rng.uniform(0.25, 0.5), "LNW_CEN_MEAN": rng.uniform(0.0, 0.4),
            "LNW_CEN_SIGMA": rng.uniform(0.2, 0.4), "LNW_SAT_MEAN": rng.uniform(-0.4, 0.0),
            "LNW_SAT_SIGMA": rng.uniform(0.25, 0.5), "LNW_FG_MEAN": rng.uniform(-0.8, -0.4),
            "LNW_FG_SIGMA": rng.uniform(0.3, 0.6)}


def _cluster(rng, model, K, n=None, z=None, lam=None):
    """One cluster: neighbour arrays [K] (float32; the first n valid, the rest copies of entry 0
    as rema's padding) and cluster scalars. Entry 0 is the seed at the position (r = 1e-6)."""
    z = np.float32(rng.uniform(0.1, 0.8) if z is None else z)
    D, mstar = float(model.mpc_per_deg(z)), float(model.mstar(z))
    lam = np.float32(rng.uniform(8.0, 150.0) if lam is None else lam)
    rl = np.float32((lam / 100.0) ** 0.2)
    n = int(rng.integers(K // 3, K)) if n is None else n
    # A concentrated and a uniform population, within 1.6 r_lambda of the position.
    rr = np.where(rng.uniform(size=n) < 0.4, 0.6 * rl * rng.uniform(size=n) ** 1.5,
                  1.6 * rl * np.sqrt(rng.uniform(size=n)))
    rr[0] = 0.0
    ang = rng.uniform(0, 2 * np.pi, n)
    ra0, dec0 = rng.uniform(0.0, 360.0), rng.uniform(-60.0, 60.0)
    dec = dec0 + rr / D * np.sin(ang)
    ra = ra0 + rr / D * np.cos(ang) / np.cos(np.radians(dec0))
    th = ref.sep_deg(unit_vectors(ra, dec), unit_vectors(ra0, dec0)[None])
    member = rng.uniform(size=n) < 0.6
    zred = np.where(member, z + rng.normal(0.0, 0.015, n), rng.uniform(0.02, 1.0, n))
    zred_e = rng.uniform(0.006, 0.04, n)
    zchi = rng.uniform(0.0, 30.0, n)
    hi = rng.uniform(size=n) < 0.05
    zchi[hi] = rng.uniform(100.0, 200.0, hi.sum())
    failed = rng.uniform(size=n) < 0.05
    zred[failed], zred_e[failed], zchi[failed] = -1.0, -1.0, -1.0
    r = np.maximum(th * D, 1e-6)
    p = np.where(member & (r < 2 * rl), rng.uniform(0.02, 1.0, n), 0.0)
    pmem = p * np.where(r < rl, rng.uniform(0.3, 1.0, n), 0.0)
    pfree = np.where(rng.uniform(size=n) < 0.8, 1.0, rng.uniform(0.0, 1.0, n))
    pfree[rng.uniform(size=n) < 0.03] = 0.5
    refmag = np.clip(mstar + rng.normal(0.8, 1.2, n), mstar - 3.5, mstar + 2.5)
    d = dict(xyz=unit_vectors(ra, dec), refmag=refmag, zred=zred, zred_e=zred_e, zred_chisq=zchi,
             r=r, p=p, pmem=pmem, pfree=pfree)
    d = {k: np.concatenate([v, np.repeat(v[:1], K - n, axis=0)]).astype(np.float32) for k, v in d.items()}
    d["valid"] = np.arange(K) < n
    d["is_center"] = np.arange(K) == 0
    return d, dict(z=z, Lambda=lam, scaleval=np.float32(rng.uniform(1.0, 1.4)), r_lambda=rl)


def _device(rows):
    """Batched device inputs (nb, xyz, zchi, rich, z) from a list of (neighbours, cluster)."""
    nbs, cls = zip(*rows)
    st = lambda k: jnp.asarray(np.stack([d[k] for d in nbs]))
    sc = lambda k: jnp.asarray(np.array([c[k] for c in cls], np.float32))
    B, K = len(rows), nbs[0]["refmag"].size
    zero, one = jnp.zeros((B, K)), jnp.ones((B, K))
    nb = Neighbors(theta=zero, refmag=st("refmag"), refmag_err=zero + 0.05, flux=jnp.zeros((B, K, 1)),
                   ivar=jnp.ones((B, K, 1)), zred=st("zred"), zred_e=st("zred_e"), pfree=st("pfree"),
                   valid=st("valid"), is_center=st("is_center"))
    rich = Richness(lam=sc("Lambda"), lam_e=jnp.zeros(B), r_lambda=sc("r_lambda"),
                    scaleval=sc("scaleval"), maskfrac=jnp.zeros(B), lnlamlike=jnp.zeros(B),
                    p=st("p"), pmem=st("pmem"), pcol=zero, theta_i=one, theta_r=one, chisq=zero,
                    r=st("r"))
    return nb, st("xyz"), st("zred_chisq"), rich, sc("z")


def _row(cen, b):
    return jax.tree_util.tree_map(lambda a: np.asarray(a)[b], cen)


def _lookup(zbkg):
    """The kernel's zred background as a numpy function (inputs padded to a few sizes, so that
    the eager lookup compiles once per size)."""
    def fn(zr, m):
        n = np.size(zr)
        N = max(256, 1 << int(np.ceil(np.log2(max(n, 1)))))
        pz, pm = np.zeros(N, np.float32), np.full(N, 20.0, np.float32)
        pz[:n], pm[:n] = zr, m
        return np.asarray(zbkg.lookup(jnp.asarray(pz), jnp.asarray(pm)))[:n]
    return fn


def _reference(model, wc, d, cl, params, maxrad=np.inf, ncand="model"):
    """The oracle on the same (float32) inputs; m*, D and the phi_sat normalisation as the kernel."""
    z = jnp.float32(cl["z"])
    cld = dict(cl, mstar=float(model.mstar(z)), mpc_scale=float(model.mpc_per_deg(z)), alpha=model.alpha,
               lumnorm=float(prof.lumnorm(model.mstar(z), model.maxmag(z), model.alpha)))
    cfgd = dict(pbcg_cut=wc.pbcg_cut, zred_chisq_max=wc.zred_chisq_max, rsoft=wc.rsoft,
                maxlambda=wc.maxlambda, pivot=wc.pivot, uselum=wc.uselum, maxcen=wc.maxcen)
    return ref.find_center(d, cld, params, cfgd, _lookup(wc.zbkg), maxrad=maxrad,
                           ncand=wc.ncand if ncand == "model" else ncand)


def _fragile(o, d, cl, wc, maxrad=np.inf, tol_d=1e-5, tol_e=1e-4, pairs=True):
    """True if a decision of the reference sits so close to its threshold that float32 rounding
    could flip it (such random configurations are skipped). ``pairs=False`` leaves out the
    softened member-candidate distances (for rsoft = r_lambda, which puts every pair on or beyond
    r_lambda in float32 and float64 alike)."""
    def near(x, t, tol):
        x = np.asarray(x, np.float64)
        return bool(np.any(np.abs(x / t - 1.0) < tol)) if x.size else False

    v = d["valid"]
    if near(d["r"][v], min(float(cl["r_lambda"]), maxrad), tol_d):
        return True
    zz = v & (d["zred_e"] > 0) & (d["pmem"] == 0)
    if near(np.abs(np.float64(cl["z"]) - d["zred"][zz]), 5.0 * np.float64(d["zred_e"][zz]), tol_d):
        return True
    g = o["dbg"]
    if "pdis" not in g:
        return False
    if pairs and near(g["pdis"], float(cl["r_lambda"]), tol_d):
        return True
    for x in (g["ucen_raw"], g["usat_raw"], g["s"] * g["fwmax"]):
        if near(x[np.isfinite(x) & (x > 0)], 1e-10, tol_e):
            return True
    pb = np.sort(g["Pcen_basic"][g["Pcen_basic"] > 0])[::-1][:wc.maxcen + 1]
    if np.any((np.diff(pb) != 0) & (np.abs(np.diff(pb)) < tol_e * pb[1:])):
        return True
    sc = np.sort(g["score"][g["pre"]])[::-1]
    return sc.size > wc.ncand and near(sc[wc.ncand], sc[wc.ncand - 1], tol_e)


def _check(c, o, atol=1e-5, rtol=1e-4):
    np.testing.assert_array_equal(c.index, o["index"])
    assert int(c.ngood) == o["ngood"]
    assert int(c.ncand) == o["ncand"]
    for k in OUT:
        np.testing.assert_allclose(getattr(c, k), o[k], rtol=rtol, atol=atol, err_msg=k)
    assert float(c.q_miss) == pytest.approx(o["q_miss"], rel=rtol, abs=atol)


def _compare(c, model, wc, d, cl, params, maxrad=np.inf, ncand="model"):
    """Check one cluster's output against the oracle, on a configuration that is not fragile."""
    o = _reference(model, wc, d, cl, params, maxrad=maxrad, ncand=ncand)
    assert not _fragile(o, d, cl, wc, maxrad=maxrad)
    _check(c, o)
    return o


def _draw(rng, env, K, nrows, params=None, static=None, **kw):
    """Random robust configurations: [(neighbours, cluster, params, wcen model, reference)].
    ``static`` replaces static fields of the wcen model."""
    cfg, model, zbkg = env
    rows = []
    while len(rows) < nrows:
        prm = _params(rng) if params is None else params
        wc = dataclasses.replace(C.WcenModel.create(prm, zbkg, cfg), **(static or {}))
        d, cl = _cluster(rng, model, K, **kw)
        o = _reference(model, wc, d, cl, prm)
        if not _fragile(o, d, cl, wc):
            rows.append((d, cl, prm, wc, o))
    return rows


# --------------------------------------------------------------------------- against the oracle
def test_center_wcen_matches_reference(env):
    _, model, _ = env
    rng = np.random.default_rng(1)
    rows = _draw(rng, env, 256, 40)
    ngood = []
    for d, cl, prm, wc, o in rows:
        c = _row(C.center_wcen(*_device([(d, cl)]), 1e3, model, wc), 0)
        _check(c, o)
        k = c.index[:c.ngood]
        assert np.all(d["valid"][k] & (d["zred_e"][k] > 0) & (d["pfree"][k] >= 0.5))
        assert np.all(np.diff(c.p_c[:c.ngood]) <= 0) and np.all(np.diff(c.p_cen[:c.ngood]) <= 0)
        if c.ngood:
            assert c.p_cen.sum() == pytest.approx(1.0, abs=1e-5)
            assert c.q_cen.sum() + c.q_miss == pytest.approx(1.0, abs=1e-5)
            psum = (c.p_cen + c.p_sat + c.p_fg)[:c.ngood]
            assert np.all((np.abs(psum - 1) < 1e-5) | (c.p_sat[:c.ngood] + c.p_fg[:c.ngood] == 0))
        ngood.append(int(c.ngood))
    assert max(ngood) == 5


def test_no_candidate(env):
    cfg, model, zbkg = env
    rng = np.random.default_rng(2)
    wc = C.WcenModel.create(PARAMS, zbkg, cfg)
    d, cl = _cluster(rng, model, 256)
    claimed = dict(d, pfree=np.full(256, 0.3, np.float32))           # every galaxy claimed
    bad = dict(cl, Lambda=np.float32(-1.0))                           # failed richness
    uncal = C.WcenModel.create(dict(PARAMS, LNW_CEN_SIGMA=-9999.0), zbkg, cfg)
    for dd, cc, w, ncand in ((claimed, cl, wc, 0), (d, bad, wc, 0), (d, cl, uncal, None)):
        c = _row(C.center_wcen(*_device([(dd, cc)]), 1e3, model, w), 0)
        assert c.ngood == 0 and np.all(c.index == -1) and c.q_miss == 1.0
        assert all(np.all(getattr(c, k) == 0) for k in OUT)
        if ncand is not None:
            assert c.ncand == ncand
        else:          # candidates, but none with P_C > 0 (redMaPPer's placeholders)
            assert c.ncand > 0
        _compare(c, model, w, dd, cc, dict(PARAMS, LNW_CEN_SIGMA=w.lnw_cen_sigma.item()))


def _isolate(d, keep):
    """Only the entries ``keep`` stay possible candidates (the others are claimed)."""
    d = dict(d, pfree=np.where(np.isin(np.arange(d["pfree"].size), keep), 1.0, 0.2).astype(np.float32))
    return d


def test_one_and_few_candidates(env):
    cfg, model, zbkg = env
    rng = np.random.default_rng(4)
    wc = C.WcenModel.create(PARAMS, zbkg, cfg)
    d, cl = _cluster(rng, model, 256, n=200, z=0.4, lam=50.0)
    z, mbar = float(cl["z"]), float(model.mstar(cl["z"])) + PARAMS["DELTA0"]
    # Bright red galaxies near the position: the obvious centres.
    near = np.flatnonzero(d["valid"] & (d["r"] < 0.3))[:4]
    d["refmag"][near] = mbar + np.array([0.0, 0.3, 0.6, 0.9], np.float32)[:near.size]
    d["zred"][near], d["zred_e"][near], d["zred_chisq"][near] = z, 0.02, 3.0
    for keep in (near[:1], near[:3]):
        dd = _isolate(d, keep)
        c = _row(C.center_wcen(*_device([(dd, cl)]), 1e3, model, wc), 0)
        _compare(c, model, wc, dd, cl, PARAMS)
        assert 1 <= c.ngood <= keep.size and c.ncand == keep.size
        assert np.all(c.index[c.ngood:] == -1)
        if keep.size == 1:      # P_CEN = 1, Q_CEN = P_C, Q_MISS = 1 - P_C
            assert c.index[0] == keep[0] and c.p_cen[0] == pytest.approx(1.0)
            assert c.q_cen[0] == pytest.approx(c.p_c[0], rel=1e-5)
            assert c.q_miss == pytest.approx(1.0 - c.p_c[0], rel=1e-4, abs=1e-6)


def test_pfree_and_maxrad_cuts(env):
    cfg, model, zbkg = env
    rng = np.random.default_rng(5)
    wc = C.WcenModel.create(PARAMS, zbkg, cfg)
    d, cl = _cluster(rng, model, 256, n=220, z=0.3, lam=60.0)
    z, mbar = float(cl["z"]), float(model.mstar(cl["z"])) + PARAMS["DELTA0"]
    best = int(np.flatnonzero(d["valid"] & (d["r"] > 0.05) & (d["r"] < 0.2))[0])
    d["refmag"][best], d["zred"][best], d["zred_e"][best], d["zred_chisq"][best] = mbar, z, 0.01, 2.0
    d["pmem"][best] = d["p"][best] = 0.9
    for pf, expect in ((0.5, True), (0.49, False)):          # pfree >= pbcg_cut (inclusive)
        dd = dict(d, pfree=d["pfree"].copy())
        dd["pfree"][best] = pf
        c = _row(C.center_wcen(*_device([(dd, cl)]), 1e3, model, wc), 0)
        _compare(c, model, wc, dd, cl, PARAMS)
        assert (best in c.index) == expect
    # Scan mode: candidates within maxrad of the position only.
    c = _row(C.center_wcen(*_device([(d, cl)]), 0.4, model, wc), 0)
    _compare(c, model, wc, d, cl, PARAMS, maxrad=0.4)
    k = c.index[:c.ngood]
    assert c.ngood > 0 and np.all(d["r"][k] < 0.4)
    full = _row(C.center_wcen(*_device([(d, cl)]), 1e3, model, wc), 0)
    assert full.ncand > c.ncand


def test_candidate_cap_is_exact(env):
    """More than ncand candidates, few of them able to reach ucen >= 1e-10: the capped kernel
    equals redMaPPer without a cap; with a binding cap it follows the documented rule."""
    cfg, model, zbkg = env
    rng = np.random.default_rng(6)
    K = 512
    d, cl = _cluster(rng, model, K, n=480, z=0.35, lam=120.0)
    z, ms = float(cl["z"]), float(model.mstar(cl["z"]))
    v = np.flatnonzero(d["valid"])
    d["pfree"][v] = 1.0
    d["zred"][v], d["zred_e"][v], d["zred_chisq"][v] = z + rng.normal(0, 0.01, v.size), 0.02, 5.0
    d["refmag"][v] = ms + 2.5                       # far too faint for a central ...
    bright = v[(d["r"][v] < 0.5 * float(cl["r_lambda"]))][:20]
    d["refmag"][bright] = ms + PARAMS["DELTA0"] + rng.uniform(-0.3, 1.2, bright.size)   # ... but these
    wc = C.WcenModel.create(PARAMS, zbkg, cfg)
    assert wc.ncand == 64
    c = _row(C.center_wcen(*_device([(d, cl)]), 1e3, model, wc), 0)
    assert c.ncand > 3 * wc.ncand and c.ngood >= 1
    _compare(c, model, wc, d, cl, PARAMS, ncand=None)
    # A cap below the number of possible centres.
    wc8 = dataclasses.replace(wc, ncand=8)
    c8 = _row(C.center_wcen(*_device([(d, cl)]), 1e3, model, wc8), 0)
    _compare(c8, model, wc8, d, cl, PARAMS)
    assert c8.ncand == c.ncand


# --------------------------------------------------------------------------- exact thresholds
# The random draws never land on a threshold (and _fragile skips those that come close), so the
# strict comparisons and the floors are checked here on constructed values, exact in float32 and
# float64 alike.
def _set(d, j, **cols):
    """A copy of the neighbour arrays with entry ``j`` of the columns ``cols`` replaced."""
    d = {k: v.copy() for k, v in d.items()}
    for k, v in cols.items():
        d[k][j] = v
    return d


def _same(a, b):
    """Bit-identical centring outputs."""
    jax.tree_util.tree_map(lambda x, y: np.testing.assert_array_equal(np.asarray(x), np.asarray(y)), a, b)


def test_candidate_cuts_are_strict(env):
    """A galaxy exactly on a candidate threshold is not a candidate (redMaPPer centering.py:162-166
    compares with <): r = r_lambda, r = maxrad (scan mode), zred chi^2 = zred_chisq_max and, with
    pmem = 0, |zred - z| = 5 zred_e on either side. The galaxy is a candidate just inside and none
    just outside; on the threshold the output equals the one outside. (pfree = pbcg_cut is a
    candidate: test_pfree_and_maxrad_cuts.)"""
    cfg, model, zbkg = env
    rng = np.random.default_rng(19)
    wc = C.WcenModel.create(PARAMS, zbkg, cfg)
    d, cl = _cluster(rng, model, 256, n=200, z=0.5, lam=60.0)
    z, rl = float(cl["z"]), float(cl["r_lambda"])
    mbar = float(model.mstar(jnp.float32(z))) + PARAMS["DELTA0"]
    j = int(np.flatnonzero(d["valid"] & (d["r"] > 0.1) & (d["r"] < 0.2))[0])
    d = _set(d, j, refmag=mbar, zred=z, zred_e=1 / 64, zred_chisq=2.0, p=0.9, pmem=0.9, pfree=1.0)
    cases = (  # (maxrad, {column: values of entry j inside, on and beyond the threshold})
        (1e3, dict(r=(0.5 * rl, rl, 1.5 * rl))),
        (0.25, dict(r=(0.125, 0.25, 0.375))),
        (1e3, dict(zred_chisq=(99.0, 100.0, 150.0))),
        (1e3, dict(zred=(z + 4 / 64, z + 5 / 64, z + 6 / 64), pmem=(0.0, 0.0, 0.0))),
        (1e3, dict(zred=(z - 4 / 64, z - 5 / 64, z - 6 / 64), pmem=(0.0, 0.0, 0.0))),
    )
    for maxrad, cols in cases:
        c = []
        for i in range(3):
            dd = _set(d, j, **{k: v[i] for k, v in cols.items()})
            c.append(_row(C.center_wcen(*_device([(dd, cl)]), maxrad, model, wc), 0))
            o = _reference(model, wc, dd, cl, PARAMS, maxrad=maxrad)
            assert i == 1 or not _fragile(o, dd, cl, wc, maxrad=maxrad)    # (the tie is on purpose)
            _check(c[i], o)
        inside, on, out = c
        assert inside.ncand == out.ncand + 1 and j not in on.index, cols
        _same(on, out)
        if "r" in cols or "zred_chisq" in cols:       # (a kept centre: the index shows it too)
            assert j in inside.index[:inside.ngood], cols


def test_connectivity_counts_a_member_at_the_candidate_position(env):
    """A distinct member at exactly the position of a candidate counts in its w, at the softened
    distance rsoft: only the candidate itself is left out (redMaPPer centering.py:203-207 drops
    self-pairs by identity, use[i1] != u[i2], not by distance)."""
    _, model, _ = env
    rng = np.random.default_rng(16)
    K, done = 256, 0
    for _ in range(40):
        (d, cl, prm, wc, o), = _draw(rng, env, K, 1)
        if o["ngood"] == 0:
            continue
        c0 = int(o["index"][0])
        j = int(rng.choice(np.flatnonzero(d["valid"] & (np.arange(K) > 0) & (np.arange(K) != c0))))
        # j: a bright member (never a candidate: pfree < pbcg_cut) moved onto the best candidate.
        alone = _set(d, j, xyz=d["xyz"][c0], r=d["r"][c0], refmag=float(model.mstar(cl["z"])) - 1.0,
                     pfree=0.2, p=0.0)
        both = _set(alone, j, p=0.9)
        o0, o1 = (_reference(model, wc, dd, cl, prm) for dd in (alone, both))
        if _fragile(o1, both, cl, wc) or c0 not in o1["index"]:
            continue
        # Without j in its w (as a distance-based exclusion would have it), P_C of c0 would differ.
        k0 = int(np.flatnonzero(o1["dbg"]["use"] == c0)[0])
        if abs(o1["dbg"]["Pcen_basic"][k0] - o0["dbg"]["Pcen_basic"][k0]) < 1e-2:
            continue
        _check(_row(C.center_wcen(*_device([(both, cl)]), 1e3, model, wc), 0), o1)
        done += 1
        if done == 6:
            break
    assert done == 6


def test_connectivity_radius_is_strict(env):
    """A member at softened distance exactly r_lambda from a candidate is not in its w (redMaPPer
    centering.py:218 keeps pdis < r_lambda). With rsoft = r_lambda = 0.75 (an exact square), a
    member at the exact position of the best candidate is on the threshold in float32 and float64
    alike, and every other pair beyond it: every candidate has the floor w = 1e-3 and the member
    changes nothing. (The ln w Gaussians are centred near ln 1e-3, so that candidates at the
    floor are centres; counted, the member would give w = 0 and P_C = 0 to the best one.)"""
    cfg, model, zbkg = env
    rng = np.random.default_rng(17)
    prm = dict(PARAMS, LNW_CEN_MEAN=-6.5, LNW_CEN_SIGMA=1.0, LNW_SAT_MEAN=-6.0, LNW_SAT_SIGMA=1.0,
               LNW_FG_MEAN=-5.5, LNW_FG_SIGMA=1.0)
    wc = dataclasses.replace(C.WcenModel.create(prm, zbkg, cfg), rsoft=0.75)
    d, cl = _cluster(rng, model, 256, n=200, z=0.4, lam=24.0)
    cl = dict(cl, r_lambda=np.float32(0.75))
    j = int(np.flatnonzero(d["valid"] & (np.arange(256) > 0))[0])
    alone = _set(d, j, pfree=0.2, p=0.0)              # j: neither a candidate nor (yet) a member
    o0 = _reference(model, wc, alone, cl, prm)
    assert not _fragile(o0, alone, cl, wc, pairs=False) and o0["ngood"] >= 2
    assert np.all(o0["dbg"]["w"] == 1e-3)
    c0 = int(o0["index"][0])
    both = _set(alone, j, xyz=alone["xyz"][c0], r=alone["r"][c0], p=0.9,
                refmag=float(model.mstar(cl["z"])) - 1.0)
    o1 = _reference(model, wc, both, cl, prm)
    assert np.all(o1["dbg"]["w"] == 1e-3)
    c = [_row(C.center_wcen(*_device([(dd, cl)]), 1e3, model, wc), 0) for dd in (alone, both)]
    _check(c[0], o0)
    _check(c[1], o1)
    _same(c[1], c[0])
    assert c[1].index[0] == c0 and c[1].p_c[0] > 0


def test_satellite_term_floor(env):
    """usat below 1e-10 is set to 0 (redMaPPer centering.py:259-260). With the satellite and
    foreground Gaussians far below the candidates' ln w, usat of the kept candidates falls in
    (0, 1e-10) and bcounts underflows to 0: both splits are 0/0, so P_SAT = P_FG = 0 (an
    unfloored usat would give P_SAT = 1 - P_CEN). lambda = 30 keeps sigscale near 1, where usat
    is below the floor but well above the float32 underflow."""
    _, model, _ = env
    rng = np.random.default_rng(18)
    prm = dict(PARAMS, LNW_SAT_MEAN=-3.0, LNW_SAT_SIGMA=0.3, LNW_FG_MEAN=-8.0, LNW_FG_SIGMA=0.05)
    nfloor = 0
    for d, cl, _, wc, o in _draw(rng, env, 256, 10, params=prm, lam=30.0):
        c = _row(C.center_wcen(*_device([(d, cl)]), 1e3, model, wc), 0)
        _check(c, o)
        n = o["ngood"]
        k = np.searchsorted(o["dbg"]["use"], o["index"][:n])       # kept, among the candidates
        usat, b = o["dbg"]["usat_raw"][k], o["dbg"]["bcounts"][k]
        floor = (usat > 1e-30) & (usat < 1e-10) & (b == 0) & (c.p_cen[:n] < 0.999)
        assert np.all(c.p_sat[:n][floor] == 0) and np.all(c.p_fg[:n][floor] == 0)
        nfloor += int(floor.sum())
    assert nfloor >= 25                   # of the <= 50 kept candidates


# --------------------------------------------------------------------------- batching and shapes
def test_batched_with_padding_equals_rows(env):
    cfg, model, zbkg = env
    rng = np.random.default_rng(7)
    rows = _draw(rng, env, 256, 5, params=PARAMS)
    wc = C.WcenModel.create(PARAMS, zbkg, cfg)
    data = [(d, cl) for d, cl, *_ in rows]
    bad = (data[1][0], dict(data[1][1], Lambda=np.float32(-1.0)))           # failed richness
    empty = (dict(data[2][0], valid=np.zeros(256, bool)), data[2][1])       # no neighbours
    data += [bad, empty, data[0]]                                           # row 7 pads with row 0
    cen = C.center_wcen(*_device(data), 1e3, model, wc)
    assert cen.index.shape == (8, 5) and cen.index.dtype == jnp.int32 and cen.ngood.shape == (8,)
    for b, row in enumerate(data):
        one = _row(C.center_wcen(*_device([row]), 1e3, model, wc), 0)
        got = _row(cen, b)
        for k in ("index", "ngood", "ncand"):
            np.testing.assert_array_equal(getattr(got, k), getattr(one, k))
        for k in OUT + ("q_miss",):
            np.testing.assert_allclose(getattr(got, k), getattr(one, k), rtol=1e-5, atol=1e-7)
        if b < 5:
            _check(got, rows[b][4])
    for b in (5, 6):
        assert cen.ngood[b] == 0 and np.all(np.asarray(cen.index[b]) == -1) and cen.q_miss[b] == 1
    jax.tree_util.tree_map(lambda a: np.testing.assert_array_equal(np.asarray(a)[7], np.asarray(a)[0]), cen)


def test_without_luminosity_weights(env):
    _, model, _ = env
    rng = np.random.default_rng(14)
    for d, cl, prm, wc, o in _draw(rng, env, 256, 6, static=dict(uselum=False)):
        assert not wc.uselum
        _check(_row(C.center_wcen(*_device([(d, cl)]), 1e3, model, wc), 0), o)


@pytest.mark.parametrize("ncand", [64, 256])
def test_small_k(env, ncand):
    """K = 128 neighbours (scan mode's smallest bucket), including K < ncand; B = 1."""
    cfg, model, zbkg = env
    rng = np.random.default_rng(8)
    for d, cl, prm, wc, _ in _draw(rng, env, 128, 4):
        wc = dataclasses.replace(wc, ncand=ncand)
        c = _row(C.center_wcen(*_device([(d, cl)]), 1e3, model, wc), 0)
        _compare(c, model, wc, d, cl, prm)


def test_as_centering():
    c = C.as_centering(jnp.asarray([3, 7, 0], jnp.int32), jnp.asarray([True, False, True]))
    np.testing.assert_array_equal(np.asarray(c.index), [[3, -1, -1, -1, -1], [-1] * 5, [0, -1, -1, -1, -1]])
    np.testing.assert_array_equal(np.asarray(c.p_cen)[:, 0], [1, 0, 1])
    np.testing.assert_array_equal(np.asarray(c.q_cen), np.asarray(c.p_cen))
    assert not np.any(np.asarray(c.p_sat)) and not np.any(np.asarray(c.p_fg)) and not np.any(np.asarray(c.p_c))
    np.testing.assert_array_equal(np.asarray(c.q_miss), [0, 1, 0])
    np.testing.assert_array_equal(np.asarray(c.ngood), [1, 0, 1])
    np.testing.assert_array_equal(np.asarray(c.ncand), [1, 0, 1])
    assert C.as_centering(jnp.zeros(2, jnp.int32), jnp.ones(2, bool), maxcen=3).p_cen.shape == (2, 3)


def test_model_and_calibrated_flag(env):
    cfg, _, zbkg = env
    wc = C.WcenModel.create({k.lower(): v for k, v in PARAMS.items()}, zbkg, cfg)
    assert wc.delta0.dtype == jnp.float32 and wc.delta0.shape == () and float(wc.delta0) == -1.5
    c = cfg.centering
    assert (wc.pivot, wc.rsoft, wc.maxcen, wc.ncand, wc.uselum, wc.pbcg_cut) == (
        c.wcen_pivot, c.wcen_rsoft, c.maxcen, c.wcen_ncand, c.wcen_uselum, c.pbcg_cut)
    assert C.WcenModel.create(dict(PARAMS, PIVOT=20.0), zbkg, cfg).pivot == 20.0
    # Same static configuration: one compilation serves every calibration.
    assert (jax.tree_util.tree_structure(wc)
            == jax.tree_util.tree_structure(C.WcenModel.create(dict(PARAMS, DELTA0=-1.0), zbkg, cfg)))
    # ... and every z -> zred_uncorr mapping on the same nodes: its table is a traced leaf.
    m1, m2 = (C.WcenModel.create(PARAMS, zbkg, cfg, zrmod=_zlcorr(shift=s).zrmod_table())
              for s in (0.0, 0.03))
    assert jax.tree_util.tree_structure(m1) == jax.tree_util.tree_structure(wc)
    assert ([a.shape for a in jax.tree_util.tree_leaves(m1)]
            == [a.shape for a in jax.tree_util.tree_leaves(m2)])
    assert not np.array_equal(np.asarray(m1.zrmod), np.asarray(m2.zrmod))
    assert m1.zrmod.dtype == jnp.float32 and m1.zrmod_z.shape == (485,)
    # Without one: the identity table.
    np.testing.assert_array_equal(np.asarray(wc.zrmod_z), [0.0, 1.0])
    np.testing.assert_array_equal(np.asarray(wc.zrmod), [0.0, 1.0])
    assert C.wcen_calibrated(PARAMS)
    assert not C.wcen_calibrated(None) and not C.wcen_calibrated({})
    assert not C.wcen_calibrated(dict(PARAMS, LNW_CEN_SIGMA=-9999.0))
    assert not C.wcen_calibrated(dict(PARAMS, SIGMA_M=0.0))
    assert not C.wcen_calibrated(dict(PARAMS, SIGMA_M=np.nan))
    assert not C.wcen_calibrated({k: v for k, v in PARAMS.items() if k != "LNW_FG_MEAN"})


# --------------------------------------------------------------------------- LNCGLIKE and W
def test_lncglike_matches_reference(env):
    cfg, model, zbkg = env
    rng = np.random.default_rng(9)
    rows, prm = [], []
    for _ in range(10):
        d, cl = _cluster(rng, model, 256)
        d["zred"][0], d["zred_e"][0] = cl["z"] + rng.normal(0, 0.02), rng.uniform(0.01, 0.03)
        d["refmag"][0] = float(model.mstar(cl["z"])) + rng.uniform(-2.5, 0.5)
        rows.append((d, cl))
    rows[5][0]["zred"][0] = rows[5][0]["zred_e"][0] = -1.0                 # failed zred: g = 1e-10
    # A valid zred 9 zred_e off z: g < 1e-12 is floored at 1e-10 (a shift of more than ln 100).
    d, cl = rows[6]
    d["zred_e"][0], d["zred"][0] = 0.01, cl["z"] + 0.09
    dz = (float(d["zred"][0]) - float(cl["z"])) / float(d["zred_e"][0])
    assert np.exp(-0.5 * dz**2) / (np.sqrt(2 * np.pi) * float(d["zred_e"][0])) < 1e-12
    # A second galaxy at the seed's position (r < 1e-5) is no member (a distance cut).
    d, cl = rows[7]
    dup = int(np.flatnonzero(d["valid"] & (d["pmem"] > 0) & (np.arange(256) > 0))[0])
    d["r"][dup], d["pmem"][dup], d["refmag"][dup] = d["r"][0], 0.9, float(model.mstar(cl["z"])) - 1.0
    rows[8] = (dict(rows[8][0], is_center=np.zeros(256, bool)), rows[8][1])  # no central
    rows[9] = (rows[9][0], dict(rows[9][1], Lambda=np.float32(-1.0)))      # failed richness
    wc = C.WcenModel.create(PARAMS, zbkg, cfg)
    nb, _, _, rich, z = _device(rows)
    got = np.asarray(C.lncglike(nb, rich, z, model, wc))
    assert got.shape == (10,) and got.dtype == np.float32
    cfgd = dict(rsoft=wc.rsoft, maxlambda=wc.maxlambda, pivot=wc.pivot)
    want = []
    for b, (d, cl) in enumerate(rows[:8]):
        zz = jnp.float32(cl["z"])
        want.append(ref.lnbcglike(d, dict(cl, mstar=float(model.mstar(zz))), PARAMS, cfgd, icen=0))
        assert np.isfinite(want[b]) and got[b] == pytest.approx(want[b], rel=1e-5, abs=1e-4)
    assert np.isnan(got[8]) and np.isnan(got[9])
    # The second galaxy matters: as a member (just beyond r = 1e-5) it changes LNCGLIKE by far
    # more than the tolerance.
    d, cl = rows[7]
    d = dict(d, r=np.where(np.arange(256) == dup, 2e-5, d["r"]).astype(np.float32))
    member = ref.lnbcglike(d, dict(cl, mstar=float(model.mstar(jnp.float32(cl["z"])))), PARAMS, cfgd,
                           icen=0)
    assert abs(member - want[7]) > 100 * (1e-4 + 1e-5 * abs(want[7]))
    # Members only beyond r_lambda can give w <= 0: NaN, as redMaPPer's log of a negative.
    d, cl = rows[0]
    far = dict(d, pmem=np.where(d["r"] > 0.95 * cl["r_lambda"], 0.5, 0.0).astype(np.float32))
    far["r"] = np.where(far["pmem"] > 0, 2.0 * cl["r_lambda"], far["r"]).astype(np.float32)
    nb, _, _, rich, z = _device([(far, cl)])
    assert np.isnan(np.asarray(C.lncglike(nb, rich, z, model, wc))[0])


# --------------------------------------------------------------------------- z -> zred_uncorr
def test_zrmod_table_matches_reference():
    """The table of the z -> zred_uncorr mapping is redMaPPer's ZlambdaCorrectionPar grid and
    natural spline, and the kernel's interpolation in it is ``interpol`` (linear, extrapolating
    the end segments)."""
    zl = _zlcorr()
    zz, zr = zl.zrmod_table()
    rz, rv = _ref_table(zl)
    np.testing.assert_allclose(zz, rz, rtol=0, atol=1e-12)
    np.testing.assert_allclose(zr, rv, rtol=0, atol=1e-12)
    assert zz.size == 485 and zz[0] == pytest.approx(0.03) and zz[-1] == pytest.approx(0.998)
    x = np.concatenate([np.linspace(-0.2, 1.3, 301), zz[:3], zz[-3:]])
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    got = np.asarray(jax.jit(C._interpol)(f32(zr), f32(zz), f32(x)))
    want = ref.interpol(rv, rz, x)
    inside = (x >= zz[0]) & (x <= zz[-1])
    assert inside.sum() > 100 and (~inside).sum() > 100
    np.testing.assert_allclose(got[inside], want[inside], rtol=0, atol=1e-6)
    # Far outside, the float32 slope of a 0.002 end segment limits the agreement.
    np.testing.assert_allclose(got[~inside], want[~inside], rtol=0, atol=1e-4)
    # Not fitted: no table (the wcen model then uses the identity).
    assert dataclasses.replace(zl, zred_uncorr=None).zrmod_table() is None


def test_zrmod_identity_is_exact():
    """Without a mapping, zrmod(z) = z bit for bit, inside and outside (0, 1)."""
    z = np.random.default_rng(20).uniform(-1.0, 3.0, 1000).astype(np.float32)
    zz, zr = (jnp.asarray(a, jnp.float32) for a in C.ZRMOD_IDENTITY)
    np.testing.assert_array_equal(np.asarray(C._interpol(zr, zz, jnp.asarray(z))), z)
    np.testing.assert_array_equal(np.asarray(jax.jit(C._interpol)(zr, zz, jnp.asarray(z))), z)


def test_lncglike_with_zred_uncorr_matches_reference(env):
    """LNCGLIKE with a z -> zred_uncorr mapping: the central's zred against zrmod(z), as
    redMaPPer's likelihood pass with a zlambdafile (run_likelihoods.py:213-219), also for z
    beyond the table (0.02 and 1.0: linear extrapolation)."""
    cfg, model, zbkg = env
    rng = np.random.default_rng(21)
    zl = _zlcorr(shift=0.04, amp=0.02)             # zred_uncorr - z between 0.02 and 0.06
    table = _ref_table(zl)
    wc0 = C.WcenModel.create(PARAMS, zbkg, cfg)
    wc = C.WcenModel.create(PARAMS, zbkg, cfg, zrmod=zl.zrmod_table())
    rows = []
    for zc in [*rng.uniform(0.1, 0.85, 10), 0.02, 1.0]:
        d, cl = _cluster(rng, model, 256, z=zc)
        # The central's zred near zrmod(z): its zred term is above the floor in both models.
        zrm = float(ref.interpol(table[1], table[0], float(cl["z"])))
        d["zred"][0], d["zred_e"][0] = zrm + rng.normal(0, 0.002), rng.uniform(0.01, 0.03)
        d["refmag"][0] = float(model.mstar(cl["z"])) + rng.uniform(-2.5, 0.5)
        rows.append((d, cl))
    nb, _, _, rich, z = _device(rows)
    got = np.asarray(C.lncglike(nb, rich, z, model, wc))
    ident = np.asarray(C.lncglike(nb, rich, z, model, wc0))
    cfgd = dict(rsoft=wc.rsoft, maxlambda=wc.maxlambda, pivot=wc.pivot)
    for b, (d, cl) in enumerate(rows):
        cld = dict(cl, mstar=float(model.mstar(jnp.float32(cl["z"]))))
        want = ref.lnbcglike(d, cld, PARAMS, cfgd, icen=0, zlambda_corr=table)
        want0 = ref.lnbcglike(d, cld, PARAMS, cfgd, icen=0)
        assert np.isfinite(want) and got[b] == pytest.approx(want, rel=1e-5, abs=1e-4)
        assert ident[b] == pytest.approx(want0, rel=1e-5, abs=1e-4)
        assert want - want0 > 100 * (1e-4 + 1e-5 * abs(want))          # the mapping matters


def test_center_wcen_ignores_zred_uncorr(env):
    """With a mapping in the model, the centring still compares zred with z (redMaPPer's
    percolation and zscan pass no z_lambda correction): bit-identical outputs."""
    cfg, model, zbkg = env
    rng = np.random.default_rng(22)
    rows = _draw(rng, env, 256, 6, params=PARAMS)
    args = _device([(d, cl) for d, cl, *_ in rows])
    wc0 = C.WcenModel.create(PARAMS, zbkg, cfg)
    wc1 = C.WcenModel.create(PARAMS, zbkg, cfg, zrmod=_zlcorr(shift=0.04).zrmod_table())
    c0, c1 = (C.center_wcen(*args, 1e3, model, w) for w in (wc0, wc1))
    _same(c0, c1)
    for b, row in enumerate(rows):
        _check(_row(c1, b), row[4])


def test_w_column_matches_reference():
    rng = np.random.default_rng(10)
    for uselum in (True, False):
        for _ in range(10):
            n = 300
            r = np.maximum(rng.uniform(0.0, 1.5, n), 1e-6)
            r[0] = 1e-6
            p = np.where(rng.uniform(size=n) < 0.7, rng.uniform(0.0, 1.0, n), 0.0)
            refmag = rng.uniform(17.0, 22.0, n)
            valid = rng.uniform(size=n) < 0.9
            valid[0] = True
            rl, ms = rng.uniform(0.6, 1.2), 19.5
            got = C.w_column(r, p, refmag, valid, ms, rl, rsoft=0.05, uselum=uselum)
            want = ref.w_catalog(r, p, refmag, ms, rl, rsoft=0.05, uselum=uselum, valid=valid)
            assert got == pytest.approx(want, rel=1e-12) and got > 0
    assert np.isnan(C.w_column(np.array([1e-6, 2.0]), np.array([1.0, 1.0]), np.array([18.0, 18.0]),
                               np.array([True, True]), 19.0, 1.0))
    assert np.isnan(C.w_column(np.zeros(3), np.ones(3), np.ones(3), np.zeros(3, bool), 19.0, 1.0))


# --------------------------------------------------------------------------- mock cluster
@pytest.mark.slow
def test_mock_cluster_central_is_recovered(env):
    """A bright central injected in a mock cluster is P_CEN[0], whether the richness is centred on
    it or on a satellite (as for a percolation seed)."""
    cfg, cosmo = env[0], env[1].cosmo
    rng = np.random.default_rng(12)
    rs, ms = RSModel.from_template(), MStar(cfg.model.mstar)
    depth5 = np.array([24.9, 24.7, 24.2, 23.6])
    box = Box(30.0, 30.7, -0.35, 0.35)
    z, lam, ra0, dec0 = 0.3, 40.0, 30.35, 0.0
    field = mocks.mock_field(rng, rs, box, density=12000, depth5=depth5, mag_range=(12.0, 22.5))
    cl = mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, ra0, dec0, z, lam, depth5, poisson=False,
                            central_dmag=-1.5)
    icen = field["RA"].size                        # the central is the cluster's first row
    gal = mocks.concat(field, cl)
    area = box.area_deg2()
    grid = ZredGrid.create(rs, ms, cosmo, cfg.model.zrange, cfg.model.zbin_coarse)
    zr = compute_zred(gal["FLUX"], gal["FLUX_IVAR"], grid, rs.iref, mode=cfg.model.chisq_mode,
                      eps=cfg.survey.flux_floor, alpha=cfg.model.alpha)
    bkg = build_chisq_bkg(gal["FLUX"], gal["FLUX_IVAR"], gal["REFMAG"], rs, ms, lambda m: area,
                          zrange=cfg.model.zrange, iref=rs.iref, mag_max=24.0, chunk=8192)
    model = FilterModel.create(rs, bkg, cfg, cosmo=cosmo, mstar=ms)
    wc = C.WcenModel.create(PARAMS, build_zred_bkg(zr.zred, zr.chisq, gal["REFMAG"], lambda m: area), cfg)
    # Seeds: the central, and the brightest member 0.15-0.5 h^-1 Mpc away from it.
    D = float(cosmo.mpc_per_deg(z))
    sep = ref.sep_deg(unit_vectors(gal["RA"], gal["DEC"]), unit_vectors(ra0, dec0)[None]) * D
    mem = np.arange(icen + 1, gal["RA"].size)
    sat = mem[(sep[mem] > 0.15) & (sep[mem] < 0.5)]
    sat = int(sat[np.argmin(gal["REFMAG"][sat])])
    seeds = np.array([icen, sat])
    stage, quad = Stage.make(1.0, 0.2), RadialQuad.make()
    pad = NeighborIndex(gal["RA"], gal["DEC"]).query(gal["RA"][seeds], gal["DEC"][seeds],
                                                     float(stage.maxrad) / D)
    i = pad.idx
    nb = Neighbors(theta=jnp.asarray(pad.theta, jnp.float32), refmag=jnp.asarray(gal["REFMAG"][i]),
                   refmag_err=jnp.asarray(gal["REFMAG_ERR"][i]), flux=jnp.asarray(gal["FLUX"][i]),
                   ivar=jnp.asarray(gal["FLUX_IVAR"][i]), zred=jnp.asarray(zr.zred[i]),
                   zred_e=jnp.asarray(zr.zred_e[i]), pfree=jnp.ones(i.shape, jnp.float32),
                   valid=jnp.asarray(pad.valid), is_center=jnp.asarray(pad.valid & (i == seeds[:, None])))
    zz = jnp.full(2, z, jnp.float32)
    ones = jnp.ones((2, quad.r.shape[0]))
    rich = richness(nb, zz, ones, ones, quad, model, stage)
    assert np.all(np.asarray(rich.lam) > 0.5 * lam)
    xyz = jnp.asarray(unit_vectors(gal["RA"][i], gal["DEC"][i]), jnp.float32)
    cen = C.center_wcen(nb, xyz, jnp.asarray(zr.chisq[i]), rich, zz, 1e3, model, wc)
    for b in range(2):
        assert i[b, int(cen.index[b, 0])] == icen
        assert float(cen.p_cen[b, 0]) > 0.5 and int(cen.ngood[b]) >= 1
    # LNCGLIKE prefers the seed that is the central; W at the central is a positive log.
    lncg = np.asarray(C.lncglike(nb, rich, zz, model, wc))
    assert np.all(np.isfinite(lncg)) and lncg[0] > lncg[1]
    w = C.w_column(rich.r[0], rich.p[0], gal["REFMAG"][i[0]], pad.valid[0], float(model.mstar(z)),
                   float(rich.r_lambda[0]), rsoft=wc.rsoft)
    assert np.isfinite(w) and w > 0


@pytest.mark.slow
def test_likelihood_pass_uses_zred_uncorr(env):
    """Region.build passes the calibration's z -> zred_uncorr mapping to the wcen model, and the
    blind likelihood pass centres the zred term of LNCGLIKE on zrmod(z): LNCGLIKE changes by
    exactly that term, lambda not at all."""
    cfg, cosmo = env[0], env[1].cosmo
    rng = np.random.default_rng(23)
    rs, ms = RSModel.from_template(), MStar(cfg.model.mstar)
    depth5 = np.array([24.9, 24.7, 24.2, 23.6])
    box = Box(30.0, 30.7, -0.35, 0.35)
    field = mocks.mock_field(rng, rs, box, density=12000, depth5=depth5, mag_range=(12.0, 22.5))
    cl = mocks.mock_cluster(rng, rs, ms, cosmo.mpc_per_deg, 30.35, 0.0, 0.4, 40.0, depth5,
                            poisson=False, central_dmag=-1.5)
    gal = mocks.concat(field, cl)
    gal["ID"] = np.arange(gal["RA"].size, dtype=np.int64)
    reg0 = Region.build(gal, rs, cfg, area_deg2=box.area_deg2(), wcen_params=PARAMS)
    zl = _zlcorr(shift=0.04, amp=0.02)
    reg1 = Region.build(reg0.gal, rs, cfg, area_deg2=box.area_deg2(), bkg=reg0.model.bkg,
                        zbkg=reg0.zbkg, zlcorr=zl, wcen_params=PARAMS)
    np.testing.assert_array_equal(np.asarray(reg0.wcen.zrmod_z), [0.0, 1.0])
    np.testing.assert_allclose(np.asarray(reg1.wcen.zrmod), zl.zrmod_table()[1], rtol=1e-6)
    # Seeds: the 48 brightest cluster galaxies with a zred (the central first), at their zred.
    g = reg0.gal
    mem = field["RA"].size + np.arange(cl["RA"].size)
    mem = mem[g["ZRED_E"][mem] > 0]
    seeds = mem[np.argsort(g["REFMAG"][mem], kind="stable")[:48]]
    z0 = g["ZRED"][seeds].astype(np.float32)
    st = Stage.make(1.0, 0.2, cfg.richness.maxrad_factor)
    q = RadialQuad.make(rmax=1.0 * 20.0**0.2 + 5 * cfg.model.rsig)
    lk0, lk1 = (B._run_batched(r, seeds, z0, st, q, "richness", log_every=0) for r in (reg0, reg1))
    np.testing.assert_array_equal(lk1["LAMBDA"], lk0["LAMBDA"])
    zred, ze = g["ZRED"][seeds].astype(np.float64), g["ZRED_E"][seeds].astype(np.float64)

    def ln_g(mu):
        v = -0.5 * ((zred - mu) / ze) ** 2 - np.log(np.sqrt(2 * np.pi) * ze)
        return np.maximum(v, np.log(1e-10))

    table = _ref_table(zl)
    z64 = z0.astype(np.float64)
    expect = ln_g(ref.interpol(table[1], table[0], z64)) - ln_g(z64)
    fin = np.isfinite(lk0["LNCGLIKE"])
    assert fin.sum() > 20 and np.array_equal(fin, np.isfinite(lk1["LNCGLIKE"]))
    got = lk1["LNCGLIKE"][fin].astype(np.float64) - lk0["LNCGLIKE"][fin]
    np.testing.assert_allclose(got, expect[fin], rtol=1e-4, atol=1e-3)
    assert np.median(np.abs(expect[fin])) > 0.3                      # the mapping matters


# --------------------------------------------------------------------------- GPU compile time
@pytest.mark.gpu
def test_gpu_cold_compile_time(env):
    """Cold compile of the blind-mode shape (B = 64, K = 2048) on a GPU stays well under a minute."""
    if jax.default_backend() != "gpu":
        pytest.skip("needs a CUDA device (JAX_PLATFORMS=cuda)")
    cfg, model, zbkg = env
    rng = np.random.default_rng(13)
    wc = C.WcenModel.create(PARAMS, zbkg, cfg)
    rows = [_cluster(rng, model, 2048, n=1500) for _ in range(64)]
    args = _device(rows) + (jnp.float32(1e3), model, wc)
    prev = jax.config.jax_enable_compilation_cache
    jax.config.update("jax_enable_compilation_cache", False)
    try:
        t0 = time.time()
        C.center_wcen.lower(*args).compile()
        dt = time.time() - t0
    finally:
        jax.config.update("jax_enable_compilation_cache", prev)
    assert dt < 60.0
