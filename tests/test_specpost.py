"""Spectroscopic post-processing against astropy and against a reference Clerc et al. (2016) clipping."""

import numpy as np
import pytest
from astropy.stats import biweight_location as bw_loc, biweight_scale as bw_scale

from rema.modes import specpost as S

C = S.C_KMS


# ---- reference recursive clipping (no bootstrap), kept as the test oracle ----------------
def _oracle_gapper(v):
    v = np.sort(v)
    n = len(v)
    w = np.arange(1, n) * np.arange(n - 1, 0, -1)
    return np.sqrt(np.pi) / (n * (n - 1)) * np.sum(w * np.diff(v))


def _oracle_clip(z, nsig=3, init_clip=5000, z_initial=None, depth=0, max_depth=20):
    if not isinstance(z, np.ma.MaskedArray):
        z = np.ma.masked_array(z)
    if depth == 0:
        if z_initial is None or z_initial < 0:
            z_initial = bw_loc(z.compressed(), 6)
        v = C * (z - z_initial) / (1 + z_initial)
        vclip = init_clip
    else:
        zbiwt = bw_loc(z.compressed(), 6)
        v = C * (z - zbiwt) / (1 + zbiwt)
        vc = v.compressed()
        vdisp = _oracle_gapper(vc) if vc.size < 15 else bw_scale(vc, 9)
        vclip = nsig * vdisp
    if depth >= max_depth or z.compressed().size <= 2:
        return z, zbiwt, v, vdisp, vclip
    z.mask = ~(np.absolute(v) <= np.absolute(vclip))
    if z.compressed().size <= 2:
        return None
    return _oracle_clip(z=z, nsig=nsig, depth=depth + 1)


def test_estimators_match_astropy(rng):
    x = rng.normal(0.3, 0.01, (40, 25))
    mask = rng.uniform(size=x.shape) > 0.2
    for i in range(x.shape[0]):
        xi = x[i][mask[i]]
        assert S.biweight_location(x[i:i + 1], mask[i:i + 1])[0] == pytest.approx(bw_loc(xi, 6), abs=1e-12)
        assert S.biweight_scale(x[i:i + 1], mask[i:i + 1], 9.0)[0] == pytest.approx(bw_scale(xi, 9), rel=1e-10)
        assert S.gapper(x[i:i + 1], mask[i:i + 1])[0] == pytest.approx(_oracle_gapper(xi), rel=1e-10)


@pytest.mark.parametrize("nmem", [3, 5, 10, 14, 15, 30, 80])
def test_clip_matches_oracle(nmem, rng):
    zc, sig = 0.35, 700.0
    for _ in range(20):
        v = rng.normal(0, sig, nmem)
        ninter = rng.integers(0, max(1, nmem // 4))
        v[:ninter] = rng.uniform(-6000, 6000, ninter)
        z = zc + v / C * (1 + zc)
        ref = _oracle_clip(z.copy())
        res = S.clip_velocity_batch(z[None, :], np.ones((1, nmem), bool))
        if ref is None:
            assert not res["ok"][0]
            continue
        zm, zb, vv, vdisp, vclip = ref
        assert res["ok"][0]
        assert res["zspec"][0] == pytest.approx(zb, abs=1e-12)
        assert res["vdisp"][0] == pytest.approx(vdisp, rel=1e-9)
        assert res["n_members"][0] == zm.compressed().size


def test_process_recovers_dispersion():
    rng = np.random.default_rng(3)
    ncl, nmem = 30, 40
    cat = {"MEM_MATCH_ID": np.arange(ncl), "Z_LAMBDA": np.full(ncl, 0.31), "Z_LAMBDA_E": np.full(ncl, 0.01),
           "ID_CENT": np.arange(ncl) * 1000}
    mid, zz, ids = [], [], []
    for c in range(ncl):
        v = rng.normal(0, 800.0, nmem)
        v[:5] = rng.uniform(-8000, 8000, 5)          # interlopers
        mid += [c] * nmem
        zz += list(0.3 + v / C * 1.3)
        ids += list(c * 1000 + np.arange(nmem))
    zz = np.array(zz)
    zz[np.arange(ncl * nmem) % 7 == 3] = -1.0          # members without spectroscopy
    mem = {"MEM_MATCH_ID": np.array(mid), "ID": np.array(ids), "ZSPEC": zz}
    out, mo = S.process(cat, mem, nboot=32, seed=1)
    assert np.all(out["BEST_Z_TYPE"] == "spec_z_boot")
    assert np.median(out["VDISP"]) == pytest.approx(800.0, rel=0.12)
    assert np.all(np.abs(out["SPEC_Z_BOOT"] - 0.3) < 0.002)
    assert np.all(out["VDISP_ERR"] > 0)
    out2, _ = S.process(cat, mem, nboot=32, seed=1)
    np.testing.assert_array_equal(out2["VDISP_BOOT"], out["VDISP_BOOT"])     # reproducible
    assert np.isfinite(mo["VEL"][mo["ISMEMBER_SPEC"]]).all()


def test_best_z_fallbacks():
    cat = {"MEM_MATCH_ID": np.array([1, 2, 3]), "Z_LAMBDA": np.array([0.2, 0.3, -1.0]),
           "Z_LAMBDA_E": np.array([0.01, 0.01, -1.0]), "ID_CENT": np.array([11, 21, 31])}
    mem = {"MEM_MATCH_ID": np.array([1, 1, 2]), "ID": np.array([11, 12, 21]),
           "ZSPEC": np.array([0.201, -1.0, -1.0])}
    out, _ = S.process(cat, mem, nboot=4)
    assert list(out["BEST_Z_TYPE"]) == ["cg_spec_z", "photo_z", "none"]
    assert out["BEST_Z"][0] == pytest.approx(0.201) and out["BEST_Z"][1] == pytest.approx(0.3)
