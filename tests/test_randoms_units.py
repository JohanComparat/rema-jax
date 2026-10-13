"""Pieces of the cluster randoms: domain, z_vlim map, position pool, weights, selections, specs."""

import numpy as np
import pytest

from rema.abundance.area import ZvlimMap
from rema.config import RemaConfig
from rema.model.profiles import MStar, maxmag_from_mstar
from rema.randoms.config import VlimConfig
from rema.randoms.domain import RunDomain
from rema.randoms.generate import FLAG_NOAREA, PositionPool, generate
from rema.randoms.products import info_header
from rema.randoms.samples import SampleSpec
from rema.randoms.selection import ExternalSelection, NoSelection, get_selection, register
from rema.randoms.selection.base import eddington_average
from rema.randoms.vlim import depth_mag, domain_fraction, zvlim_map_from_footprint
from rema.randoms.weigh import copy_weights, detection_q, effective_area
from rema.sky.maps import Footprint, SparseMap


# --------------------------------------------------------------------------- domain
def _two_parts(**kw):
    boxes = np.array([[350.0, 370.0, -10.0, 0.0], [10.0, 30.0, -10.0, 0.0], [350.0, 390.0, 0.0, 10.0]])
    return RunDomain(boxes, parts=np.array([0, 1, 1]), regions=np.array([0, 0, 1]), glat_min=0.0, **kw)


def test_domain_seams_and_membership():
    d = _two_parts()
    # Seams: RA 10 between the first two boxes (Dec -10..0), Dec 0 between part 0 and the top box.
    kinds = sorted((k, round(v, 6)) for k, v, _, _ in d.seams)
    assert kinds == [(0, 10.0), (1, 0.0)]
    ra = np.array([355.0, 5.0, 15.0, 20.0, 40.0])
    dec = np.array([-5.0, -5.0, -5.0, 5.0, -5.0])
    assert d.part_of(ra, dec).tolist() == [0, 0, 1, 1, -1]
    np.testing.assert_allclose(d.seam_distance(np.array([5.0]), np.array([-5.0])), [5.0 * np.cos(np.radians(5))],
                               rtol=1e-3)
    np.testing.assert_allclose(d.seam_distance(np.array([0.0]), np.array([-3.0])), [3.0])
    # The Dec seam stops at RA 10: beyond, the distance is to its end.
    assert d.seam_distance(np.array([20.0]), np.array([-1.0]))[0] > 9.0
    # A buffer removes the area near the seams; the coarse classification agrees with the exact test.
    b = _two_parts(seam_buffer_deg=1.0)
    rng = np.random.default_rng(3)
    ra, dec = rng.uniform(340, 400, 300_000) % 360, rng.uniform(-12, 12, 300_000)
    np.testing.assert_array_equal(b.contains(ra, dec), b.contains_exact(ra, dec))
    np.testing.assert_array_equal(d.contains(ra, dec), d.contains_exact(ra, dec))
    near = b.seam_distance(ra, dec) < 1.0
    assert not np.any(b.contains(ra, dec) & near) and np.any(d.contains(ra, dec) & near)


def test_domain_glat_and_io(tmp_path):
    d = RunDomain(np.array([[0.0, 360.0, -90.0, 90.0]]), [0], [0], glat_min=20.0)
    from rema.pipeline import galactic_latitude

    rng = np.random.default_rng(1)
    ra, dec = rng.uniform(0, 360, 200_000), np.degrees(np.arcsin(rng.uniform(-1, 1, 200_000)))
    np.testing.assert_array_equal(d.contains(ra, dec), np.abs(galactic_latitude(ra, dec)) >= 20)
    assert d.seams == [] and np.all(np.isinf(d.seam_distance(ra[:3], dec[:3])))
    _two_parts(seam_buffer_deg=0.5).write(tmp_path / "d.fits")
    e = RunDomain.read(tmp_path / "d.fits")
    assert e.digest() == _two_parts(seam_buffer_deg=0.5).digest() and e.seam_buffer_deg == 0.5


# --------------------------------------------------------------------------- z_vlim map
def _footprint(nside=64, sigf_z=None):
    """Full-sky footprint with a deep and a shallow hemisphere (Dec < 0 shallow)."""
    import healpy as hp

    pix = np.arange(12 * nside**2)
    _, dec = hp.pix2ang(nside, pix, nest=True, lonlat=True)
    deep = 10 ** (-0.4 * (24.0 - 22.5)) / 5          # 24 mag at 5 sigma
    shallow = 10 ** (-0.4 * (21.5 - 22.5)) / 5
    vals = {"FRACGOOD": np.where(np.abs(dec) < 80, 1.0, 0.0).astype(np.float32)}
    for b in "GRIZ":
        vals[f"SIGF_{b}"] = np.where(dec < 0, shallow, deep).astype(np.float32)
    return Footprint(SparseMap(nside, pix, vals), ("G", "R", "I", "Z"), 2500.0)


def test_zvlim_map_analytic():
    cfg = RemaConfig()
    fp = _footprint()
    zm = zvlim_map_from_footprint(fp, cfg, None, VlimConfig(nside=64))
    ms = MStar(cfg.model.mstar)
    z = np.asarray(ms.z)
    mlim = np.asarray(maxmag_from_mstar(np.asarray(ms.m), 0.2))
    _, dec = zm.radec()
    for depth, sel in ((24.0, dec > 0), (21.5, dec < 0)):
        np.testing.assert_allclose(zm.zvlim[sel], np.interp(depth, mlim, z), atol=1e-6)
    np.testing.assert_allclose(zm.extra["DEPTH_Z"][dec > 0], 24.0, atol=1e-4)
    # 10 sigma: brighter limit, lower z_vlim; 0.4 L*: brighter galaxies, higher z_vlim.
    z10 = zvlim_map_from_footprint(fp, cfg, None, VlimConfig(nside=64, nsigma=10.0))
    z04 = zvlim_map_from_footprint(fp, cfg, None, VlimConfig(nside=64, lstar_frac=0.4))
    assert np.all(z10.zvlim < zm.zvlim) and np.all(z04.zvlim > zm.zvlim)
    assert zm.area.sum() == pytest.approx(41253 * np.sin(np.radians(80)), rel=0.01)
    assert depth_mag(0.0, 5.0) != depth_mag(0.0, 5.0)                   # NaN without depth
    with pytest.raises(ValueError, match="NSIDE"):
        zvlim_map_from_footprint(fp, cfg, None, VlimConfig(nside=128))
    with pytest.raises(ValueError, match="density"):
        zvlim_map_from_footprint(fp, cfg, None, VlimConfig(nside=64), randoms=iter([]))


def test_zvlim_map_domain_fraction_and_io(tmp_path):
    cfg = RemaConfig()
    fp = _footprint()
    d = RunDomain(np.array([[0.0, 90.0, -30.0, 30.0]]), [0], [0], glat_min=0.0)
    zm = zvlim_map_from_footprint(fp, cfg, d, VlimConfig(nside=64))
    box = 90 * (np.degrees(np.sin(np.radians(30))) * 2)                  # 90 deg x 2 sin(30) rad
    assert zm.area.sum() == pytest.approx(box, rel=0.01)
    assert 0 < zm.extra["DOMFRAC"].min() < 1 and zm.extra["DOMFRAC"].max() == 1
    fr = domain_fraction(d, 64, np.array([0, 12 * 64**2 - 1]))
    assert fr.tolist() == [0.0, 0.0] or fr.max() <= 1
    zm.write(tmp_path / "z.fits")
    back = ZvlimMap.read(tmp_path / "z.fits")
    assert back.meta["NSIGMA"] == 5.0 and back.meta["DOMDIG"] == d.digest()
    np.testing.assert_array_equal(back.zvlim, zm.zvlim)
    assert back.select(back.zvlim > 0.5).meta == back.meta


# --------------------------------------------------------------------------- position pool
def test_pool_draws_uniformly_in_the_volume_limited_area():
    rng = np.random.default_rng(5)
    n = 200_000
    zv = rng.choice([0.3, 0.5, 0.7], n, p=[0.5, 0.3, 0.2])
    pool = PositionPool(rng.uniform(0, 10, n), rng.uniform(0, 10, n), zv)
    assert np.all(np.diff(pool.zvlim) >= 0) and pool.n_above(0.4) == np.sum(zv > 0.4)
    i = pool.draw(np.full(50_000, 0.45), rng)
    assert np.all(pool.zvlim[i] > 0.45)
    # Uniform over D(0.45): the 0.5 and 0.7 points in proportion 3:2, positions uniform.
    frac = np.mean(pool.zvlim[i] == 0.7)
    assert frac == pytest.approx(0.4, abs=0.01)
    from scipy.stats import kstest

    assert kstest(pool.ra[i] / 10, "uniform").pvalue > 1e-3
    assert pool.draw(np.array([0.8]), rng).tolist() == [-1]
    sub = pool.restrict(lambda ra, dec: ra < 5)
    assert sub.size == np.sum(pool.ra < 5) and sub.meta["NPOOL"] == sub.size


def test_generate_copies_and_flags():
    rng = np.random.default_rng(6)
    pool = PositionPool(np.arange(10.0), np.zeros(10), np.linspace(0.1, 1.0, 10))
    src = {"Z": np.array([0.15, 0.95, 1.5]), "LAMBDA": np.array([20.0, 30.0, 40.0])}
    rand, s = generate(src, pool, 7, rng, z_col="Z", carry=("Z", "LAMBDA"))
    assert s["N_SAMP"].tolist() == [7, 7, 0] and s["FLAG"].tolist() == [0, 0, FLAG_NOAREA]
    assert s["N_POOL"].tolist() == [9, 1, 0]
    assert rand["SRC"].tolist() == [0] * 7 + [1] * 7
    assert np.all(rand["ZVLIM"] > rand["Z"]) and np.all(rand["LAMBDA"][7:] == 30.0)
    assert np.all(rand["RA"][7:] == 9.0)                                 # the only deep enough point


def test_pool_from_chunks():
    cfg = RemaConfig()
    zm = ZvlimMap(4, np.arange(12 * 16), np.full(12 * 16, 1.0), np.full(12 * 16, 0.5))
    n = 1000
    rng = np.random.default_rng(0)
    r = {"RA": rng.uniform(0, 360, n), "DEC": rng.uniform(-60, 60, n), "MASKBITS": np.zeros(n, np.int16),
         "EBV": np.zeros(n)}
    for b in "GRIZ":
        r[f"NOBS_{b}"] = np.full(n, 2)
        r[f"GALDEPTH_{b}"] = np.full(n, 100.0)
    r["MASKBITS"][:100] = 2
    pool = PositionPool.from_chunks([r], cfg, zm, extra_mask=lambda ra, dec: dec > 0,
                                    max_points=200, rng=rng)
    assert pool.size == 200 and np.all(pool.dec > 0) and pool.meta["NREAD"] == n


# --------------------------------------------------------------------------- weights
def test_weights_reduce_to_redmapper():
    """With q in {0, 1}, w = n_samp / n_keep for the kept copies (RandomWeigher)."""
    rng = np.random.default_rng(8)
    src = np.repeat(np.arange(50), 30)
    q = (rng.uniform(size=src.size) < 0.7).astype(float)
    w, d = copy_weights(src, q, np.full(50, 30))
    nkeep = np.bincount(src, weights=q)
    np.testing.assert_allclose(w[q > 0], (30 / nkeep)[src[q > 0]])
    assert np.all(w[q == 0] == 0) and d["NPASS"].tolist() == nkeep.astype(int).tolist()
    np.testing.assert_allclose(d["NEFF"], nkeep)
    # A source with no passing copy gets no weight.
    w, d = copy_weights(np.array([0, 0, 1]), np.array([0.0, 0.0, 1.0]), np.array([2, 1]))
    assert w.tolist() == [0.0, 0.0, 1.0] and d["SUMQ"].tolist() == [0.0, 1.0]


def test_detection_and_effective_area():
    q = detection_q([20, 20, 20, 20], [1.0, 8.0, np.inf, 1.0], [0.1, 0.1, 0.1, 0.3], 3.0, 0.2)
    assert q.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert detection_q([20, 20], [5.0, 5.0], [0, 0], np.array([3.0, 5.0]), 0.2).tolist() == [1.0, 0.0]
    a = effective_area([0.1, 0.3, 0.5], np.array([0.2, 0.2, 0.4, 0.4, 0.6]), np.array([1, 0, 1, 1, 1.0]),
                       lambda z: np.array([100.0, 50.0]))
    np.testing.assert_allclose(a, [50.0, 50.0])


# --------------------------------------------------------------------------- selections
def test_selection_registry_and_base():
    with pytest.raises(KeyError, match="unknown selection"):
        get_selection("nope")
    assert isinstance(get_selection("none"), NoSelection)

    @register("toy_test")
    class Toy(ExternalSelection):
        columns = ("SNR",)

        def __init__(self, cut=5.0):
            self.cut = cut

        def own_cut(self, cat):
            return np.asarray(cat["SNR"]) >= self.cut

    t = get_selection({"name": "toy_test", "cut": 4.0})
    assert t.cut == 4.0 and t.own_cut({"SNR": np.array([3.0, 4.5])}).tolist() == [False, True]
    base = ExternalSelection()
    assert base.in_footprint(np.zeros(3), np.zeros(3)).all() and base.p_detect({}, np.zeros(2), np.zeros(2)).tolist() == [1, 1]
    assert base.own_cut({"A": np.zeros(4)}).all() and base.provenance() == {"SELNAME": "base", "APPROX": False}


def test_eddington_average_against_sampling():
    from scipy.special import ndtr

    rng = np.random.default_rng(9)
    alpha, sig, y_hat = 3.0, 1.0, 6.0
    p = lambda y: ndtr(y - 5.0)
    # Brute force: draw y from the prior x likelihood by importance sampling.
    y = rng.normal(y_hat, sig * 3, 2_000_000)
    y = y[y > 1.0]                     # the power law is a prior near y_hat, not down to 0
    w = np.exp(-0.5 * ((y - y_hat) / sig) ** 2) * y ** (-alpha) / np.exp(-0.5 * ((y - y_hat) / (3 * sig)) ** 2)
    ref = np.sum(w * p(y)) / np.sum(w)
    got = eddington_average(p, np.array([y_hat]), sig, alpha, n=32)[0]
    assert got == pytest.approx(ref, abs=2e-3) and got < p(y_hat)


# --------------------------------------------------------------------------- specs and products
def test_sample_spec(tmp_path):
    s = SampleSpec(cuts=("snr >= 5", "FLAG == 0"), selection={"name": "none"})
    cat = {"RA": np.zeros(3), "SNR": np.array([4.0, 6.0, 7.0]), "FLAG": np.array([0, 0, 1])}
    assert s.apply_cuts(cat).tolist() == [False, True, False]
    assert {"SNR", "FLAG", "SCALEVAL", "MASKFRAC"} <= set(s.columns(get_selection(None)))
    with pytest.raises(ValueError, match="operator"):
        SampleSpec(cuts=("SNR ~ 5",)).apply_cuts(cat)
    (tmp_path / "s.yaml").write_text(s.to_yaml())
    assert SampleSpec.from_yaml(tmp_path / "s.yaml") == s
    cfg = RemaConfig()
    b = SampleSpec(obs_edges=(20.0, 40.0, 100.0), detection_from_bin=True)
    assert b.detection_threshold(np.array([0, 1]), cfg).tolist() == [20.0, 40.0]
    assert SampleSpec(lambda_det=5.0).detection_threshold(np.array([0]), cfg).tolist() == [5.0]
    assert b.maskfrac_max(cfg) == 0.2 and SampleSpec(max_maskfrac=0.1).maskfrac_max(cfg) == 0.1


def test_info_header():
    h = {"XTENSION": "BINTABLE", "NAXIS2": 3, "TFORM1": "D", "TTYPE1": "RA", "EXTNAME": "R",
         "SAMPLE": "x", "NSAMP": 4, "TDIM3": "(2)"}
    assert info_header(h) == {"SAMPLE": "x", "NSAMP": 4}


# --------------------------------------------------------------------------- external selections
def test_act_selection_two_noise_levels(tmp_path):
    """P = Phi(y / sigma - q): the density ratio of the copies between a shallow and a deep half
    is the ratio of the two probabilities."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy.special import ndtr

    from rema.randoms.selection.act import ACTDR6Selection
    from rema.randoms.selection.carmaps import images_to_sparsemap

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    w.wcs.crval, w.wcs.crpix, w.wcs.cdelt = [10, 0], [101, 101], [-0.01, 0.01]
    img = np.full((200, 200), 1.0)
    img[:, :100] = 2.0                      # x < 100 is RA > 10: twice the noise
    img[:5] = 0.0                           # a masked edge
    fits.PrimaryHDU(img, header=w.to_header()).writeto(tmp_path / "rms.fits")
    rms = images_to_sparsemap([tmp_path / "rms.fits"], 512, "RMS")
    rms.hdu("RMS")
    from rema.io.tables import write_fits

    write_fits(tmp_path / "selfn.fits", [rms.hdu("RMS")])
    sel = get_selection({"name": "act_dr6", "selfn": str(tmp_path / "selfn.fits"), "q_cut": 5.0})
    ra, dec = np.array([9.5, 10.5, 30.0]), np.array([0.0, 0.0, 0.0])
    assert sel.in_footprint(ra, dec).tolist() == [True, True, False]
    np.testing.assert_allclose(sel.sigma(ra, dec), [1.0, 2.0, 0.0])
    src = {"EXT_FIXED_Y_C": np.full(3, 11.0), "EXT_FIXED_SNR": np.full(3, 11.0)}
    p = sel.p_detect(src, ra, dec)
    np.testing.assert_allclose(p, [ndtr(11 - 5), ndtr(5.5 - 5), 0.0])
    cat = {"EXT_FIXED_SNR": np.array([4.0, 6.0, 6.0]), "EXT_FLAGS": np.array([0, 0, 1])}
    assert sel.own_cut(cat).tolist() == [False, True, False]
    # Eddington: lower than the plain probability for a falling population near the threshold.
    e = ACTDR6Selection(rms_map=rms, q_cut=5.0, eddington_alpha=3.0)
    near = {"EXT_FIXED_Y_C": np.array([6.0]), "EXT_FIXED_SNR": np.array([6.0])}
    assert e.p_detect(near, ra[:1], dec[:1])[0] < sel.p_detect(near, ra[:1], dec[:1])[0]
    assert sel.provenance()["QCUT"] == 5.0
    with pytest.raises(ValueError, match="reducer"):
        images_to_sparsemap([tmp_path / "rms.fits"], 64, "RMS", reducer="max")
    assert images_to_sparsemap([tmp_path / "rms.fits"], 64, "RMS", reducer="mean").values["RMS"].max() == 2.0


def test_spt_selection(tmp_path):
    from scipy.special import ndtr

    fields = {"FIELD": np.array(["ra5h30dec-55", "ra23hdec-35"]), "RA0": np.array([70.0, 330.0]),
              "RA1": np.array([95.0, 370.0]), "DEC0": np.array([-60.0, -40.0]), "DEC1": np.array([-50.0, -30.0]),
              "GAMMA": np.array([1.6, 1.0])}
    sel = get_selection({"name": "spt", "fields": fields, "xi_cut": 5.0})
    ra, dec = np.array([80.0, 5.0, 200.0]), np.array([-55.0, -35.0, 0.0])
    assert sel.in_footprint(ra, dec).tolist() == [True, True, False]
    src = {"EXT_XI": np.full(3, 7.0), "EXT_FIELD": np.array(["ra23hdec-35"] * 3)}
    zeta = np.sqrt(49 - 3)
    np.testing.assert_allclose(sel.p_detect(src, ra, dec),
                               [ndtr(np.sqrt((1.6 * zeta) ** 2 + 3) - 5), ndtr(7 - 5), 0.0])
    assert sel.own_cut({"EXT_XI": np.array([4.0, 6.0])}).tolist() == [False, True]
    from astropy.table import Table

    Table(fields).write(tmp_path / "f.fits")
    assert get_selection({"name": "spt", "fields": str(tmp_path / "f.fits")}).gamma.tolist() == [1.6, 1.0]


def test_psz2_selection():
    import healpy as hp

    nside = 32
    l, b = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    mask = (np.abs(b) > 20).astype(float)                      # a Galactic-plane mask
    sel = get_selection({"name": "psz2", "mask_map": mask, "snr_min": 6.0})
    from rema.pipeline import galactic_latitude

    rng = np.random.default_rng(2)
    ra, dec = rng.uniform(0, 360, 2000), np.degrees(np.arcsin(rng.uniform(-1, 1, 2000)))
    inside = sel.in_footprint(ra, dec)
    gb = np.abs(galactic_latitude(ra, dec))
    assert np.mean(inside == (gb > 20)) > 0.97                  # pixelisation at the edge
    assert np.all(sel.p_detect({}, ra, dec) == 1)
    assert sel.own_cut({"EXT_SNR": np.array([5.0, 7.0])}).tolist() == [False, True]


def test_external_join_checks():
    from rema.randoms.external import join_scan

    scan = {"INPUT_ROW": np.array([0, 1]), "INPUT_NAME": np.array(["a", "x"])}
    orig = {"NAME": np.array(["a", "b"]), "SNR": np.ones(2), "Y5R500": np.ones(2), "COSMO": np.ones(2),
            "REDSHIFT": np.ones(2), "MSZ": np.ones(2)}
    with pytest.raises(ValueError, match="names differ"):
        join_scan(scan, orig, "psz2")
    with pytest.raises(ValueError, match="outside"):
        join_scan({"INPUT_ROW": np.array([5])}, orig, "psz2")
    with pytest.raises(KeyError, match="unknown catalogue"):
        join_scan(scan, orig, "nope")


def test_erass1_model_recovers_injected_parameters():
    from scipy.special import ndtr

    from rema.randoms.selection.erass1 import ERASS1ApproxSelection, HealpixColumn, fit_detection_model

    rng = np.random.default_rng(12)
    n = 40000
    t4 = rng.uniform(300, 3000, n)
    exp = 0.27 * t4 * rng.normal(1, 0.02, n)
    cr = np.exp(rng.normal(-2.5, 1.2, n))
    theta = np.exp(rng.normal(1.0, 0.4, n))
    nh = rng.uniform(1e20, 1e21, n)
    f500 = cr * 6.0
    lnl = -1.4 + 1.05 * np.log(cr * exp) - 0.45 * np.log(theta) + rng.normal(0, 0.6, n)
    det = lnl >= np.log(3.0)
    m = fit_detection_model(cr[det], f500[det], np.exp(lnl[det]), exp[det], t4[det], nh[det], theta[det])
    assert m["converged"] and not m["n_abs_fitted"] and m["n_abs"] == 1.7e21
    assert m["exp_scale"] == pytest.approx(0.27, rel=0.01)
    assert m["c1"] == pytest.approx(1.05, abs=0.03) and m["c2"] == pytest.approx(-0.45, abs=0.05)
    assert m["sigma"] == pytest.approx(0.6, abs=0.03) and m["c0"] == pytest.approx(-1.4, abs=0.15)
    # Absorption is fitted when the count rate falls with nH.
    m2 = fit_detection_model(cr[det], cr[det] * 6.0 * np.exp(nh[det] / 2e21), np.exp(lnl[det]), exp[det],
                             t4[det], nh[det], theta[det])
    assert m2["n_abs_fitted"] and m2["n_abs"] == pytest.approx(2e21, rel=0.01)
    # The plugin: P at the cluster's own position, and in a deeper field.
    import healpy as hp

    npix = hp.nside2npix(8)
    sel = ERASS1ApproxSelection(model=m, texp=HealpixColumn(values=np.full(npix, 1000.0)),
                                nh=HealpixColumn(values=np.full(npix, 3e20)), ext_like_cut=6.0)
    src = {"EXT_CR500": np.array([0.02]), "EXT_NH": np.array([3e20]), "EXT_THETA500": np.array([2.0]),
           "EXT_EXT_LIKE": np.array([7.0])}
    c = 0.02 * m["exp_scale"] * 1000.0
    expect = ndtr((m["c0"] + m["c1"] * np.log(c) + m["c2"] * np.log(2.0) - np.log(6.0)) / m["sigma"])
    np.testing.assert_allclose(sel.p_detect(src, np.array([200.0]), np.array([-40.0])), [expect])
    assert sel.own_cut(src).tolist() == [True] and sel.approximate and sel.provenance()["APPROX"]
    ra, dec = np.array([200.0, 30.0]), np.array([-40.0, 40.0])     # l ~ 309 and l ~ 140
    assert sel.in_footprint(ra, dec).tolist() == [True, False]


def test_build_sample_with_a_stricter_map_and_domain():
    from rema.randoms.samples import build_sample
    from rema.randoms.selection import NoSelection

    src = {"ROW": np.array([10, 11]), "Z": np.array([0.3, 0.6]), "LAMBDA": np.array([30.0, 30.0]),
           "N_SAMP": np.array([4, 4])}
    rand = {"SRC": np.repeat([0, 1], 4), "RA": np.array([1.0, 1.0, 3.0, 3.0, 1.0, 3.0, 3.0, 3.0]),
            "DEC": np.zeros(8), "SCALEVAL": np.ones(8), "MASKFRAC": np.zeros(8)}
    lam_det = np.zeros(20)
    idx, w, d = build_sample(src, rand, np.array([10, 11]), lam_det, NoSelection(), 0.2)
    assert np.all(w["WEIGHT"] == 1.0)
    # The stricter map: z_vlim 0.5 at RA < 2, 0.9 beyond; the domain drops RA > 3.5.
    zm = ZvlimMap(4, np.arange(192), np.ones(192), np.where(ZvlimMap(4, np.arange(192), np.ones(192),
                  np.zeros(192)).radec()[0] < 2, 0.5, 0.9))
    idx, w, d = build_sample(src, rand, np.array([10, 11]), lam_det, NoSelection(), 0.2, zmap=zm)
    ra_pix = zm.value_at(rand["RA"], rand["DEC"])
    expect_q = (ra_pix > np.repeat(src["Z"], 4)).astype(float)
    np.testing.assert_array_equal(w["Q"], expect_q)
    np.testing.assert_allclose(np.bincount(rand["SRC"], weights=w["WEIGHT"]), [4.0, 4.0])
    dom = RunDomain(np.array([[0.0, 2.0, -5.0, 5.0]]), [0], [0], glat_min=0.0)
    idx, w, d = build_sample(src, rand, np.array([10, 11]), lam_det, NoSelection(), 0.2, domain=dom)
    assert w["Q"].tolist() == [1, 1, 0, 0, 1, 0, 0, 0]
    assert d["NPASS"].tolist() == [2, 1]


def test_spt_fields_from_centres():
    from pathlib import Path

    csv = Path(__file__).resolve().parents[1] / "scripts" / "randoms" / "spt_sz_2500d_fields.csv"
    sel = get_selection({"name": "spt", "fields": str(csv)})
    assert sel.name_f.size == 19 and sel.gamma[0] == pytest.approx(1.3267815)
    ra, dec = np.array([82.7, 352.0, 15.0, 150.0, 82.7]), np.array([-55.0, -55.5, -42.0, -50.0, -30.0])
    f = sel.field_of(ra, dec)
    assert sel.name_f[f[:3]].tolist() == ["ra5h30dec-55", "ra23h30dec-55", "ra1hdec-42.5"]
    assert f[3:].tolist() == [-1, -1]                       # outside R.A. 20h-7h, and north of -40


def test_act_rms_map_from_stitched_images(tmp_path, monkeypatch):
    """The stitched CAR images (noise, area, flags) to the calibrated HEALPix noise map."""
    from astropy.io import fits
    from astropy.wcs import WCS

    from rema import cli
    from rema.io.tables import write_table
    from rema.randoms.selection.act import build_rms_map
    from rema.randoms.selection.erass1 import HealpixColumn

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    w.wcs.crval, w.wcs.crpix, w.wcs.cdelt = [10, 0], [101, 101], [-0.01, 0.01]
    rms = np.full((200, 200), 1.0e-5, np.float32)
    rms[:, :100] = 2.0e-5                                      # RA > 10: noisier
    area = np.ones((200, 200), np.uint8)
    area[:80] = 0                                              # Dec < -0.2: outside
    flag = np.zeros((200, 200), np.uint8)
    flag[150:] = 1                                             # Dec > 0.5: flagged
    for name, a in (("rms", rms), ("area", area), ("flag", flag)):
        fits.PrimaryHDU(a, header=w.to_header()).writeto(tmp_path / f"{name}.fits")
    m = build_rms_map(tmp_path / "rms.fits", tmp_path / "area.fits", tmp_path / "flag.fits", nside=256, chunk_rows=37)
    v = m.lookup_np(np.array([9.5, 10.5, 9.5, 9.5]), np.array([0.0, 0.0, -0.7, 0.85]), "RMS")
    np.testing.assert_allclose(v, [1e-5, 2e-5, 0, 0], rtol=1e-6)
    # The command: in units of 1e-4, calibrated on a catalogue whose fixed_SNR is 1% above y / RMS.
    write_table(tmp_path / "cat.fits", {"RADEG": np.array([9.5, 10.5]), "DECDEG": np.zeros(2),
                                        "FLAGS": np.zeros(2, np.int16), "FIXED_Y_C": np.array([0.5, 1.0]),
                                        "FIXED_SNR": np.array([5.0, 5.0]) / 1.01})
    monkeypatch.setattr(cli, "_setup_jax", lambda: None)
    assert cli.main(["cluster-randoms", "act-selfn", "--rms", str(tmp_path / "rms.fits"), "--area",
                     str(tmp_path / "area.fits"), "--flag", str(tmp_path / "flag.fits"), "--nside", "256",
                     "--catalog", str(tmp_path / "cat.fits"), "--out", str(tmp_path / "selfn.fits")]) == 0
    sel = get_selection({"name": "act_dr6", "selfn": str(tmp_path / "selfn.fits")})
    np.testing.assert_allclose(sel.sigma(np.array([9.5]), np.array([0.0])), [0.1 * 1.01], rtol=1e-5)
    # The sparse map as a QA template: NaN outside.
    t = HealpixColumn(str(tmp_path / "selfn.fits"), "RMS")
    out = t(np.array([9.5, 30.0]), np.array([0.0, 0.0]))
    assert out[0] == pytest.approx(0.101, rel=1e-5) and np.isnan(out[1])
