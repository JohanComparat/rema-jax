"""DR11 sweep ingestion on synthetic sweep and photo-z files: selection, columns, photo-z
matching, per-sweep tables and their headers."""

import numpy as np
import pytest

from rema.config import RemaConfig
from rema.io import legacy as L
from rema.io.tables import read_header, read_table, write_table
from rema.sky.regions import Box, sweep_box

BANDS = "GRIZ"
SWEEP_NAME = "sweep-010m005-015p000.fits"
# Rows with a defect that must be rejected, and the one masked row that is kept (SGA galaxy).
REJECT = {0: "psf", 1: "masked", 3: "nobs", 4: "low snr", 5: "too faint", 6: "dup",
          7: "no transmission", 8: "nan position"}
KEPT_LARGEGAL = 2


def _sweep_columns(rng, n, box, mag_max):
    """Sweep rows (RELEASE ... NOBS_<band>) of clean galaxies, then one defect per REJECT row."""
    ra = rng.uniform(box.ra_min, box.ra_max, n)
    dec = rng.uniform(box.dec_min, box.dec_max, n)
    zmag = rng.uniform(17.0, mag_max - 0.5, n)
    cols = {"RELEASE": np.full(n, 10000, np.int16), "BRICKID": rng.integers(1, 400_000, n).astype(np.int32),
            "OBJID": np.arange(n, dtype=np.int32), "RA": ra, "DEC": dec,
            "TYPE": np.array(["REX", "EXP", "DEV", "SER"] * (n // 4) + ["REX"] * (n % 4), dtype="S3"),
            "EBV": rng.uniform(0.01, 0.08, n).astype(np.float32),
            "MASKBITS": np.zeros(n, np.int16), "FITBITS": np.zeros(n, np.int16)}
    for k, b in enumerate(BANDS):
        mw = np.full(n, 0.8 + 0.05 * k, np.float32)
        flux = 10 ** (-0.4 * (zmag + 0.3 * (3 - k) - 22.5)) * mw
        cols[f"FLUX_{b}"] = flux.astype(np.float32)
        cols[f"FLUX_IVAR_{b}"] = np.full(n, 400.0, np.float32)
        cols[f"MW_TRANSMISSION_{b}"] = mw
        cols[f"NOBS_{b}"] = np.full(n, 3, np.int16)
    cols["TYPE"][0] = b"PSF"
    cols["MASKBITS"][1] = 1 << 1
    cols["MASKBITS"][KEPT_LARGEGAL] = 1 << 1
    cols["FITBITS"][KEPT_LARGEGAL] = 1 << L.FITBITS_LARGEGALAXY
    cols["NOBS_G"][3] = 0
    cols["FLUX_Z"][4] = 2.0 / np.sqrt(400.0) * cols["MW_TRANSMISSION_Z"][4]   # S/N 2
    cols["FLUX_Z"][5] = 10 ** (-0.4 * (mag_max + 0.5 - 22.5)) * cols["MW_TRANSMISSION_Z"][5]
    cols["TYPE"][6] = b"DUP"
    cols["MW_TRANSMISSION_R"][7] = 0.0
    cols["RA"][8] = np.nan
    return cols


def _pz_columns(sweep, n):
    """Row-matched photo-z columns: ZSPEC on every fifth row, with an excluded and an unknown
    survey, and failed photo-z on every seventh row."""
    zspec = np.where(np.arange(n) % 5 == 0, 0.1 + 0.6 * np.arange(n) / n, -99.0)
    survey = np.array(["SDSS", "DESI", "BOSS", "GAMA", "AGES"] * (n // 5 + 1), dtype="S11")[:n]
    survey[10], survey[15] = b"COSMOS2015", b"NEWSURVEY"
    zphot = np.where(np.arange(n) % 7 == 0, -99.0, 0.4).astype(np.float32)
    return {"RELEASE": sweep["RELEASE"], "BRICKID": sweep["BRICKID"], "OBJID": sweep["OBJID"],
            "Z_SPEC": zspec, "SURVEY": survey, "Z_PHOT_MEDIAN_I": zphot,
            "Z_PHOT_STD_I": np.full(n, 0.05, np.float32)}


def fake_dr11(root, rng, names=(SWEEP_NAME,), n=60, cfg=None, pz=True):
    """``root/sweep/11.0/<name>`` and ``root/sweep/11.0-photo-z/<name>-pz.fits`` (DR11 layout);
    returns the sweep paths and their columns."""
    mag_max = L.default_mag_max(cfg or RemaConfig())
    paths, tables = [], []
    for name in names:
        cols = _sweep_columns(rng, n, sweep_box(name), mag_max)
        p = write_table(root / "sweep" / "11.0" / name, cols)
        if pz:
            write_table(L.pz_path(p), _pz_columns(cols, n))
        paths.append(p)
        tables.append(cols)
    return paths, tables


def test_ls_id_and_pz_path():
    assert int(L.ls_id(10000, 5, 7)[()]) == (10000 << 42) | (5 << 22) | 7
    p = L.pz_path("/d/sweep/11.0/sweep-000m005-005p000.fits")
    assert str(p) == "/d/sweep/11.0-photo-z/sweep-000m005-005p000-pz.fits"


def test_magnitude_limit_and_survey_hash():
    cfg = RemaConfig()
    m = L.default_mag_max(cfg)
    assert 20.0 < m < 24.0
    assert L.default_mag_max(cfg.replace(survey={"mag_max": 21.0})) == 21.0
    assert L.survey_hash(cfg) == L.survey_hash(RemaConfig())
    assert L.survey_hash(cfg) != L.survey_hash(cfg.replace(survey={"ebv_max": 0.1}))
    assert len(L.survey_hash(cfg)) == 12


def test_ingest_sweep_selection_and_columns(tmp_path):
    rng = np.random.default_rng(31)
    n = 60
    (path,), (src,) = fake_dr11(tmp_path, rng, n=n)
    t = L.ingest_sweep(path, RemaConfig())
    keep = np.setdiff1d(np.arange(n), list(REJECT))
    np.testing.assert_array_equal(t["ID"], L.ls_id(src["RELEASE"], src["BRICKID"], src["OBJID"])[keep])
    assert set(t) == {"ID", "RA", "DEC", "FLUX", "FLUX_IVAR", "REFMAG", "REFMAG_ERR", "EBV", "TYPE",
                      "MASKBITS", "ZSPEC", "ZSPEC_SRC", "ZPHOT", "ZPHOT_STD"}
    assert t["FLUX"].shape == (keep.size, 4) and t["FLUX"].dtype == np.float32
    # Dereddened fluxes and inverse variances; REFMAG and its error from the z band.
    mw = src["MW_TRANSMISSION_Z"][keep]
    np.testing.assert_allclose(t["FLUX"][:, 3], src["FLUX_Z"][keep] / mw, rtol=1e-6)
    np.testing.assert_allclose(t["FLUX_IVAR"][:, 3], 400.0 * mw**2, rtol=1e-6)
    np.testing.assert_allclose(t["REFMAG"], 22.5 - 2.5 * np.log10(t["FLUX"][:, 3]), atol=1e-4)
    snr = t["FLUX"][:, 3] * np.sqrt(t["FLUX_IVAR"][:, 3])
    np.testing.assert_allclose(t["REFMAG_ERR"], 2.5 / np.log(10) / snr, rtol=1e-5)
    # TYPE codes; the SGA galaxy keeps its MASKBITS.
    assert set(np.unique(t["TYPE"])) <= {1, 2, 3, 4}
    assert t["MASKBITS"][t["ID"] == L.ls_id(10000, src["BRICKID"][KEPT_LARGEGAL], KEPT_LARGEGAL)] == 2
    # Photo-z file: ZSPEC on every fifth row except the COSMOS2015 one; survey index or 255.
    zs = t["ZSPEC"]
    rows = keep
    has = (rows % 5 == 0) & (rows != 10)
    assert np.all(zs[has] > 0) and np.all(zs[~has] == -1)
    assert np.all(t["ZSPEC_SRC"][~has] == 0)
    assert t["ZSPEC_SRC"][rows == 15][0] == 255
    assert t["ZSPEC_SRC"][rows == 20][0] == L.SURVEYS.index("SDSS") + 1
    np.testing.assert_array_equal(t["ZPHOT"] > 0, rows % 7 != 0)
    np.testing.assert_array_equal(t["ZPHOT_STD"][rows % 7 == 0], -1.0)


def test_ingest_sweep_options(tmp_path):
    rng = np.random.default_rng(32)
    (path,), (src,) = fake_dr11(tmp_path, rng, n=80)
    cfg = RemaConfig()
    full = L.ingest_sweep(path, cfg)
    # A box keeps the galaxies inside it.
    box = Box(10.0, 12.5, -5.0, 0.0)
    part = L.ingest_sweep(path, cfg, box=box)
    assert 0 < part["ID"].size < full["ID"].size
    np.testing.assert_array_equal(part["ID"], full["ID"][box.contains(full["RA"], full["DEC"])])
    # Explicit cuts: E(B-V), PSF kept, SGA galaxies not kept.
    ebv = L.ingest_sweep(path, cfg.replace(survey={"ebv_max": 0.04}))
    assert ebv["ID"].size == np.sum(full["EBV"] < 0.04)
    psf = L.ingest_sweep(path, cfg.replace(survey={"reject_psf": False}))
    assert psf["ID"].size == full["ID"].size + 1 and 0 in psf["TYPE"]
    nosga = L.ingest_sweep(path, cfg.replace(survey={"keep_largegalaxy": False}))
    assert nosga["ID"].size == full["ID"].size - 1
    # A brighter limit removes galaxies; an empty selection still has every column.
    bright = L.ingest_sweep(path, cfg, mag_max=19.0)
    assert np.all(bright["REFMAG"] < 19.0) and bright["ID"].size < full["ID"].size
    empty = L.ingest_sweep(path, cfg, box=Box(100, 101, 0, 1))
    assert set(empty) == set(full) and empty["ID"].size == 0 and empty["FLUX"].shape == (0, 4)
    # Without a photo-z file: -1, or an error when required; pz=None skips it.
    nopz = L.ingest_sweep(path, cfg, pz=None)
    assert np.all(nopz["ZSPEC"] == -1) and np.all(nopz["ZPHOT"] == -1)
    L.pz_path(path).unlink()
    assert np.all(L.ingest_sweep(path, cfg)["ZSPEC"] == -1)
    with pytest.raises(FileNotFoundError, match="photo-z"):
        L.ingest_sweep(path, cfg, require_pz=True)


def test_ingest_sweep_rejects_unmatched_photoz(tmp_path):
    rng = np.random.default_rng(33)
    (path,), (src,) = fake_dr11(tmp_path, rng, n=40)
    pz = read_table(L.pz_path(path))
    pz["OBJID"] = pz["OBJID"][::-1].copy()
    write_table(L.pz_path(path), pz)
    with pytest.raises(ValueError, match="row-matched"):
        L.ingest_sweep(path, RemaConfig())


def test_ingest_several_sweeps_to_one_table(tmp_path):
    rng = np.random.default_rng(34)
    names = ("sweep-010m005-015p000.fits", "sweep-015m005-020p000.fits")
    paths, _ = fake_dr11(tmp_path, rng, names=names, n=50)
    cfg = RemaConfig()
    out = tmp_path / "gal.fits"
    tab = L.ingest(paths, out, cfg)
    assert tab["ID"].size == 2 * (50 - len(REJECT))
    back = read_table(out, hdu="GALAXIES")
    np.testing.assert_array_equal(back["ID"], tab["ID"])
    hdr = read_header(out, "GALAXIES")
    assert hdr["NSWEEPS"] == 2 and hdr["SURVHASH"] == L.survey_hash(cfg)
    assert hdr["BANDS"] == "g,r,i,z" and hdr["REFBAND"] == "z"
    # A box skips the sweeps it does not overlap.
    box = Box(16.0, 19.0, -4.0, -1.0)
    sub = L.ingest(paths, None, cfg, box=box)
    assert sub["ID"].size > 0 and np.all(box.contains(sub["RA"], sub["DEC"]))
    assert L.concat([{}, {}]) == {}


def test_ingest_sweeps_per_sweep_tables_and_counts(tmp_path, caplog):
    rng = np.random.default_rng(35)
    names = ("sweep-010m005-015p000.fits", "sweep-015m005-020p000.fits")
    paths, _ = fake_dr11(tmp_path, rng, names=names, n=50)
    cfg = RemaConfig()
    outdir = tmp_path / "galaxies"
    out = L.ingest_sweeps(paths, outdir, cfg)
    assert [p.name for p in out] == list(names)
    hdr = read_header(out[0], "GALAXIES")
    ngal = 50 - len(REJECT)
    assert hdr["SWEEP"] == names[0] and hdr["NGAL"] == ngal and hdr["SURVHASH"] == L.survey_hash(cfg)
    assert hdr["NZSPEC"] == int(np.sum(read_table(out[0])["ZSPEC"] > 0)) > 0
    # Same configuration: kept as is; another one: ingested again.
    mtime = out[0].stat().st_mtime_ns
    L.ingest_sweeps(paths[:1], outdir, cfg)
    assert out[0].stat().st_mtime_ns == mtime
    cfg2 = cfg.replace(survey={"ebv_max": 0.05})
    L.ingest_sweeps(paths[:1], outdir, cfg2)
    assert read_header(out[0], "GALAXIES")["SURVHASH"] == L.survey_hash(cfg2)
    # Mixed survey hashes are refused when the box spans both tables.
    with pytest.raises(ValueError, match="different survey configurations"):
        L.read_galaxies(outdir, Box(10, 20, -5, 0))
    with pytest.raises(FileNotFoundError, match="overlaps"):
        L.read_galaxies(outdir, Box(100, 101, 0, 1))
    # Header-only summary of the directory.
    c = L.galaxy_counts(outdir)
    assert list(c["NAME"]) == list(names)
    np.testing.assert_array_equal(c["RA0"], [10.0, 15.0])
    np.testing.assert_array_equal(c["DEC1"], [0.0, 0.0])
    assert c["NGAL"][1] == ngal and 0.01 < c["EBVMEAN"][1] < 0.08
    empty = L.galaxy_counts(tmp_path / "nothing")
    assert empty["NAME"].size == 0
    # Missing photo-z files are an error by default for per-sweep tables.
    L.pz_path(paths[1]).unlink()
    with pytest.raises(FileNotFoundError):
        L.ingest_sweeps(paths[1:], tmp_path / "other", cfg)
