"""Stellar-population photometry (rema.model.sps) and the builder of the model tables
(rema.data.build)."""

import urllib.error

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits

from rema.data import build
from rema.model.cosmo import _float64
from rema.model.sps import (AB_FNU, C_ANGSTROM, Bandpass, SSPGrid, ab_mag_10pc, csp_fnu,
                            exponential_weights, lookback_time, passive_mags)


@pytest.fixture(autouse=True)
def x64():
    with _float64():
        yield


def _boxcar(lam0, lam1, n=400):
    lam = np.linspace(lam0, lam1, n)
    return Bandpass.from_wavelength(lam, np.ones(n), "angstrom")


def test_exponential_weights_match_quadrature(rng):
    age = np.concatenate([[0.0], np.sort(rng.uniform(0.0, 3.0, 40)), [3.5]])
    F = rng.uniform(0.5, 2.0, age.size)
    for T in (0.05, 0.731, 2.0, 3.5):
        for tau in (0.03, 0.1, 1.0):
            t = np.linspace(0.0, T, 400001)
            exact = np.trapezoid(np.exp(-(T - t) / tau) * np.interp(t, age, F), t)
            w = np.asarray(exponential_weights(T, jnp.asarray(age), tau))
            assert np.dot(w, F) == pytest.approx(exact, rel=1e-6)
            # nodes beyond the first one at or above T do not contribute
            assert np.all(w[age > age[np.searchsorted(age, T)]] == 0)


def test_csp_of_a_constant_ssp():
    nu = jnp.linspace(1e14, 2e15, 50)
    age = jnp.linspace(0.0, 13.0, 131)
    ssp = SSPGrid(nu=nu, age=age, fnu=jnp.ones((nu.size, age.size)))
    T = jnp.array([0.01, 0.5, 10.0])
    out = np.asarray(csp_fnu(ssp, T, 0.1))
    np.testing.assert_allclose(out, np.broadcast_to(0.1 * (1 - np.exp(-np.asarray(T) / 0.1)), out.shape),
                               rtol=1e-12)


def test_ab_magnitude_of_a_flat_spectrum():
    nu = jnp.asarray(C_ANGSTROM / np.geomspace(20000.0, 1000.0, 3000))
    fnu = jnp.full(nu.size, AB_FNU)                      # 0 mag AB at every frequency
    band = _boxcar(6000.0, 7000.0)
    assert float(ab_mag_10pc(nu, fnu, 0.0, band)) == pytest.approx(0.0, abs=1e-10)
    # Observed from redshift z, F_nu is unchanged and the bandwidth term gives -2.5 log10(1 + z).
    assert float(ab_mag_10pc(nu, fnu, 0.5, band)) == pytest.approx(-2.5 * np.log10(1.5), abs=1e-10)
    # Twice the flux: 0.753 mag brighter.
    assert float(ab_mag_10pc(nu, 2 * fnu, 0.0, band)) == pytest.approx(-2.5 * np.log10(2.0), abs=1e-10)


def test_bandpass_units_and_readers(tmp_path):
    a = Bandpass.from_wavelength([400.0, 500.0, 600.0], [0.0, 1.0, 0.0], "nm")
    b = Bandpass.from_wavelength([4000.0, 5000.0, 6000.0], [0.0, 1.0, 0.0], "angstrom")
    np.testing.assert_allclose(np.asarray(a.nu), np.asarray(b.nu))
    assert np.all(np.diff(np.asarray(a.nu)) > 0) and float(a.response[1]) == 1.0
    (tmp_path / "f.ecsv").write_text(
        "# %ECSV 0.9\n# ---\n# datatype:\n# - {name: wavelength, unit: micron, datatype: float64}\n"
        "# - {name: response, datatype: float64}\nwavelength response\n0.4 0.0\n0.5 1.0\n0.6 0.0\n")
    np.testing.assert_allclose(np.asarray(Bandpass.from_ecsv(tmp_path / "f.ecsv").nu), np.asarray(b.nu))
    (tmp_path / "f.dat").write_text("4000 0\n5000 1\n6000 0\n")
    np.testing.assert_allclose(np.asarray(Bandpass.from_ascii(tmp_path / "f.dat").nu), np.asarray(b.nu))


def test_ssp_from_ezgal_file(tmp_path):
    nu, age = np.array([3e15, 1e15, 2e15]), np.array([0.0, 1e8, 1e10])
    sed = np.arange(9.0).reshape(3, 3)
    hdus = [fits.PrimaryHDU(sed), fits.BinTableHDU.from_columns([fits.Column("vs", "D", array=nu)]),
            fits.BinTableHDU.from_columns([fits.Column("ages", "D", array=age),
                                           fits.Column("masses", "D", array=np.ones(3))])]
    fits.HDUList(hdus).writeto(tmp_path / "m.model")
    ssp = SSPGrid.from_ezgal(tmp_path / "m.model")
    np.testing.assert_allclose(np.asarray(ssp.nu), [1e15, 2e15, 3e15])
    np.testing.assert_allclose(np.asarray(ssp.age), [0.0, 0.1, 10.0])
    np.testing.assert_allclose(np.asarray(ssp.fnu), sed[[1, 2, 0]])


def test_lookback_time_matches_astropy():
    from astropy.cosmology import FlatLambdaCDM
    from ggah_mod.cosmology import Cosmology

    z = np.array([0.1, 0.5, 1.0, 3.0])
    ours = np.asarray(lookback_time(jnp.asarray(z), Cosmology.create(Omega_m=0.3, h=0.7)))
    ref = FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=0.0).lookback_time(z).value
    np.testing.assert_allclose(ours, ref, rtol=2e-3)       # ggah_mod adds radiation and neutrinos


def test_passive_mags_normalisation_and_colours():
    from ggah_mod.cosmology import Cosmology

    nu = jnp.asarray(C_ANGSTROM / np.geomspace(60000.0, 900.0, 4000))
    age = jnp.linspace(0.0, 14.0, 141)
    # F_nu ~ nu^-2 fading with age: red, passive.
    fnu = (nu[:, None] / 1e15) ** -2.0 * jnp.exp(-age[None, :] / 5.0) * AB_FNU
    ssp = SSPGrid(nu=nu, age=age, fnu=fnu)
    bands = [_boxcar(4000.0, 5000.0), _boxcar(6000.0, 7000.0)]
    cosmo = Cosmology.create(Omega_m=0.3, h=0.7)
    z = jnp.array([0.1, 0.2, 0.6])
    raw = np.asarray(passive_mags(ssp, bands, z, 3.0, 0.1, cosmo))
    m = np.asarray(passive_mags(ssp, bands, z, 3.0, 0.1, cosmo, norm=(bands[1], 0.2, 17.0)))
    assert m[1, 1] == pytest.approx(17.0, abs=1e-9)
    np.testing.assert_allclose(m[:, 0] - m[:, 1], raw[:, 0] - raw[:, 1], atol=1e-12)
    assert np.all(m[:, 0] > m[:, 1])                       # red: fainter in the bluer band
    assert np.all(np.diff(m[:, 1]) > 0)                    # fainter with redshift


def test_mstar_des_z03_resampling(tmp_path, monkeypatch):
    z = np.round(np.arange(1, 121) * 0.01, 6)
    m = 15.0 + 5.0 * z
    fits.BinTableHDU.from_columns([fits.Column("Z", "D", array=z),
                                   fits.Column("MSTAR", "D", array=m)]).writeto(tmp_path / "src.fit")
    monkeypatch.setattr(build, "fetch", lambda name, cache=None: tmp_path / "src.fit")
    cols, header = build.build_mstar_des_z03()
    assert cols["Z"].size == 151 and cols["Z"][0] == pytest.approx(0.01) and cols["Z"][-1] == pytest.approx(1.51)
    np.testing.assert_allclose(cols["MSTAR"], 15.0 + 5.0 * cols["Z"], atol=1e-10)   # linear, extrapolated
    assert header["ZEXTRAP"][0] == pytest.approx(1.2)


def test_fetch_uses_the_cache_and_checks_the_hash(tmp_path, monkeypatch):
    (tmp_path / "x.txt").write_bytes(b"rema")
    good = build.Input("http://invalid.example/x.txt", build.sha256(tmp_path / "x.txt"))
    monkeypatch.setitem(build.INPUTS, "x.txt", good)
    assert build.fetch("x.txt", tmp_path) == tmp_path / "x.txt"
    monkeypatch.setitem(build.INPUTS, "x.txt", build.Input(good.url, "0" * 64))
    with pytest.raises(ValueError, match="SHA-256"):
        build.fetch("x.txt", tmp_path)
    monkeypatch.setenv("REMA_CACHE_DIR", str(tmp_path / "c"))
    assert build.default_cache() == tmp_path / "c" / "inputs"


@pytest.mark.network
def test_packaged_tables_are_reproduced():
    """The tables in rema/data are what rema.data.build makes from the public inputs (to 1e-8 mag)."""
    from importlib import resources

    try:
        tables = build.build_all()
    except urllib.error.URLError as e:                     # offline
        pytest.skip(f"inputs cannot be downloaded: {e}")
    diffs = build.compare(tables, str(resources.files("rema.data")))
    assert max(diffs.values()) < 1e-8, diffs


def test_fetch_downloads_into_the_cache(tmp_path, monkeypatch):
    src = tmp_path / "src.txt"
    src.write_bytes(b"inputs")
    monkeypatch.setitem(build.INPUTS, "src.txt", build.Input(src.as_uri(), build.sha256(src)))
    got = build.fetch("src.txt", tmp_path / "cache")
    assert got == tmp_path / "cache" / "src.txt" and got.read_bytes() == b"inputs"
    assert not list((tmp_path / "cache").glob("*.part"))


def test_build_main_writes_and_checks(tmp_path, monkeypatch, capsys):
    from rema.io.tables import read_table

    tables = {"mstar_x.fits": ({"Z": np.array([0.1, 0.2]), "MSTAR": np.array([17.0, 18.0])}, {"BUILDER": "t"})}
    monkeypatch.setattr(build, "build_all", lambda cache=None: tables)
    assert build.main(["--out", str(tmp_path)]) == 0
    np.testing.assert_array_equal(read_table(tmp_path / "mstar_x.fits")["MSTAR"], [17.0, 18.0])
    monkeypatch.setattr(build.resources, "files", lambda pkg: tmp_path)
    assert build.main(["--check"]) == 0                     # identical
    tables["mstar_x.fits"][0]["MSTAR"] = np.array([17.0, 18.5])
    assert build.main(["--check"]) == 1                     # differs
    assert "max |difference| 0.5" in capsys.readouterr().out
