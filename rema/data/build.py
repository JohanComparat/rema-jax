"""Build the model tables of :mod:`rema.data` from their public inputs.

    python -m rema.data.build                 # rewrite the tables in rema/data
    python -m rema.data.build --out DIR       # write them elsewhere
    python -m rema.data.build --check         # compare with the packaged tables, write nothing

The inputs (16 MB) are downloaded once from fixed commits, checked against their SHA-256 and
kept in ``--cache`` (default ``$REMA_CACHE_DIR/inputs``, else ``~/.cache/rema/inputs``):

- ``mstar_des_z03.fits``: redMaPPer's m*(z) in DES z (``redmapper/data/mstar/mstar_des_z03.fit``,
  0.01 <= z <= 1.2), resampled with linear interpolation on z = 0.01, 0.02, ..., 1.51 and
  extrapolated linearly above z = 1.2.
- ``mstar_legacy_z_ezgal.fits`` and ``colors_bc03_legacy_grizw1.fits``: a Bruzual & Charlot (2003)
  population (Salpeter IMF, Z = 0.02; EzGal's SSP grid) with exponential star formation,
  tau = 0.1 Gyr, formed at z_f = 3, normalised to SDSS i = 17.85 (AB) at z = 0.2, seen through
  the DECam 2014 griz and WISE W1 responses (speclite) at z = 0.01, ..., 1.50; flat cosmology with
  Omega_m = 0.3, h = 0.7 (ggah_mod). m*(z) is the DECam z magnitude; the colours are the
  differences of adjacent magnitudes g-r, r-i, i-z, z-W1. See :mod:`rema.model.sps`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

_EZGAL = "https://raw.githubusercontent.com/dpgettings/ezgal/de4a58879eaee0bddbcdb42dddfa2b398dbc3ea9/ezgal/data"
_SPECLITE = ("https://raw.githubusercontent.com/desihub/speclite/8159ea61f1c8539136660b4c2e87fbbf94a70097"
             "/speclite/data/filters")
_REDMAPPER = "https://raw.githubusercontent.com/erykoff/redmapper/601994a9296536c5d6bf4516f37e9b74ba1d538a/redmapper/data"


@dataclass(frozen=True)
class Input:
    url: str
    sha256: str


INPUTS = {
    "bc03_ssp_z_0.02_salp.model": Input(f"{_EZGAL}/models/bc03_ssp_z_0.02_salp.model",
                                        "5b2e2f9f2ecdbbd329ff4993f84a5d4bd7f8bfef6877b38d2042e9a0cf96e1fd"),
    "sloan_i": Input(f"{_EZGAL}/filters/sloan_i",
                     "982401b49f8f70fa9cd3f7989e90fedbecc1c636430e6c16421dc596aa5e6e8c"),
    "decam2014-g.ecsv": Input(f"{_SPECLITE}/decam2014-g.ecsv",
                              "de326ad1be640d8ce0dde16986279d6eefe22ebe62b1abefc20f7f01ceb65128"),
    "decam2014-r.ecsv": Input(f"{_SPECLITE}/decam2014-r.ecsv",
                              "19d13d3660aa60bafb0ba794a262bf19a5c46e143bb439edf3c1ba82c1383cd6"),
    "decam2014-i.ecsv": Input(f"{_SPECLITE}/decam2014-i.ecsv",
                              "be1028192bdf528f5571d090337795348ca45b8f0db050ef6d7eecb066b7da8e"),
    "decam2014-z.ecsv": Input(f"{_SPECLITE}/decam2014-z.ecsv",
                              "0db2b279dffe3e9c31397bda7a33f59e7b20c18b5986cf94534aa31bd8d09a7b"),
    "wise2010-W1.ecsv": Input(f"{_SPECLITE}/wise2010-W1.ecsv",
                              "73d3c1595c3fb4bcff1c82f0f5f814b37754391924d8572e9328a9de54212a28"),
    "mstar_des_z03.fit": Input(f"{_REDMAPPER}/mstar/mstar_des_z03.fit",
                               "5b0772b96238d74868153ddb3cd0af19876f2b04f2ab1e28512e3794ca67fbe0"),
}

# The passive population of the EzGal tables.
TAU_GYR, ZF, IMF, METALLICITY = 0.1, 3.0, "salpeter", 0.02
NORM_BAND, NORM_Z, NORM_MAG = "sloan_i", 0.2, 17.85
OMEGA_M, H = 0.3, 0.7
BANDS = ("g", "r", "i", "z", "w1")
BAND_FILES = ("decam2014-g.ecsv", "decam2014-r.ecsv", "decam2014-i.ecsv", "decam2014-z.ecsv",
              "wise2010-W1.ecsv")
Z_GRID = np.round(np.arange(1, 151) * 0.01, 10)       # 0.01, ..., 1.50


def default_cache() -> Path:
    root = os.environ.get("REMA_CACHE_DIR")
    return (Path(root) if root else Path.home() / ".cache" / "rema") / "inputs"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, cache: str | Path | None = None) -> Path:
    """Path of input ``name`` in the cache, downloaded first if needed; checks its SHA-256."""
    item = INPUTS[name]
    path = Path(cache or default_cache()) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        req = urllib.request.Request(item.url, headers={"User-Agent": "rema"})
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.replace(path)
    got = sha256(path)
    if got != item.sha256:
        raise ValueError(f"{path}: SHA-256 {got} differs from the expected {item.sha256}; "
                         "delete it to download it again")
    return path


def build_mstar_des_z03(cache=None) -> tuple[dict, dict]:
    """redMaPPer's m*(z) in DES z on z = z0, 2 z0, ..., 151 z0 (linear, extrapolated); z0 = 0.01."""
    from astropy.io import fits

    with fits.open(fetch("mstar_des_z03.fit", cache)) as h:
        z = np.asarray(h[1].data["Z"], np.float64)
        m = np.asarray(h[1].data["MSTAR"], np.float64)
    o = np.argsort(z)
    z, m = z[o], m[o]
    zn = z[0] + z[0] * np.arange(151)                  # z0, 2 z0, ..., 151 z0 (1.51)
    mn = np.interp(zn, z, m)
    lo, hi = zn < z[0], zn > z[-1]
    mn[lo] = m[0] + (m[1] - m[0]) / (z[1] - z[0]) * (zn[lo] - z[0])
    mn[hi] = m[-1] + (m[-1] - m[-2]) / (z[-1] - z[-2]) * (zn[hi] - z[-1])
    header = {"BUILDER": "rema.data.build", "SOURCE": INPUTS["mstar_des_z03.fit"].url,
              "ZEXTRAP": (float(z[-1]), "linear extrapolation above this redshift")}
    return {"Z": zn, "MSTAR": mn}, header


def build_ezgal_tables(cache=None) -> tuple[tuple[dict, dict], tuple[dict, dict]]:
    """(m*(z) in DECam z, adjacent colours g-r, r-i, i-z, z-W1) of the passive population."""
    import jax.numpy as jnp
    from ggah_mod.cosmology import Cosmology

    from ..model.cosmo import _float64
    from ..model.sps import Bandpass, SSPGrid, passive_mags

    with _float64():
        ssp = SSPGrid.from_ezgal(fetch("bc03_ssp_z_0.02_salp.model", cache))
        bands = [Bandpass.from_ecsv(fetch(f, cache), b) for f, b in zip(BAND_FILES, BANDS)]
        inorm = Bandpass.from_ascii(fetch(NORM_BAND, cache), "angstrom", NORM_BAND)
        cosmo = Cosmology.create(Omega_m=OMEGA_M, h=H)
        mags = np.asarray(passive_mags(ssp, bands, jnp.asarray(Z_GRID), ZF, TAU_GYR, cosmo,
                                       norm=(inorm, NORM_Z, NORM_MAG)), np.float64)
    if not np.all(np.isfinite(mags)):
        raise ValueError("non-finite model magnitudes")
    common = {"BUILDER": "rema.data.build", "MODEL": "BC03 exponential SFH (EzGal SSP grid)",
              "TAU": (TAU_GYR, "e-folding time [Gyr]"), "ZF": (ZF, "formation redshift"),
              "IMF": IMF, "METAL": (METALLICITY, "metallicity Z"),
              "NORMBAND": NORM_BAND, "NORMZ": NORM_Z, "NORMMAG": (NORM_MAG, "AB"),
              "OMEGAM": OMEGA_M, "H": H, "SSP": INPUTS["bc03_ssp_z_0.02_salp.model"].url,
              "FILTERS": _SPECLITE}
    mstar = ({"Z": Z_GRID.copy(), "MSTAR": mags[:, BANDS.index("z")]},
             {**common, "BAND": "decam z"})
    colors = ({"Z": Z_GRID.copy(), "COLOR": mags[:, :-1] - mags[:, 1:]},
              {**common, "BANDS": ",".join(BANDS)})
    return mstar, colors


def build_all(cache=None) -> dict[str, tuple[dict, dict]]:
    """{file name: (columns, header)} of every packaged model table."""
    mstar, colors = build_ezgal_tables(cache)
    return {"mstar_des_z03.fits": build_mstar_des_z03(cache),
            "mstar_legacy_z_ezgal.fits": mstar,
            "colors_bc03_legacy_grizw1.fits": colors}


def compare(tables: dict, ref_dir) -> dict[str, float]:
    """Largest absolute difference of each table from the one in ``ref_dir``."""
    from ..io.tables import read_table

    out = {}
    for name, (cols, _) in tables.items():
        ref = read_table(Path(ref_dir) / name)
        out[name] = max(float(np.max(np.abs(np.asarray(cols[k]) - np.asarray(ref[k])))) for k in cols)
    return out


def main(argv=None) -> int:
    from ..io.tables import write_table

    p = argparse.ArgumentParser(prog="python -m rema.data.build", description=__doc__.split("\n")[0])
    p.add_argument("--out", help="output directory (default: the rema.data package directory)")
    p.add_argument("--cache", help="download cache (default: $REMA_CACHE_DIR/inputs or ~/.cache/rema/inputs)")
    p.add_argument("--check", action="store_true", help="compare with the packaged tables, write nothing")
    args = p.parse_args(argv)
    tables = build_all(args.cache)
    packaged = Path(str(resources.files("rema.data")))
    if args.check:
        diffs = compare(tables, packaged)
        for name, d in diffs.items():
            print(f"{name}: max |difference| {d:.3g}")
        return 0 if max(diffs.values()) < 1e-8 else 1
    out = Path(args.out) if args.out else packaged
    out.mkdir(parents=True, exist_ok=True)
    for name, (cols, header) in tables.items():
        write_table(out / name, cols, header=header)
        print(out / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
