"""Shared code of the figures of docs/redmapper_dr11.rst (redMaPPer blind mode on DR11).

    python docs/figures/redmapper_dr11/make_all.py          # prepare.py, then every fig_*.py

Each ``fig_*.py`` script also runs on its own and writes its PNG figures next to this file.
``docs/notebooks/build_notebooks.py redmapper`` assembles this file, ``prepare.py`` and the
``fig_*.py`` scripts, cell by cell (``# %%`` markers), into the notebook
``docs/notebooks/redmapper_dr11.ipynb``, where the figures are shown inline. Cells marked
``# %% [script-only]`` are left out of the notebook.

Paths come from environment variables, with the defaults of the DR11 notebooks:

REMA_PRODUCTS  the DR11 south production runs (``rema_dr11_v0.2.0_ra0-240`` and ``_ra240-360``)
REMA_WORK      where ``prepare.py`` writes the reduced tables and maps (default
               ``$REMA_PRODUCTS/notebooks/redmapper``)
REMA_RANDOMS   one DR11 randoms file (default ``randoms/randoms-south-1-0.fits``)
REMA_EXTERNAL  public cluster catalogues (default ``~/data/cluster_catalogues``); a figure panel
               whose catalogue is missing is skipped with a note
"""

# %% [markdown]
# # redMaPPer blind mode on DR11
#
# The figures of the documentation page *redMaPPer blind mode on DR11*, made from the DR11
# south production catalogue of rema (blind mode). Each section reproduces figures of the
# redMaPPer papers (Rykoff et al. 2014, 2016; Kluge et al. 2024) and of later papers that
# validate redMaPPer catalogues; the captions on the documentation page give the references.
#
# The first cells set the paths, the figure style and the shared helpers. `prepare.py` then
# reduces the catalogues (2.5 million clusters, 61 million members per part) and the DR11
# randoms to the small tables and HEALPix maps that the figures read; it is skipped when its
# products are up to date.

# %%
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

os.environ.setdefault("JAX_PLATFORMS", "cpu")        # only the red-sequence model uses JAX here

# Paths: the environment variables, else the DR11 data system at CC-IN2P3 when /sps is mounted,
# else a local copy with the same layout.
CC = Path("/sps/lsst/datasets/desi/legacysurveys")
LS_DIR = Path(os.environ.get("LEGACYSURVEY_DIR", CC if CC.exists() else Path.home() / "data" / "legacysurvey"))
DR11 = Path(os.environ.get("REMA_DR11_DIR", LS_DIR / "dr11" / "south"))
PRODUCTS = Path(os.environ.get("REMA_PRODUCTS", DR11 / "rema"))
# The two parts of the production run: they meet at RA 0 and 240 deg and at Dec -85 deg.
RUNS = [PRODUCTS / "rema_dr11_v0.2.0_ra0-240", PRODUCTS / "rema_dr11_v0.2.0_ra240-360"]
CALIB = RUNS[0] / "calib" / "calib.fits"
WORK = Path(os.environ.get("REMA_WORK", PRODUCTS / "notebooks" / "redmapper"))
RANDOMS = Path(os.environ.get("REMA_RANDOMS", DR11 / "randoms" / "randoms-south-1-0.fits"))
EXTERNAL = Path(os.environ.get("REMA_EXTERNAL", Path.home() / "data" / "cluster_catalogues"))
# Figures are written next to the scripts when they run as scripts; the notebook shows them.
try:
    FIGDIR = Path(__file__).resolve().parent
except NameError:
    FIGDIR = None
WORK.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# Figure style: the palette of the other documentation figures (categorical colours in a fixed
# order, a one-hue blue ramp for magnitudes, blue/red around a grey midpoint for contrasts),
# hairline grids and no top or right spines.

# %%
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#a3a29d"
RAMP = LinearSegmentedColormap.from_list("ramp", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGE = LinearSegmentedColormap.from_list("diverge", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b02f2f"])
RAMP.set_bad(SURFACE)
DIVERGE.set_bad(SURFACE)
DPI = 120

plt.rcParams.update({
    "figure.dpi": 100, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.grid": True,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 9, "axes.titlesize": 9, "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6})


def png_bytes(fig, dpi):
    """The figure as a PNG with a 256-colour palette (2-3 times smaller, no visible change)."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    out = io.BytesIO()
    img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(out, "png", optimize=True)
    return out.getvalue()


def save(fig, name):
    """Write ``<name>.png`` next to the scripts (script run) or show the figure (notebook)."""
    if FIGDIR is not None:
        (FIGDIR / f"{name}.png").write_bytes(png_bytes(fig, DPI))
        print(f"wrote {name}.png")
    else:
        from IPython.display import Image, display

        display(Image(data=png_bytes(fig, 90)))
    plt.close(fig)


def plain_log_ticks(ax, axis="x", subs=(1.0, 2.0, 5.0)):
    """Plain numbers (1, 2, 5, 10, ...) on a log axis instead of powers of ten."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(subs=subs))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def panel_label(ax, text):
    ax.text(0.02, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK)


# %% [markdown]
# Numbers that the documentation page quotes next to the literature values are collected with
# `record()` into `values.json` in the work directory (`make_all.py` turns them into a table).

# %%
VALUES_FILE = WORK / "values.json"


def record(key, value):
    """Store a number (or a short list) quoted by the documentation page."""
    vals = json.loads(VALUES_FILE.read_text()) if VALUES_FILE.exists() else {}
    vals[key] = value if isinstance(value, (str, list, dict)) else float(value)
    VALUES_FILE.write_text(json.dumps(vals, indent=1, sort_keys=True))


# %% [markdown]
# Literature values used as references in the figures (read from the papers' tables and text;
# values marked "read off" were read from their figures).

# %%
LIT = {
    # Rykoff et al. (2014), SDSS DR8: lambda/S > 20, 0.08 < z < 0.55.
    "R14_N": 25236, "R14_area": 10400.0,
    "R14_sigz": 0.01, "R14_out4": 0.01,
    # comoving density at z < 0.35 for lambda/S > 10, 20, 40 [h^3 Mpc^-3] (Fig. 18, read off)
    "R14_nV": {10: 4.5e-5, 20: 1.15e-5, 40: 2.5e-6},
    # Rykoff et al. (2016): SDSS DR8 v6.3 and DES SVA1.
    "R16_dr8_N": 26111, "R16_dr8_area": 10134.0, "R16_sva_N": 787, "R16_sva_area": 116.0,
    "R16_pcen": 0.82, "R16_rho0": 0.78, "R16_rho0_err": 0.11, "R16_sigma1": 0.31,
    # Kluge et al. (2024), LS DR10 south grz, lambda_norm > 16, z < z_vlim.
    "K24_N_grz": 112609, "K24_area_grz": 19342.0, "K24_N_griz": 91790, "K24_area_griz": 15326.0,
    "K24_lam_ratio_desy1": 0.79, "K24_snorm_griz": 1.039,
    # bias and empirical uncertainty of z_lambda in z_spec bins (Table 5)
    "K24_zbins": [(0.0, 0.05), (0.05, 0.4), (0.4, 0.8), (0.8, 1.2)],
    "K24_bias": [0.0040, 0.0004, -0.0041, -0.0178],
    "K24_dz_hi": [0.0093, 0.0062, 0.0095, 0.0153], "K24_dz_lo": [-0.0078, -0.0061, -0.0110, -0.0242],
    # log10 lambda_norm = a log10 sigma_v + b (Fig. 18)
    "K24_lam_sig": (2.401, -5.074),
    # DES Y1 redMaPPer (McClintock et al. 2019): 6729 clusters lambda > 20 over 1437 deg2
    "DESY1_N": 6729, "DESY1_area": 1437.0,
}


def ider_chitham_lambda(lam_sdss, z):
    """Ider Chitham et al. (2020), Eq. 1: Legacy Surveys lambda from SDSS DR8 lambda."""
    return lam_sdss / (1.04 + 0.17 * np.exp(5.4 * (z - 0.36)))


# %% [markdown]
# Shared helpers: cosmology (that of the calibration), statistics and matching.

# %%
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.cosmology import FlatLambdaCDM  # noqa: E402
import astropy.units as u  # noqa: E402

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)        # cosmology section of the calibration
H = 0.7
C_KMS = 299792.458
CALIB_BOXES = [(160.0, 180.0, -10.0, 10.0), (190.0, 210.0, -10.0, 10.0)]   # spec-z training area


def nmad(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size else np.nan


def binned(x, y, edges, min_n=10):
    """Per bin of x: centre, count, median, 16th/84th percentiles, mean, NMAD of y."""
    out = {k: [] for k in ("x", "n", "med", "p16", "p84", "mean", "nmad", "std")}
    idx = np.digitize(x, edges) - 1
    for k in range(len(edges) - 1):
        v = y[idx == k]
        out["x"].append(0.5 * (edges[k] + edges[k + 1]))
        out["n"].append(v.size)
        ok = v.size >= min_n
        out["med"].append(np.median(v) if ok else np.nan)
        out["p16"].append(np.percentile(v, 16) if ok else np.nan)
        out["p84"].append(np.percentile(v, 84) if ok else np.nan)
        out["mean"].append(v.mean() if ok else np.nan)
        out["nmad"].append(nmad(v) if ok else np.nan)
        out["std"].append(v.std() if ok else np.nan)
    return {k: np.asarray(v) for k, v in out.items()}


def nearest(ra1, dec1, ra2, dec2):
    """For each position 1, the index of the nearest position 2 and the separation [arcmin]."""
    c1 = SkyCoord(np.asarray(ra1) * u.deg, np.asarray(dec1) * u.deg)
    c2 = SkyCoord(np.asarray(ra2) * u.deg, np.asarray(dec2) * u.deg)
    j, sep, _ = c1.match_to_catalog_sky(c2)
    return j, sep.arcmin


def match_within(ra1, dec1, z1, ra2, dec2, z2, radius_arcmin, dz_max=None, rank2=None):
    """One-to-one matches: each object 1 takes the best-ranked object 2 within ``radius_arcmin``
    (and |dz|/(1+z1) < dz_max); ``rank2`` (higher is better, e.g. lambda) defaults to closeness.
    Returns index arrays (i1, i2)."""
    c1 = SkyCoord(np.asarray(ra1) * u.deg, np.asarray(dec1) * u.deg)
    c2 = SkyCoord(np.asarray(ra2) * u.deg, np.asarray(dec2) * u.deg)
    i1, i2, sep, _ = c2.search_around_sky(c1, radius_arcmin * u.arcmin)
    ok = np.ones(i1.size, bool)
    if dz_max is not None:
        ok = np.abs(np.asarray(z2)[i2] - np.asarray(z1)[i1]) / (1 + np.asarray(z1)[i1]) < dz_max
    i1, i2, sep = i1[ok], i2[ok], sep.arcmin[ok]
    score = -sep if rank2 is None else np.asarray(rank2)[i2] - 1e-6 * sep
    order = np.lexsort((-score, i1))
    i1, i2 = i1[order], i2[order]
    used1, used2, a, b = set(), set(), [], []
    for p, q in zip(i1, i2):
        if p in used1 or q in used2:
            continue
        used1.add(p)
        used2.add(q)
        a.append(p)
        b.append(q)
    return np.asarray(a, int), np.asarray(b, int)


def match_physical(ra1, dec1, z1, ra2, dec2, z2, rmax=1.0, dz_max=0.05, rank2=None):
    """One-to-one matches within a projected distance ``rmax`` [h^-1 Mpc] at z1 and
    |z2 - z1|/(1+z1) < dz_max; each object 1 takes the best-ranked free object 2 (``rank2``, higher
    is better; default the closest). Returns (i1, i2, projected distance [h^-1 Mpc])."""
    z1, z2 = np.asarray(z1, np.float64), np.asarray(z2, np.float64)
    ra2, dec2 = np.asarray(ra2), np.asarray(dec2)
    idx1 = np.flatnonzero(np.isfinite(z1) & (z1 > 0.01))
    scale = np.radians(1.0 / 60.0) * COSMO.angular_diameter_distance(z1[idx1]).value * H   # h^-1 Mpc per arcmin
    A, B, D = [], [], []
    # Search in slices of z1, each with its own angular radius and only the candidates in z.
    edges = np.unique(np.quantile(z1[idx1], np.linspace(0, 1, 21))) if idx1.size else np.array([])
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = np.flatnonzero((z1[idx1] >= lo) & (z1[idx1] <= hi))
        cand = np.flatnonzero(np.abs(z2 - 0.5 * (lo + hi)) < 0.5 * (hi - lo) + dz_max * (1 + hi))
        if s.size == 0 or cand.size == 0:
            continue
        c1 = SkyCoord(np.asarray(ra1)[idx1[s]] * u.deg, np.asarray(dec1)[idx1[s]] * u.deg)
        c2 = SkyCoord(ra2[cand] * u.deg, dec2[cand] * u.deg)
        a, b, sep, _ = c2.search_around_sky(c1, rmax / scale[s].min() * u.arcmin)
        d = sep.arcmin * scale[s][a]
        a, b = s[a], cand[b]
        ok = (d < rmax) & (np.abs(z2[b] - z1[idx1][a]) / (1 + z1[idx1][a]) < dz_max)
        A.append(a[ok])
        B.append(b[ok])
        D.append(d[ok])
    if not A:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    a, b, d = np.concatenate(A), np.concatenate(B), np.concatenate(D)
    score = -d if rank2 is None else np.asarray(rank2)[b] - 1e-6 * d
    order = np.lexsort((-score, a))
    used1, used2, out = set(), set(), []
    for p in order:
        if a[p] in used1 or b[p] in used2:
            continue
        used1.add(a[p])
        used2.add(b[p])
        out.append(p)
    out = np.asarray(out, int)
    return idx1[a[out]], b[out], d[out]


def in_calib(ra, dec):
    ra, dec = np.asarray(ra), np.asarray(dec)
    out = np.zeros(ra.size, bool)
    for r0, r1, d0, d1 in CALIB_BOXES:
        out |= (ra >= r0) & (ra < r1) & (dec >= d0) & (dec < d1)
    return out


# %% [markdown]
# Readers of the products of `prepare.py`: each table is a directory of `.npy` columns, memory
# mapped, so that the figures load only what they use.

# %%
def load(name, columns=None):
    """A reduced table of prepare.py (clusters, members, maps, or external/<catalogue>)."""
    d = WORK / name
    if not d.exists():
        raise FileNotFoundError(f"{name}: run prepare.py first")
    cols = columns or sorted(p.stem for p in d.glob("*.npy"))
    return {c: np.load(d / f"{c}.npy", mmap_mode="r") for c in cols}


def meta():
    return json.loads((WORK / "meta.json").read_text())


def external(name):
    """A public cluster catalogue normalised by prepare.py, or None (with a note) if absent."""
    d = WORK / "external" / name
    if not d.exists():
        print(f"{name}: not available, skipped")
        return None
    return {p.stem: np.load(p) for p in d.glob("*.npy")}


# %% [markdown]
# Sky maps: HEALPix (NEST) values drawn on a Mollweide projection centred on RA = 120 deg, RA
# increasing to the left.

# %%
import healpy as hp  # noqa: E402

RA_CENTRE = 120.0


def sky_axes(fig, rect=111):
    ax = fig.add_subplot(rect, projection="mollweide")
    ax.grid(True, color=GRID, lw=0.5)
    ticks = np.arange(-150, 181, 60)
    ax.set_xticks(np.radians(ticks))
    ax.set_xticklabels([f"{(RA_CENTRE - t) % 360:.0f}°" for t in ticks], fontsize=7, color=INK2)
    ax.tick_params(axis="y", labelsize=7, colors=INK2)
    return ax


def sky_image(ax, nside, pix, values, cmap=RAMP, vmin=None, vmax=None, nx=1440, ny=720):
    """Rasterise a partial NEST map (pixels ``pix`` with ``values``) on a Mollweide axis."""
    full = np.full(hp.nside2npix(nside), np.nan)
    full[np.asarray(pix)] = values
    lon = np.linspace(-np.pi, np.pi, nx)
    lat = np.linspace(-np.pi / 2, np.pi / 2, ny)
    LON, LAT = np.meshgrid(lon, lat)
    ra = (RA_CENTRE - np.degrees(LON)) % 360.0
    p = hp.ang2pix(nside, ra, np.degrees(LAT), lonlat=True, nest=True)
    img = np.ma.masked_invalid(full[p])
    return ax.pcolormesh(LON, LAT, img, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)


def sky_xy(ra, dec):
    """Mollweide axis coordinates [rad] of RA, Dec [deg]."""
    x = np.radians(RA_CENTRE - np.asarray(ra))
    x = (x + np.pi) % (2 * np.pi) - np.pi
    return x, np.radians(np.asarray(dec))
