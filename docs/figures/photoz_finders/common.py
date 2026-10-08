"""Shared code of the figures of docs/photoz_finders.rst (photo-z cluster finding on DR11).

    python docs/figures/photoz_finders/make_all.py

The figures read the runs of ``scripts/photoz/photoz_strip.sh`` (and of the CC driver
``scripts/photoz/photoz_test.sh``, copied back) from ``$REMA_PHOTOZ`` (default
``~/data/legacysurvey/dr11/south/rema/notebooks/photoz``): ``<area>/<variant>/clusters.fits``,
with area ``strip`` or a region number (``0009``, ``0082``). The public cluster catalogues are the
reduced copies of the redMaPPer figures page (``$REMA_EXTERNAL``, default
``.../notebooks/redmapper/external``). A figure whose inputs are missing is skipped with a note.
The numbers quoted by the page are collected in ``values.json`` next to this file.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from rema.io.tables import read_catalog  # noqa: E402
from rema.model.cosmo import CosmoTable  # noqa: E402
from rema.sky.regions import Box  # noqa: E402
from rema.validate.compare import match_physical, null_threshold  # noqa: E402

DATA = Path.home() / "data" / "legacysurvey" / "dr11" / "south" / "rema"
RUNS = Path(os.environ.get("REMA_PHOTOZ", DATA / "notebooks" / "photoz"))
EXTERNAL = Path(os.environ.get("REMA_EXTERNAL", DATA / "notebooks" / "redmapper" / "external"))
FIGDIR = HERE
VALUES_FILE = HERE / "values.json"
COSMO = CosmoTable.create()

# Areas: own box and footprint (for the unmasked area of the own box).
AREAS = {
    "strip": dict(own=Box(1.0, 4.0, -14.0, -1.0), footprint=DATA / "strip" / "footprint_strip.fits",
                  label="strip (DECaLS)"),
    "0009": dict(own=Box(15.0, 30.0, -55.0, -45.0), footprint=RUNS / "0009" / "footprint.fits",
                 label="region 9 (DES)"),
    "0082": dict(own=Box(140.0, 150.0, 5.0, 15.0), footprint=RUNS / "0082" / "footprint.fits",
                 label="region 82 (DECaLS)"),
}

# The palette and style of the other documentation figures.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#a3a29d"
DPI = 120
plt.rcParams.update({
    "figure.dpi": 100, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.grid": True,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 9, "axes.titlesize": 9, "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6})

# The finders: label, colour, redshift column, detection statistics (the first is the default),
# and the null run that calibrates their thresholds.
FINDERS = {
    "rs_wcen": dict(label="red sequence", color=INK2, z="Z_LAMBDA", stats=("LAMBDA", "SNR"), null="null_rs"),
    "pz_v0": dict(label="photo-z filter", color=BLUE, z="Z_LAMBDA", stats=("SNR", "LAMBDA"), null="null_pz"),
    "pz_v1": dict(label="photo-z filter, z < 22", color=AQUA, z="Z_LAMBDA", stats=("SNR", "LAMBDA"),
                  null="null_pz1"),
    "pscd_0": dict(label="PSCD", color=ORANGE, z="Z", stats=("SNR", "SNR_NOCL", "LAMBDA_STAR"), null="null_pscd"),
    "pscd_1": dict(label="PSCD, z < 22", color=MAGENTA, z="Z", stats=("SNR", "SNR_NOCL", "LAMBDA_STAR"),
                   null="null_pscd1"),
}
RATES = (0.1, 0.3, 1.0)          # false detections per deg^2 of the matched thresholds
RATE = 0.3                       # the default


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png")


def record(key, value):
    """Store a number quoted by the documentation page in values.json."""
    vals = json.loads(VALUES_FILE.read_text()) if VALUES_FILE.exists() else {}
    vals[key] = value if isinstance(value, (str, list, dict)) else float(value)
    VALUES_FILE.write_text(json.dumps(vals, indent=1, sort_keys=True))


def need(*paths):
    """True when every input exists; otherwise print what is missing."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        print("skipped (missing inputs):", ", ".join(missing))
    return not missing


def logaxis(ax, which="x"):
    """Log scale without the crowded minor-tick labels."""
    from matplotlib.ticker import NullFormatter

    getattr(ax, f"set_{which}scale")("log")
    getattr(ax, f"{which}axis").set_minor_formatter(NullFormatter())


def panel_label(ax, text):
    ax.text(0.02, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK)


def run_path(area, variant):
    return RUNS / area / variant / "clusters.fits"


_CACHE = {}


def load(area, variant, members=False):
    """(clusters, members or None) of a run, cut to the own box of the area."""
    key = (area, variant, members)
    if key not in _CACHE:
        cat, mem, _ = read_catalog(run_path(area, variant), members=members)
        if cat and "SNR" not in cat and "LNLAMLIKE" in cat:      # catalogues of rema < 0.4
            from rema.modes.common import snr
            cat["SNR"] = snr(cat["LNLAMLIKE"])
        own = AREAS[area]["own"]
        if cat:
            keep = own.contains(cat["RA"], cat["DEC"])
            cat = {k: v[keep] for k, v in cat.items()}
            if members and mem:
                mem = {k: v[np.isin(mem["MEM_MATCH_ID"], cat["MEM_MATCH_ID"])] for k, v in mem.items()}
        _CACHE[key] = (cat, mem if members else None)
    return _CACHE[key]


def own_area(area):
    """Unmasked area [deg^2] of the area's own box (FRACGOOD of its footprint)."""
    from rema.sky.healpix import pix_area_deg2
    from rema.sky.maps import Footprint

    fp = Footprint.read(AREAS[area]["footprint"])
    inside = fp.pixels_in(AREAS[area]["own"])
    return float(np.sum(fp.fine.values["FRACGOOD"][inside]) * pix_area_deg2(fp.nside))


def available(area, variants=None):
    return [v for v in (variants or FINDERS) if run_path(area, v).exists()]


def threshold(area, variant, stat=None, rate=RATE):
    """(threshold, purity) of a finder's statistic at ``rate`` false detections per deg^2."""
    f = FINDERS[variant]
    stat = stat or f["stats"][0]
    cat, _ = load(area, variant)
    null, _ = load(area, f["null"])
    if not run_path(area, f["null"]).exists() or not cat:
        return np.nan, np.nan
    return null_threshold(cat[stat], null.get(stat, np.zeros(0)), own_area(area), rate)


def selected(area, variant, stat=None, rate=RATE):
    """The finder's clusters above its matched threshold."""
    cat, _ = load(area, variant)
    stat = stat or FINDERS[variant]["stats"][0]
    t, _ = threshold(area, variant, stat, rate)
    keep = np.asarray(cat[stat]) > t
    return {k: v[keep] for k, v in cat.items()}


def external(name):
    """A public cluster catalogue (reduced copy), or None (with a note) if absent."""
    d = EXTERNAL / name
    if not d.exists():
        print(f"{name}: not available, skipped")
        return None
    return {p.stem: np.load(p) for p in d.glob("*.npy")}


def matched(cat, zcol, ext, rmax=1.0, dz_max=0.05):
    """Indices (i_ext, i_cat) of one-to-one matches of external clusters to a finder's clusters."""
    return match_physical(ext["RA"], ext["DEC"], ext["Z"], cat["RA"], cat["DEC"], cat[zcol],
                          COSMO.mpc_per_deg, rmax, dz_max)[:2]
