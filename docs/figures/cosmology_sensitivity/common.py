"""Shared code of the figures of docs/cosmology_sensitivity.rst.

    python docs/figures/cosmology_sensitivity/make_all.py

The figures read the results of the scripts in ``scripts/cosmo_sens`` from
``$REMA_WORK/cosmo_sens`` (default ``~/rema_work/cosmo_sens``, see
``scripts/cosmo_sens/cosmo_common.py``):
``tierA/*.fits`` (rema remeasure), ``response_*.fits``, ``fit_*.json`` and ``forecast_*.json``.
A figure whose inputs are missing is skipped with a note. The numbers quoted by the page are
collected in ``values.json`` next to this file.
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
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "cosmo_sens"))
import cosmo_common as CS  # noqa: E402  (scripts/cosmo_sens/cosmo_common.py: paths and binning)

RESULTS = CS.OUT
FIGDIR = HERE
VALUES_FILE = HERE / "values.json"

# The palette and style of the other documentation figures.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#a3a29d"
DPI = 120
plt.rcParams.update({
    "figure.dpi": 100, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.grid": True,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 9, "axes.titlesize": 9, "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6})


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


def panel_label(ax, text):
    ax.text(0.02, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK)
