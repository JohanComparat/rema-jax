"""Shared code of the figures of docs/cluster_statistics_dr11.rst.

    python docs/figures/cluster_stats/make_all.py

The figures read the measurements of ``sum_stat`` (``scripts/measure_cluster_sumstat.py``) from
``$REMA_WORK/cluster_sumstat`` (default ``~/rema_work/cluster_sumstat``):
``<sample>_wp.h5``, ``<sample>_esd.h5`` and ``xgal/<sample>_x_M<threshold>.h5``.
A figure whose inputs are missing is skipped with a note. The numbers quoted by the page are
collected in ``values.json`` next to this file.
"""

import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("REMA_WORK", Path.home() / "rema_work")) / "cluster_sumstat"
# The sample files of ``rema cluster-randoms sample`` (for the mean observable of each bin).
SAMPLES = Path(os.environ.get("REMA_DATA", Path.home() / "data/legacysurvey/dr11/south")) / "rema" / \
    "rema_dr11_v0.2.0_randoms"
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
    "font.size": 9, "axes.titlesize": 9, "legend.frameon": False, "legend.fontsize": 7.5,
    "lines.linewidth": 1.4, "errorbar.capsize": 0})

# Sample names on the page, and the bins of their specs (scripts/randoms/*.yaml).
LABEL = {"lambda_vlim": r"$\lambda$ (vlim)", "lambda_vlim_s": r"$\lambda$ (vlim, $\lambda/S\geq$ edge)",
         "erass1": "eRASS1", "act_dr6": "ACT DR6", "spt_2500d": "SPT-SZ 2500d", "psz2_noise": "PSZ2"}
Z_EDGES = {"lambda_vlim": [0.10, 0.20, 0.30, 0.45, 0.60, 0.75], "lambda_vlim_s": [0.10, 0.20, 0.30, 0.45, 0.60, 0.75],
           "erass1": [0.10, 0.30, 0.50, 0.80], "act_dr6": [0.10, 0.30, 0.50, 0.70],
           "spt_2500d": [0.25, 0.50, 0.75], "psz2_noise": [0.05, 0.25, 0.50]}
OBS_EDGES = {"lambda_vlim": [10, 20, 30, 45, 60, None], "lambda_vlim_s": [10, 20, 30, 45, 60, None],
             "erass1": [0, 12.59, 50.12, 199.5, None], "act_dr6": [0, 1, 2, None],
             "spt_2500d": [5, 7, None], "psz2_noise": [6, None]}
OBS_NAME = {"lambda_vlim": r"\lambda", "lambda_vlim_s": r"\lambda", "erass1": r"\log L_{500}",
            "act_dr6": r"\tilde y_0/10^{-4}", "spt_2500d": r"\xi", "psz2_noise": r"{\rm S/N}"}
EXTERNAL = ["erass1", "act_dr6", "spt_2500d", "psz2_noise"]


def obs_label(sample, ob):
    lo, hi = OBS_EDGES[sample][ob], OBS_EDGES[sample][ob + 1]
    # eRASS1 L500 is in 1e42 erg/s: the labels give log10 L500 [erg/s].
    fmt = (lambda v: f"{np.log10(v) + 42:.1f}") if sample == "erass1" else (lambda v: f"{v:g}")
    if sample == "erass1" and lo == 0:
        return f"${OBS_NAME[sample]}<{fmt(hi)}$"
    return f"${fmt(lo)}\\leq {OBS_NAME[sample]}<{fmt(hi)}$" if hi else f"${OBS_NAME[sample]}\\geq {fmt(lo)}$"


def z_label(sample, zb):
    e = Z_EDGES[sample]
    return f"${e[zb]:.2f}\\leq z<{e[zb + 1]:.2f}$"


def load(path, stat):
    """{(z_bin, obs_bin): group of ``stat``} of a measurement file, or {} if it is missing."""
    path = Path(path)
    if not path.exists():
        print(f"missing {path}: skipped")
        return {}
    f = h5py.File(path, "r")
    out = {}
    for s in f["clusters"]:
        for k, g in f["clusters"][s].items():
            if stat in g:
                zb, ob = (int(x[1:]) for x in k.split("_"))
                out[zb, ob] = g[stat]
    return out


def vector(g, x_unit="h-1Mpc", y_scale=None):
    """Bin centres in h^-1 Mpc, the value, its jackknife error and covariance, in h units."""
    h = float(g.attrs["h"])
    x = g["sep_centres"][:] * h
    s = h if y_scale is None else y_scale(h)
    v, c = g["value"][:] * s, g["cov"][:] * s ** 2
    return x, v, np.sqrt(np.diag(c)), c


def snr(v, c, n_sub):
    """S/N of a vector with its jackknife covariance (Hartlap factor); the diagonal one if singular."""
    p = len(v)
    hart = (n_sub - p - 2) / (n_sub - 1)
    try:
        if hart > 0:
            return float(np.sqrt(max(v @ np.linalg.solve(c, v) * hart, 0.0)))
    except np.linalg.LinAlgError:
        pass
    return float(np.sqrt(np.nansum(v ** 2 / np.diag(c))))


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png")


def record(key, value):
    """Store a number quoted by the documentation page in values.json."""
    vals = json.loads(VALUES_FILE.read_text()) if VALUES_FILE.exists() else {}
    vals[key] = value if isinstance(value, (str, list, dict)) else float(value)
    VALUES_FILE.write_text(json.dumps(vals, indent=1, sort_keys=True))
