"""Figures and value tables of the packaged model tables (docs/tables.rst).

    python docs/figures/make_model_tables.py

Writes ``model_tables_mstar.png``, ``model_tables_colors.png`` and ``model_tables_values.rst``
next to this file, from the tables in ``rema/data``.
"""

from importlib import resources
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rema.data.build import FILTER_SETS  # noqa: E402
from rema.io.tables import read_table  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = Path(str(resources.files("rema.data")))

# Reference categorical palette (light), in its fixed order; text and grid in neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"vis": "VIS", "y": "Y", "j": "J", "h": "H", "w1": "W1"}
TITLE = {"legacy": "DECam griz + WISE W1", "lsst": "LSST ugrizy", "euclid": "Euclid VIS, Y, J, H"}
Z_YOUNG = 2.0                     # above this the z_f = 3 population is younger than ~1 Gyr
Z_VALUES = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5)


def label(set_name, band):
    return LABEL.get(band, band) if set_name != "lsst" else band


def style(ax, xmax=2.5):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlim(0, xmax)
    ax.axvspan(Z_YOUNG, xmax, color="#f0efec", lw=0, zorder=0)


def end_labels(ax, x, items, min_sep):
    """Direct labels at the right end of the lines, spread to at least ``min_sep`` apart in y."""
    items = sorted(items)
    ys = [y for y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_sep)
    shift = (np.mean([y for y, _ in items]) - np.mean(ys)) if ys else 0.0
    for y, (_, text) in zip(ys, items):
        ax.annotate(text, (x, y + shift), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK, annotation_clip=False)


def legend_below(ax, n):
    ax.legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2 if n > 2 else 1)


def young_note(ax):
    ax.text(Z_YOUNG + 0.04, 0.98, "young\n(z_f = 3)", transform=ax.get_xaxis_transform(), va="top",
            fontsize=7, color=INK2)


def mstar_figure(path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.0), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, fs) in zip(axes, FILTER_SETS.items()):
        style(ax)
        ends = []
        for k, b in enumerate(fs.bands):
            if b not in fs.mstar_bands:
                continue
            t = read_table(DATA / fs.mstar_file(b))
            ax.plot(t["Z"], t["MSTAR"], color=SERIES[k], lw=1.6, label=f"{label(name, b)} (BC03)")
            ends.append((float(t["MSTAR"][-1]), label(name, b)))
        if len(ends) <= 4:                                   # more lines: the legend alone
            end_labels(ax, float(t["Z"][-1]), ends, 0.6)
        # redMaPPer's empirical tables, where they are tabulated (z <= 1.2), dashed.
        emp = {"legacy": [("des_z03", "z")], "lsst": [("lsst_r03", "r"), ("lsst_i03", "i"), ("lsst_z03", "z")]}
        for table, b in emp.get(name, []):
            t = read_table(DATA / f"mstar_{table}.fits")
            keep = np.asarray(t["Z"]) <= 1.2 + 1e-9
            ax.plot(np.asarray(t["Z"])[keep], np.asarray(t["MSTAR"])[keep], color=SERIES[fs.bands.index(b)],
                    lw=1.2, ls=(0, (4, 2)), label=f"{label(name, b)} (redMaPPer {table})")
        ax.set_title(TITLE[name], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("redshift", fontsize=9, color=INK2)
        legend_below(ax, len(ax.get_lines()))
    axes[0].set_ylabel("m* [AB]", fontsize=9, color=INK2)
    axes[0].invert_yaxis()
    young_note(axes[2])
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def colors_figure(path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.0), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, fs) in zip(axes, FILTER_SETS.items()):
        style(ax)
        t = read_table(DATA / fs.colors_file)
        z, c = np.asarray(t["Z"]), np.asarray(t["COLOR"])
        ends = []
        for j in range(c.shape[1]):
            text = f"{label(name, fs.bands[j])}−{label(name, fs.bands[j + 1])}"
            ax.plot(z, c[:, j], color=SERIES[j], lw=1.6, label=text)
            ends.append((float(c[-1, j]), text))
        if len(ends) <= 4:
            end_labels(ax, float(z[-1]), ends, 0.15)
        ax.set_title(TITLE[name], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("redshift", fontsize=9, color=INK2)
        legend_below(ax, c.shape[1])
    axes[0].set_ylabel("adjacent colour [mag]", fontsize=9, color=INK2)
    young_note(axes[2])
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def values_rst(path):
    """List tables of m* and of the colours at a few redshifts (the figures' data)."""
    def at(t, values):
        return np.interp(Z_VALUES, np.asarray(t["Z"]), values, right=np.nan)

    out = [".. Generated by docs/figures/make_model_tables.py; do not edit.", ""]
    for name, fs in FILTER_SETS.items():
        cols = [(f"m* {label(name, b)}", at(t := read_table(DATA / fs.mstar_file(b)), np.asarray(t["MSTAR"])))
                for b in fs.mstar_bands]
        if name == "legacy":
            t = read_table(DATA / "mstar_des_z03.fits")
            cols.append(("m* z (des_z03)", at(t, np.asarray(t["MSTAR"]))))
        if name == "lsst":
            for b in "riz":
                t = read_table(DATA / f"mstar_lsst_{b}03.fits")
                cols.append((f"m* {b} (lsst_{b}03)", at(t, np.asarray(t["MSTAR"]))))
        t = read_table(DATA / fs.colors_file)
        c = np.asarray(t["COLOR"])
        cols += [(f"{label(name, fs.bands[j])}−{label(name, fs.bands[j + 1])}", at(t, c[:, j]))
                 for j in range(c.shape[1])]
        out += [f".. list-table:: {TITLE[name]}: m* [AB] and adjacent colours [mag]",
                "   :header-rows: 1", "   :class: tight-table", "",
                "   * - z", *[f"     - {h}" for h, _ in cols]]
        for i, z in enumerate(Z_VALUES):
            out.append(f"   * - {z:.1f}")
            out += [f"     - {v[i]:.2f}" if np.isfinite(v[i]) else "     - —" for _, v in cols]
        out.append("")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    mstar_figure(HERE / "model_tables_mstar.png")
    colors_figure(HERE / "model_tables_colors.png")
    values_rst(HERE / "model_tables_values.rst")
    print("wrote", *(p.name for p in sorted(HERE.glob("model_tables_*"))))
