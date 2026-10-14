"""Comparison with Comparat et al. (2025, A&A 700, A271): the projected cluster x galaxy
cross-correlation w_p(r_p) of eRASS1 clusters with stellar-mass selected galaxies, for all
galaxies, the red sequence (RS) and the blue cloud (BC).

Our measurements: ``$REMA_WORK/cluster_sumstat/c25/*.h5`` (sum_stat
``scripts/measure_cluster_wp_cross.py``, run by ``scripts/cc_cluster_xgal.sh submit-c25``);
theirs: the 1000 resamplings of ``$CXG_DATA`` (default ``~/software/cxg/data``), averaged with a
2% systematic floor as in their Figs. 1 and 2. Figures c25_cxg.png (C1xG1075 and C0xG1025:
cross, auto and ratio) and c25_lambda.png (richness bins x G1075); values c25_* in values.json.
"""

import os
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table
from common import INK2, ORANGE, RESULTS, SERIES, plt, record, save

CXG = Path(os.environ.get("CXG_DATA", Path.home() / "software" / "cxg" / "data"))
C25 = RESULTS / "c25"
RED, BLUECLOUD, ALL = "#b2182b", "#2166ac", "#0b0b0b"
COLOURS = {"": ALL, "RS": RED, "BC": BLUECLOUD}
NAMES = {"": "all", "RS": "red sequence", "BC": "blue cloud"}
# Their large-scale biases (Sect. 2.3 and Seppi et al. 2024): b_C / b_G.
BIAS_RATIO = {"c1xg1075": 3.35 / 1.43, "c0xg1025": 2.95 / 1.29}
SYST = 0.02


def theirs_cross(pair, cls):
    """r_p mid, w_p, error of their cross-correlation (mean and scatter of the resamplings)."""
    d = CXG / {"c1xg1075": "C1xG1075", "c0xg1025": "C0xG1025"}[pair]
    files = sorted(d.glob(f"*_{cls or 'ANY'}_*NSIDE_08_J_*.fits"))
    if not files:
        return None
    t0 = Table.read(files[0])
    w = np.array([Table.read(f)["wprp"] for f in files])
    y = w.mean(0)
    return np.asarray(t0["rp_min"]), np.asarray(t0["rp_mid"]), y, np.hypot(w.std(0), SYST * y)


def theirs_auto(pair, cls):
    thr, ztag = {"c1xg1075": ("10.75", "01z03"), "c0xg1025": ("10.25", "01z02")}[pair]
    dd = next(iter(CXG.glob(f"_{cls or 'ANY'}_{thr}_Mstar_*_JK_wprp100")), None)
    if dd is None:
        return None
    full = next(iter(dd.glob(f"*HpxMask-{ztag}.fits")))
    t = Table.read(full)
    w = np.array([Table.read(f)["wprp"] for f in sorted(dd.glob("*NSIDE_08_J_*.fits"))])
    y = np.asarray(t["wprp"])
    return np.asarray(t["rp_min"]), np.asarray(t["rp_mid"]), y, np.hypot(w.std(0) / w.mean(0) * y, SYST * y)


def ours(label):
    f = C25 / f"{label}.h5"
    if not f.exists():
        return None
    g = h5py.File(f, "r")["wp"][label]
    e = g["bin_edges"][:]
    return e[:-1], g["sep_centres"][:], g["value"][:], np.sqrt(np.diag(g["cov"][:])), dict(g.attrs)


def on_common(a, b):
    """The values of a and b on their common bins (matched by lower edge)."""
    ia, ib = [], []
    for i, lo in enumerate(a[0]):
        j = np.flatnonzero(np.isclose(np.log10(b[0]), np.log10(lo), atol=1e-3))
        if j.size:
            ia.append(i), ib.append(j[0])
    return np.array(ia, int), np.array(ib, int)


def band(ax, x, y, e, color, label=None, ls="-"):
    ax.plot(x, x * y, ls, color=color, lw=1.2, label=label)
    ax.fill_between(x, x * (y - e), x * (y + e), color=color, alpha=0.2, lw=0)


def main():
    pairs = [("c1xg1075", "g1075", "C1xG1075"), ("c0xg1025", "g1025", "C0xG1025")]
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 7.0), sharex=True)
    for row, (pair, auto, title) in enumerate(pairs):
        for cls in ("", "RS", "BC"):
            col = COLOURS[cls]
            tc, ta, oc, oa = theirs_cross(pair, cls), theirs_auto(pair, cls), ours(pair + cls), ours(auto + cls)
            if tc is not None:
                band(axs[row, 0], tc[1], tc[2], tc[3], col, f"C25 {NAMES[cls]}")
            if ta is not None:
                band(axs[row, 1], ta[1], ta[2], ta[3], col, f"C25 {NAMES[cls]}", ls="--")
            if oc is not None:
                axs[row, 0].errorbar(oc[1], oc[1] * oc[2], oc[1] * oc[3], fmt="o", ms=2.5, color=col,
                                     label=f"this work {NAMES[cls]} ({oc[4]['n_clusters']} clusters)")
            if oa is not None:
                axs[row, 1].errorbar(oa[1], oa[1] * oa[2], oa[1] * oa[3], fmt="s", ms=2.5, color=col, mfc="none",
                                     label=f"this work {NAMES[cls]}")
            if tc is not None and ta is not None:
                r = tc[2] / ta[2]
                re = r * np.hypot(tc[3] / tc[2], ta[3] / ta[2])
                axs[row, 2].plot(tc[1], r, "-", color=col, lw=1.2, label=f"C25 {NAMES[cls]}")
                axs[row, 2].fill_between(tc[1], r - re, r + re, color=col, alpha=0.2, lw=0)
            if oc is not None and oa is not None:
                r = oc[2] / oa[2]
                re = r * np.hypot(oc[3] / oc[2], oa[3] / oa[2])
                axs[row, 2].errorbar(oc[1], r, re, fmt="o", ms=2.5, color=col, label=f"this work {NAMES[cls]}")
                k = (oc[1] > 3) & (oc[1] < 15)
                record(f"c25_ratio_large_{pair}{cls}", round(float(np.average(r[k], weights=1 / re[k] ** 2)), 3))
            if oc is not None and tc is not None:
                ia, ib = on_common(oc, tc)
                q = oc[2][ia] / tc[2][ib]
                for lo, hi, tag in ((0.1, 1.0, "small"), (3.0, 15.0, "large")):
                    k = (oc[1][ia] > lo) & (oc[1][ia] < hi)
                    record(f"c25_ours_over_theirs_{tag}_{pair}{cls}", round(float(np.nanmedian(q[k])), 3))
            if oa is not None and ta is not None:
                ia, ib = on_common(oa, ta)
                q = oa[2][ia] / ta[2][ib]
                k = (oa[1][ia] > 0.1) & (oa[1][ia] < 15)
                record(f"c25_auto_ours_over_theirs_{auto}{cls}", round(float(np.nanmedian(q[k])), 3))
            if oc is not None:
                record(f"c25_n_{pair}{cls}", [int(oc[4]["n_clusters"]), int(oc[4]["n_gal"]), round(float(oc[4]["z_mean"]), 3)])
        alls = ours(pair.replace("c1x", "c1allx")) if pair.startswith("c1") else None
        if alls is not None:
            axs[row, 0].errorbar(alls[1], alls[1] * alls[2], alls[1] * alls[3], fmt="x", ms=3, color=INK2,
                                 label=f"this work all, no $L_X$ cut ({alls[4]['n_clusters']})")
        axs[row, 2].axhline(BIAS_RATIO[pair], color=ORANGE, ls="--", lw=1, label=r"C25 $b_C/b_G$")
        axs[row, 0].set_ylabel(r"$r_p\,w_p(r_p)$ [$h^{-2}$Mpc$^2$]")
        axs[row, 0].set_title(f"{title}: cluster x galaxy", fontsize=8.5)
        axs[row, 1].set_title(f"{title}: galaxy x galaxy", fontsize=8.5)
        axs[row, 2].set_title(f"{title}: ratio cross / auto", fontsize=8.5)
        axs[row, 0].set_yscale("log"), axs[row, 1].set_yscale("log"), axs[row, 2].set_yscale("log")
        axs[row, 0].set_ylim(10, 3000), axs[row, 1].set_ylim(5, 1000), axs[row, 2].set_ylim(0.5, 30)
        axs[row, 0].legend(fontsize=5.8, loc="lower left")
        axs[row, 2].legend(fontsize=5.8, loc="upper right")
    for ax in axs.ravel():
        ax.set_xscale("log")
        ax.axvspan(15, 60, color="#d0d0d0", alpha=0.5, lw=0)
    for ax in axs[1]:
        ax.set_xlabel(r"$r_p$ [$h^{-1}$Mpc]")
    axs[1, 1].legend(fontsize=5.8, loc="lower left")
    axs[0, 0].set_xlim(0.05, 50)
    save(fig, "c25_cxg")

    # Richness bins x G1075, 0.1 < z < 0.3, against C1xG1075.
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=True)
    for ax, cls in zip(axs, ("", "RS", "BC")):
        tc = theirs_cross("c1xg1075", cls)
        if tc is not None:
            band(ax, tc[1], tc[2], tc[3], ALL, r"C25 C1 ($\log L_X>43.1$)")
        oc = ours("c1xg1075" + cls)
        if oc is not None:
            ax.errorbar(oc[1], oc[1] * oc[2], oc[1] * oc[3], fmt="x", ms=3, color=INK2, label="this work eRASS1 C1")
        for i, (lab, name) in enumerate((("l20", r"$20\leq\lambda<30$"), ("l30", r"$30\leq\lambda<45$"),
                                         ("l45", r"$45\leq\lambda<60$"), ("l60", r"$\lambda\geq60$"))):
            o = ours(f"{lab}xg1075{cls}")
            if o is None:
                continue
            ax.errorbar(o[1] * (1 + 0.02 * i), o[1] * o[2], o[1] * o[3], fmt="o-", ms=2.5, lw=0.8, color=SERIES[i],
                        label=f"{name} ({o[4]['n_clusters']})")
            ia = (o[1] > 3) & (o[1] < 15)
            record(f"c25_lambda_{lab}{cls}_large", round(float(np.mean(o[1][ia] * o[2][ia])), 1))
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.axvspan(15, 60, color="#d0d0d0", alpha=0.5, lw=0)
        ax.set_title(f"clusters x G1075 {NAMES[cls]}, $0.1<z<0.3$", fontsize=8.5)
        ax.set_xlabel(r"$r_p$ [$h^{-1}$Mpc]")
        ax.set_xlim(0.05, 50), ax.set_ylim(10, 3000)
    axs[0].set_ylabel(r"$r_p\,w_p(r_p)$ [$h^{-2}$Mpc$^2$]")
    axs[0].legend(fontsize=6, loc="lower left")
    save(fig, "c25_lambda")


if __name__ == "__main__":
    main()
