"""The DR11 counts and the best-fitting models, with and without the finder's response, and how
the counts move with Omega_m (fit_<tag>.json, forecast_<tag>.json in $REMA_WORK/cosmo_sens)."""

import json
import os

import numpy as np

from common import BLUE, INK2, MUTED, ORANGE, RESULTS, SERIES, need, panel_label, plt, record, save

FIT = os.environ.get("COSMO_FIT", "all_cc_sig025")             # fit_dr11.py --tag
FORECAST = os.environ.get("COSMO_FORECAST", "sig025_cc")  # forecast.py --tag
fit_path, fc_path = RESULTS / f"fit_{FIT}.json", RESULTS / f"forecast_{FORECAST}.json"
if need(fit_path):
    fit = json.loads(fit_path.read_text())
    n = np.asarray(fit["counts"])
    ze = np.asarray(fit["z_edges"])
    le = fit["lam_edges"]
    zc = 0.5 * (ze[1:] + ze[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax = axes[0]
    for i in range(n.shape[0]):
        lab = f"{le[i]:g} ≤ λ < {le[i + 1]:g}" if np.isfinite(le[i + 1]) else f"λ ≥ {le[i]:g}"
        ax.errorbar(zc, n[i], yerr=np.sqrt(n[i]), fmt="o", color=SERIES[i], ms=4, label=lab)
        for f, ls in zip(fit["fits"], ("-", "--")):
            ax.plot(zc, np.asarray(f["counts_model"])[i], color=SERIES[i], ls=ls, lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("z_λ")
    ax.set_ylabel("clusters per bin (volume-limited area)")
    ax.legend(fontsize=7)
    panel_label(ax, "points: DR11; lines: best fit without (solid) / with (dashed) response")
    ax = axes[1]
    for k, f in enumerate(fit["fits"]):
        r = (n - np.asarray(f["counts_model"])) / np.sqrt(np.maximum(np.asarray(f["counts_model"]), 1))
        for i in range(n.shape[0]):
            ax.plot(zc + 0.006 * (k - 0.5), r[i], "o" if k == 0 else "s", color=SERIES[i], ms=4,
                    mfc=SERIES[i] if k == 0 else "none")
        record(f"chi2_counts_{f['model']}", f["chi2"]["counts"])
        record(f"chi2_wl_{f['model']}", f["chi2"]["wl"])
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("z_λ")
    ax.set_ylabel("(data − model) / √model")
    save(fig, "counts_fit")
    record("n_clusters_dv", float(n.sum()))
    record("area_dv", fit["area"][-1])

if need(fc_path):
    fc = json.loads(fc_path.read_text())
    if "dlnN_dOmega_m" in fc:
        tot = np.asarray(fc["dlnN_dOmega_m"]["total"])
        fin = np.asarray(fc["dlnN_dOmega_m"]["finder"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        for i in range(tot.shape[0]):
            axes[0].plot(np.arange(tot.shape[1]), tot[i], "o-", color=SERIES[i])
            axes[1].plot(np.arange(tot.shape[1]), fin[i], "o-", color=SERIES[i])
        for ax, t in zip(axes, ("total (mass function, volume, finder)", "through the finder only")):
            ax.set_xlabel("redshift bin")
            ax.set_ylabel("d ln N / d Ω_m")
            panel_label(ax, t)
        axes[1].axhline(0, color=MUTED, lw=0.6)
        save(fig, "counts_dlnN")
        record("dlnN_dOm_total_median", float(np.median(tot)))
        record("dlnN_dOm_finder_median", float(np.median(fin)))
