"""Tiers B and C: blind re-runs in other cosmologies against the fiducial re-run
(scripts/cosmo_sens/compare_runs.py -> $REMA_COSMO/tierB/compare.json, tierC/compare.json)."""

import json

import numpy as np

from common import BLUE, INK2, MUTED, ORANGE, RESULTS, SERIES, need, panel_label, plt, record, save

for tier in ("tierB", "tierC"):
    path = RESULTS / tier / "compare.json"
    if not need(path):
        continue
    cmp_ = json.loads(path.read_text())
    runs = cmp_["runs"]
    if not runs:
        continue
    cosmos = sorted({r["cosmology"] for r in runs})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    ax = axes[0]
    for k, c in enumerate(cosmos):
        rr = [r for r in runs if r["cosmology"] == c]
        x = np.arange(len(rr)) + 0.12 * (k - len(cosmos) / 2)
        ax.errorbar(x, [100 * r["dlnlam_median"] for r in rr], yerr=[100 * r["dlnlam_nmad"] for r in rr],
                    fmt="o", color=SERIES[k % len(SERIES)], label=c, ms=4)
        record(f"{tier}_dlnlam_{c}", float(np.median([r["dlnlam_median"] for r in rr])))
        record(f"{tier}_centre_changed_{c}", float(np.median([r["centre_changed"] for r in rr])))
        record(f"{tier}_lost_gained_{c}", [int(sum(r["lost"] for r in rr)), int(sum(r["gained"] for r in rr)),
                                         int(sum(r["n_a"] for r in rr))])
    ax.set_xticks(np.arange(len(rr)), [r["region"] for r in rr])
    ax.set_xlabel("region")
    ax.set_ylabel("median Δ ln λ of matched clusters [%] (bars: NMAD)")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.legend(fontsize=7)
    ax = axes[1]
    for k, c in enumerate(cosmos):
        rr = [r for r in runs if r["cosmology"] == c]
        ax.bar(k - 0.2, sum(r["lost"] for r in rr) / max(sum(r["n_a"] for r in rr), 1) * 100, 0.4, color=BLUE,
               label="lost" if k == 0 else None)
        ax.bar(k + 0.2, sum(r["gained"] for r in rr) / max(sum(r["n_a"] for r in rr), 1) * 100, 0.4, color=ORANGE,
               label="gained" if k == 0 else None)
    ax.set_xticks(range(len(cosmos)), cosmos, rotation=20)
    ax.set_ylabel("clusters with λ ≥ 20 without a match [%]")
    ax.legend()
    ax = axes[2]
    lam_edges = cmp_["lam_edges"]
    for k, c in enumerate(cosmos):
        rr = [r for r in runs if r["cosmology"] == c and "ncum_ratio_vs_tierA" in r]
        if not rr:
            continue
        ratio = np.nanmedian(np.array([r["ncum_ratio_vs_tierA"] for r in rr]), axis=(0, 2))
        ax.plot(lam_edges, ratio, "o-", color=SERIES[k % len(SERIES)], label=c)
        record(f"{tier}_ncum_ratio_vs_tierA_{c}", ratio.tolist())
    ax.axhline(1, color=MUTED, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("λ threshold")
    ax.set_ylabel("N(> λ) re-run / fiducial with the tier-A shift")
    panel_label(ax, "1: the response of the re-measured clusters explains the counts")
    save(fig, f"rerun_{tier}")
