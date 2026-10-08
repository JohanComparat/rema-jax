"""The photo-z finders against the red-sequence finder: how many of each one's clusters the other
has, and their richness and amplitude relations."""

from common import AREAS, FINDERS, INK2, RATE, available, load, logaxis, matched, np, plt, record, run_path, save, selected

for area in AREAS:
    if not run_path(area, "rs_wcen").exists():
        continue
    rs, _ = load(area, "rs_wcen")
    others = [v for v in available(area) if v != "rs_wcen" and run_path(area, FINDERS[v]["null"]).exists()]
    if not others:
        continue
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    lam_edges = np.geomspace(5, 150, 9)
    lc = np.sqrt(lam_edges[1:] * lam_edges[:-1])
    for v in others:
        f = FINDERS[v]
        c = selected(area, v, rate=RATE)
        ref = {"RA": rs["RA"], "DEC": rs["DEC"], "Z": rs["Z_LAMBDA"]}
        # red-sequence clusters (all, lambda >= 5) found by the photo-z finder
        irs, ic = matched(c, f["z"], ref)
        hit = np.zeros(rs["RA"].size, bool)
        hit[irs] = True
        frac = [hit[(rs["LAMBDA"] >= a) & (rs["LAMBDA"] < b)].mean() if np.any((rs["LAMBDA"] >= a) & (rs["LAMBDA"] < b)) else np.nan
                for a, b in zip(lam_edges[:-1], lam_edges[1:])]
        ax[0].plot(lc, frac, "o-", color=f["color"], label=f["label"])
        record(f"{area}_{v}_finds_rs_lambda20", float(hit[rs["LAMBDA"] >= 20].mean()))
        # the photo-z finder's clusters that the red-sequence finder has (lambda >= 5)
        mine = {"RA": c["RA"], "DEC": c["DEC"], "Z": c[f["z"]]}
        im, _ = matched(rs, "Z_LAMBDA", mine)
        has = np.zeros(c["RA"].size, bool)
        has[im] = True
        record(f"{area}_{v}_in_rs", float(has.mean()))
        z = np.asarray(c[f["z"]])
        zedges = np.linspace(0.05, 0.9, 9)
        fz = [has[(z >= a) & (z < b)].mean() if np.any((z >= a) & (z < b)) else np.nan for a, b in zip(zedges[:-1], zedges[1:])]
        ax[1].plot(0.5 * (zedges[1:] + zedges[:-1]), fz, "o-", color=f["color"])
        # richness relation
        y = "LAMBDA_STAR" if v.startswith("pscd") else "LAMBDA"
        ax[2].scatter(rs["LAMBDA"][irs], np.asarray(c[y])[ic], s=3, color=f["color"], alpha=0.5)
    logaxis(ax[0])
    ax[0].set(xlabel="red-sequence $\\lambda$", ylabel="found by the photo-z finder", ylim=(0, 1.05))
    ax[1].set(xlabel="z", ylabel="also a red-sequence cluster", ylim=(0, 1.05))
    logaxis(ax[2])
    logaxis(ax[2], "y")
    ax[2].set(xlabel="red-sequence $\\lambda$", ylabel="$\\lambda$ (filter), $\\lambda_*$ (PSCD)")
    ax[2].plot([3, 300], [3, 300], color=INK2, lw=0.8, ls=":")
    ax[0].legend(loc="lower right")
    save(fig, f"overlap_{area}")
