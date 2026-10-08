"""Colours of the members: the probability-weighted fraction of members on the red sequence
(red-sequence chi^2 < 9 at the cluster redshift, for 3 colours) against redshift and richness. At
high z the colours are noisy and blue galaxies pass the cut more often."""

from common import AREAS, FINDERS, RATE, available, load, logaxis, np, plt, record, run_path, save, selected

for area in AREAS:
    fins = [v for v in available(area) if v != "rs_wcen" and run_path(area, FINDERS[v]["null"]).exists()]
    if not fins:
        continue
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for v in fins:
        f = FINDERS[v]
        c = selected(area, v, rate=RATE)
        _, mem = load(area, v, members=True)
        if not mem or "CHISQ_RS" not in mem:
            continue
        sel = np.isin(mem["MEM_MATCH_ID"], c["MEM_MATCH_ID"])
        w = np.asarray(mem["PMEM" if "PMEM" in mem else "P"], float)[sel]
        red = np.asarray(mem["CHISQ_RS"], float)[sel] < 9
        mid = mem["MEM_MATCH_ID"][sel]
        order = np.argsort(c["MEM_MATCH_ID"])
        row = order[np.searchsorted(c["MEM_MATCH_ID"], mid, sorter=order)]
        wr = np.bincount(row, weights=w * red, minlength=len(c["RA"]))
        wt = np.bincount(row, weights=w, minlength=len(c["RA"]))
        fred = np.where(wt > 0, wr / np.maximum(wt, 1e-9), np.nan)
        z = np.asarray(c[f["z"]], float)
        edges = np.linspace(0.05, 0.9, 9)
        ax[0].plot(0.5 * (edges[1:] + edges[:-1]),
                   [np.nanmedian(fred[(z >= a) & (z < b)]) if np.any((z >= a) & (z < b)) else np.nan
                    for a, b in zip(edges[:-1], edges[1:])], "o-", color=f["color"], label=f["label"])
        lam = np.asarray(c["LAMBDA_STAR" if v.startswith("pscd") else "LAMBDA"], float)
        ax[1].scatter(lam, fred, s=3, color=f["color"], alpha=0.4)
        record(f"{area}_{v}_red_fraction_median", float(np.nanmedian(fred)))
    ax[0].set(xlabel="z", ylabel="red fraction of the members (median)", ylim=(0, 1))
    logaxis(ax[1])
    ax[1].set(xlabel="$\\lambda$ (filter), $\\lambda_*$ (PSCD)", ylabel="red fraction", ylim=(0, 1))
    ax[0].legend()
    save(fig, f"colour_{area}")
