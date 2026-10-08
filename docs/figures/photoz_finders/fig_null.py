"""False detections from the null tests: cumulative density of detections against each finder's
statistic, for the real and the shuffled galaxies, and the thresholds at matched false rates."""

from common import (AREAS, FINDERS, INK2, RATE, RATES, available, load, need, np, own_area, plt,
                    record, run_path, save, threshold)

for area in AREAS:
    fins = [v for v in available(area) if run_path(area, FINDERS[v]["null"]).exists()]
    if not fins:
        print(f"{area}: no run with its null test")
        continue
    A = own_area(area)
    record(f"{area}_area", A)
    stats = [(v, s) for v in fins for s in FINDERS[v]["stats"]]
    n = len(stats)
    ncol = min(n, 4)
    fig, axes = plt.subplots(-(-n // ncol), ncol, figsize=(3.0 * ncol, 2.6 * -(-n // ncol)), squeeze=False)
    for ax, (v, s) in zip(axes.ravel(), stats):
        f = FINDERS[v]
        cat, _ = load(area, v)
        null, _ = load(area, f["null"])
        for c, col, ls, lab in ((cat, f["color"], "-", "real"), (null, INK2, "--", "null")):
            x = np.sort(np.asarray(c.get(s, np.zeros(0)), float))[::-1]
            ax.step(x, np.arange(1, x.size + 1) / A, where="post", color=col, ls=ls, label=lab)
        for rate in RATES:
            t, pur = threshold(area, v, s, rate)
            if np.isfinite(t):
                ax.axvline(t, color=f["color"], lw=0.6, alpha=0.6)
                record(f"{area}_{v}_{s}_t{rate:g}", t)
                record(f"{area}_{v}_{s}_purity{rate:g}", pur)
                record(f"{area}_{v}_{s}_n{rate:g}", int(np.sum(np.asarray(cat[s]) > t)))
        ax.axhline(RATE, color=INK2, lw=0.6, ls=":")
        ax.set(xscale="log" if s.startswith("LAMBDA") else "linear", yscale="log",
               xlabel=s, ylabel="N(> x) per deg$^2$", title=f["label"])
        ax.legend(loc="upper right")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    save(fig, f"null_{area}")
