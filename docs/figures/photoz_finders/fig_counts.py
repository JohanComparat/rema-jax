"""Redshift distributions of the finders' clusters at matched false-detection rates."""

from common import AREAS, FINDERS, RATE, available, need, np, own_area, plt, record, run_path, save, selected

for area in AREAS:
    fins = [v for v in available(area) if run_path(area, FINDERS[v]["null"]).exists()]
    if not fins:
        continue
    A = own_area(area)
    edges = np.arange(0.05, 0.951, 0.05)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
    for v in fins:
        f = FINDERS[v]
        c = selected(area, v, rate=RATE)
        z = np.asarray(c[f["z"]], float)
        h, _ = np.histogram(z, edges)
        ax[0].step(edges[:-1], h / A / np.diff(edges), where="post", color=f["color"], label=f["label"])
        x = np.sort(z)
        ax[1].plot(x, np.arange(1, x.size + 1) / A, color=f["color"], label=f["label"])
        record(f"{area}_{v}_count{RATE:g}", int(z.size))
        record(f"{area}_{v}_count{RATE:g}_z>0.6", int(np.sum(z > 0.6)))
    ax[0].set(xlabel="z", ylabel="dN/dz per deg$^2$", title=f"{AREAS[area]['label']}: {RATE:g} false per deg$^2$")
    ax[1].set(xlabel="z", ylabel="N(< z) per deg$^2$")
    ax[0].legend()
    save(fig, f"counts_{area}")
