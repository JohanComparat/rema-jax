"""Cluster redshifts against spectroscopic ones (clusters with a spectroscopic redshift from
their members, or their central galaxy's)."""

from common import AREAS, FINDERS, RATE, available, np, plt, record, run_path, save, selected

from rema.validate.photoz import nmad

for area in AREAS:
    fins = [v for v in available(area) if run_path(area, FINDERS[v]["null"]).exists()]
    if not fins:
        continue
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for v in fins:
        f = FINDERS[v]
        c = selected(area, v, rate=RATE)
        zs = np.where(np.asarray(c.get("BEST_Z_TYPE", np.full(len(c["RA"]), ""))).astype(str) != "photo_z",
                      np.asarray(c.get("BEST_Z", np.full(len(c["RA"]), -1.0)), float), -1.0)
        ok = zs > 0
        z = np.asarray(c[f["z"]], float)[ok]
        dz = (z - zs[ok]) / (1 + zs[ok])
        ax[0].scatter(zs[ok], dz, s=3, color=f["color"], alpha=0.5, label=f"{f['label']} ({ok.sum()})")
        edges = np.linspace(0.05, 0.9, 9)
        zc = 0.5 * (edges[1:] + edges[:-1])
        ax[1].plot(zc, [nmad(dz[(zs[ok] >= a) & (zs[ok] < b)]) if np.sum((zs[ok] >= a) & (zs[ok] < b)) >= 5 else np.nan
                        for a, b in zip(edges[:-1], edges[1:])], "o-", color=f["color"])
        if ok.sum():
            record(f"{area}_{v}_dz_nmad", nmad(dz))
            record(f"{area}_{v}_dz_median", float(np.median(dz)))
            record(f"{area}_{v}_dz_outliers", float(np.mean(np.abs(dz) > 0.05)))
            record(f"{area}_{v}_nspecz", int(ok.sum()))
    ax[0].set(xlabel="spectroscopic z", ylabel="(z - z$_{spec}$) / (1 + z$_{spec}$)", ylim=(-0.1, 0.1))
    ax[1].set(xlabel="spectroscopic z", ylabel="NMAD", ylim=(0, None))
    ax[0].legend(markerscale=3)
    save(fig, f"redshift_{area}")
