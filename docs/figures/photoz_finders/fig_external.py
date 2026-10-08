"""Recovery of external clusters (SZ, X-ray and the Wen & Han 2024 photo-z catalogue) by each
finder at matched false-detection rates, against redshift and mass."""

from common import (AREAS, FINDERS, INK2, RATE, available, external, logaxis, matched, np, plt, record,
                    run_path, save, selected)

CATALOGUES = [("act_dr6", "ACT DR6"), ("erass1", "eRASS1"), ("spt_2500d", "SPT-SZ"), ("wen_han_2024", "Wen & Han 2024")]

for area in AREAS:
    fins = [v for v in available(area) if run_path(area, FINDERS[v]["null"]).exists()]
    if not fins:
        continue
    own = AREAS[area]["own"]
    rows = []
    for name, label in CATALOGUES:
        e = external(name)
        if e is None:
            continue
        keep = own.contains(e["RA"], e["DEC"]) & np.isfinite(e["Z"]) & (e["Z"] > 0.05) & (e["Z"] < 0.9)
        if name == "erass1" and "PCONT" in e:
            keep &= e["PCONT"] < 0.5
        if keep.sum() < 5:
            continue
        rows.append((name, label, {k: v[keep] for k, v in e.items()}))
    if not rows:
        continue
    fig, axes = plt.subplots(2, len(rows), figsize=(3.2 * len(rows), 5.4), squeeze=False)
    for j, (name, label, e) in enumerate(rows):
        zedges = np.linspace(0.05, 0.9, 7)
        m = np.asarray(e.get("M500", np.full(e["RA"].size, np.nan)), float)
        good_m = np.isfinite(m) & (m > 0)
        medges = np.geomspace(max(np.nanmin(m[good_m]), 0.2), np.nanmax(m[good_m]), 7) if good_m.sum() > 5 else None
        for v in fins:
            f = FINDERS[v]
            c = selected(area, v, rate=RATE)
            ie, _ = matched(c, f["z"], e)
            hit = np.zeros(e["RA"].size, bool)
            hit[ie] = True
            record(f"{area}_{v}_{name}_recovered", float(hit.mean()))
            record(f"{area}_{name}_n", int(hit.size))
            zc = 0.5 * (zedges[1:] + zedges[:-1])
            fz = [hit[(e["Z"] >= a) & (e["Z"] < b)].mean() if np.sum((e["Z"] >= a) & (e["Z"] < b)) >= 3 else np.nan
                  for a, b in zip(zedges[:-1], zedges[1:])]
            axes[0, j].plot(zc, fz, "o-", color=f["color"], label=f["label"])
            if medges is not None:
                mc = np.sqrt(medges[1:] * medges[:-1])
                fm = [hit[good_m & (m >= a) & (m < b)].mean() if np.sum(good_m & (m >= a) & (m < b)) >= 3 else np.nan
                      for a, b in zip(medges[:-1], medges[1:])]
                axes[1, j].plot(mc, fm, "o-", color=f["color"])
        axes[0, j].set(title=f"{label} ({e['RA'].size})", xlabel="z", ylabel="recovered", ylim=(0, 1.05))
        logaxis(axes[1, j])
        axes[1, j].set(xlabel="M500 [10$^{14}$ M$_\\odot$]", ylabel="recovered", ylim=(0, 1.05))
    axes[0, 0].legend(loc="lower left")
    save(fig, f"external_{area}")
