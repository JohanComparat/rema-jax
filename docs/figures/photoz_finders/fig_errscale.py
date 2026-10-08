"""Photo-z widths of the DR11 galaxies against their spectroscopic redshifts (the strip)."""

from common import DATA, INK2, ORANGE, BLUE, need, plt, np, record, save

from rema.io.tables import read_table
from rema.validate.photoz import fit_err_scale, nmad

GAL = DATA / "strip" / "galaxies_3sweeps_v2.fits"
if need(GAL):
    g = read_table(GAL, ["ZPHOT", "ZPHOT_STD", "ZSPEC", "REFMAG"])
    s = g["ZSPEC"] > 0
    edges = np.arange(16.0, 23.6, 0.5)
    fit = fit_err_scale(*(g[c][s] for c in ("ZPHOT", "ZPHOT_STD", "ZSPEC", "REFMAG")), edges=edges)
    b = fit["bins"]
    m = np.array([r["mag"] for r in b])
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    ax[0].plot(m, [r["scale"] for r in b], "o-", color=BLUE)
    ax[0].axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax[0].set(xlabel="z-band magnitude", ylabel="NMAD of (ZPHOT - ZSPEC) / ZPHOT_STD", ylim=(0, 1.2))
    ax[1].plot(m, [r["nmad_dz"] for r in b], "o-", color=BLUE, label="NMAD")
    ax[1].plot(m, [abs(r["bias"]) for r in b], "s--", color=ORANGE, label="|median|")
    ax[1].set(xlabel="z-band magnitude", ylabel="dz / (1 + z)", yscale="log")
    ax[1].legend()
    ax[2].plot(m, [r["outliers"] if r["outliers"] > 0 else np.nan for r in b], "o-", color=BLUE)
    ax[2].set(xlabel="z-band magnitude", ylabel="outliers, |dz| / (1 + z) > 0.15", yscale="log")
    ax[0].text(0.03, 0.05, f"{s.sum():,} spectroscopic galaxies", transform=ax[0].transAxes, fontsize=8)
    save(fig, "errscale")
    x = (g["ZPHOT"][s] - g["ZSPEC"][s]) / g["ZPHOT_STD"][s]
    record("errscale_n", int(s.sum()))
    record("errscale_nmad_all", nmad(x))
    for r in b:
        record(f"errscale_{r['lo']:.1f}", {k: round(v, 4) for k, v in r.items()})
