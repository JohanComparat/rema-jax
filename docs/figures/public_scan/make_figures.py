"""Figures and tables of docs/public_catalogues_scan.rst: rema scan-positions on the public cluster
catalogues (dr11/south/rema/rema_dr11_v0.2.0_scan_public), against the catalogues' own redshifts
and mass or richness, and against the DR11 south production catalogue. Rows scanned with λ > 0 and
MASKFRAC < 0.2 (the production cut); "λ > 20" and the matches are counted among them.

    python docs/figures/public_scan/make_figures.py [SCAN_DIR] [PRODUCTION]

Defaults: $REMA_DATA/rema_dr11_v0.2.0_scan_public and $REMA_DATA/rema_dr11_v0.2.0/clusters_dr11.fits,
REMA_DATA = ~/data/legacysurvey/dr11/south/rema. Writes PNGs and comparison.rst next to this script.
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from astropy.io import fits

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("REMA_DATA", Path.home() / "data/legacysurvey/dr11/south/rema"))

# (file stem, label, x column for the mass/richness panel, x label)
CATS = [
    ("act_dr6", "ACT DR6", "INPUT_M500", r"$M_{500c}$ [$10^{14} M_\odot$]"),
    ("act_dr5", "ACT DR5", "INPUT_M500", r"$M_{500c}^{\rm Cal}$ [$10^{14} M_\odot$]"),
    ("spt_2500d", "SPT-SZ 2500d", "INPUT_M500", r"$M_{500c}$ [$10^{14} M_\odot$]"),
    ("psz2", "Planck PSZ2", "INPUT_M500", r"$M_{\rm SZ}$ [$10^{14} M_\odot$]"),
    ("mcxc", "MCXC", "INPUT_M500", r"$M_{500}$ [$10^{14} M_\odot$]"),
    ("erass1", "eRASS1", "INPUT_M500", r"$M_{500}$ [$10^{14} M_\odot$]"),
    ("sdss_dr8", "SDSS DR8 redMaPPer", "INPUT_LAMBDA", r"$\lambda$ (SDSS)"),
    ("des_y1", "DES Y1 redMaPPer", "INPUT_LAMBDA", r"$\lambda$ (DES)"),
    ("wen_han_2024", "Wen & Han 2024", "INPUT_LAMBDA", r"$\lambda_{\rm WH}$"),
]
COLS = ["INPUT_Z", "INPUT_M500", "INPUT_LAMBDA", "RA", "DEC", "LAMBDA", "Z_LAMBDA", "SCANNED", "NSPEC",
        "BEST_Z", "BEST_Z_TYPE", "MASKFRAC"]
MAX_MASKFRAC = 0.2      # the production cut (mask.max_maskfrac): positions outside the footprint have 1


def read(path):
    with fits.open(path, memmap=True) as h:
        d = h["CLUSTERS"].data
        return {c: np.array(d[c]) for c in COLS if c in d.names}


def match(ra, dec, pra, pdec, radius_deg):
    """Nearest production cluster within radius_deg: index (or -1)."""
    from scipy.spatial import cKDTree

    def xyz(r, d):
        r, d = np.radians(r), np.radians(d)
        return np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]

    dist, idx = cKDTree(xyz(pra, pdec)).query(xyz(ra, dec), distance_upper_bound=2 * np.sin(np.radians(radius_deg) / 2))
    return np.where(np.isfinite(dist), idx, -1)


def panel_grid():
    fig, axes = plt.subplots(3, 3, figsize=(11, 10.5))
    return fig, axes.ravel()


def main(argv):
    scan = Path(argv[1]) if len(argv) > 1 else DATA / "rema_dr11_v0.2.0_scan_public"
    prod_path = Path(argv[2]) if len(argv) > 2 else DATA / "rema_dr11_v0.2.0/clusters_dr11.fits"
    with fits.open(prod_path, memmap=True) as h:
        p = h["CLUSTERS"].data
        prod = {c: np.array(p[c]) for c in ("RA", "DEC", "LAMBDA", "Z_LAMBDA")}
    cats = {stem: read(scan / f"{stem}.fits") for stem, *_ in CATS}

    rows = []
    fz, az = panel_grid()
    fl, al = panel_grid()
    fp, ap = panel_grid()
    for k, (stem, label, xcol, xlabel) in enumerate(CATS):
        c = cats[stem]
        pos = c["SCANNED"] & (c["LAMBDA"] > 0)
        good = pos & (c["MASKFRAC"] < MAX_MASKFRAC)
        zin = c["INPUT_Z"]
        okz = good & np.isfinite(zin) & (zin > 0)
        dz = (c["Z_LAMBDA"] - zin) / (1 + zin)
        m = np.full(c["LAMBDA"].size, -1)
        m[good] = match(c["RA"][good], c["DEC"][good], prod["RA"], prod["DEC"], 3 / 60)
        mm = m >= 0
        rows.append((label, c["LAMBDA"].size, int(c["SCANNED"].sum()), int(pos.sum()), int(good.sum()),
                     int(np.sum(good & (c["LAMBDA"] > 20))), int(mm.sum()),
                     np.median(c["LAMBDA"][mm] / prod["LAMBDA"][m[mm]]),
                     np.median(np.abs(c["Z_LAMBDA"][mm] - prod["Z_LAMBDA"][m[mm]])),
                     np.median(np.abs(dz[okz])), np.mean(np.abs(dz[okz]) < 0.05),
                     int(np.sum(c["NSPEC"][good] >= 3))))

        # z_lambda against the catalogue's redshift
        ax = az[k]
        big = okz.sum() > 20000
        if big:
            ax.hexbin(zin[okz], c["Z_LAMBDA"][okz], gridsize=80, extent=(0, 1, 0, 1), bins="log", cmap="Greys",
                      mincnt=1)
        else:
            ax.scatter(zin[okz], c["Z_LAMBDA"][okz], s=3, alpha=0.4, lw=0, c="k")
        ax.plot([0, 1], [0, 1], "r-", lw=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=f"{label} ({okz.sum():,})")
        ax.text(0.04, 0.9, f"med |dz|/(1+z) = {np.median(np.abs(dz[okz])):.3f}\n"
                f"{100 * np.mean(np.abs(dz[okz]) < 0.05):.0f}% within 0.05", transform=ax.transAxes, fontsize=8,
                va="top")
        if k >= 6:
            ax.set_xlabel("catalogue redshift")
        if k % 3 == 0:
            ax.set_ylabel(r"$z_\lambda$ (rema scan)")

        # lambda against the catalogue's mass or richness
        ax = al[k]
        x = c[xcol] if xcol in c else np.full(c["LAMBDA"].size, np.nan)
        okx = good & np.isfinite(x) & (x > 0)
        if okx.sum() > 20000:
            ax.hexbin(np.log10(x[okx]), np.log10(c["LAMBDA"][okx]), gridsize=80, bins="log", cmap="Greys", mincnt=1)
            ax.set_xlabel("log10 " + xlabel)
            ax.set_ylabel(r"log10 $\lambda$ (rema scan)")
        else:
            ax.scatter(x[okx], c["LAMBDA"][okx], s=3, alpha=0.4, lw=0, c="k")
            ax.set(xscale="log", yscale="log", xlabel=xlabel)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if k % 3 == 0:
                ax.set_ylabel(r"$\lambda$ (rema scan)")
            if xcol == "INPUT_LAMBDA":
                lim = (3, 300)
                ax.plot(lim, lim, "r-", lw=0.8)
        ax.set_title(f"{label} ({okx.sum():,})")

        # lambda against the production catalogue
        ax = ap[k]
        lp, ls = prod["LAMBDA"][m[mm]], c["LAMBDA"][mm]
        if mm.sum() > 20000:
            ax.hexbin(np.log10(lp), np.log10(ls), gridsize=80, extent=(0.4, 2.6, 0.4, 2.6), bins="log", cmap="Greys",
                      mincnt=1)
            ax.plot([0.4, 2.6], [0.4, 2.6], "r-", lw=0.8)
            ax.set(xlabel=r"log10 $\lambda$ production", ylabel=r"log10 $\lambda$ scan")
        else:
            ax.scatter(lp, ls, s=3, alpha=0.4, lw=0, c="k")
            ax.plot([3, 400], [3, 400], "r-", lw=0.8)
            ax.set(xscale="log", yscale="log", xlim=(3, 400), ylim=(1, 400))
            if k >= 6:
                ax.set_xlabel(r"$\lambda$ production (nearest within 3')")
            if k % 3 == 0:
                ax.set_ylabel(r"$\lambda$ (rema scan)")
        ax.set_title(f"{label} ({mm.sum():,})")

    for fig, name in ((fz, "z_vs_catalogue.png"), (fl, "lambda_vs_mass.png"), (fp, "lambda_vs_production.png")):
        fig.tight_layout()
        fig.savefig(HERE / name, dpi=110)
        print("wrote", HERE / name)

    head = ("Catalogue", "Rows", "Scanned", "λ > 0", "MASKFRAC < 0.2", "λ > 20", "Matched (3′)", "λ / production",
            "\\|Δz\\| production", "\\|Δz\\|/(1+z) catalogue", "Within 0.05", "≥ 3 spec-z")
    out = [".. list-table::", "   :header-rows: 1", ""]
    for r in [head] + [(lab, f"{n:,}", f"{s:,}", f"{ps:,}", f"{g:,}", f"{l20:,}", f"{nm:,}", f"{rat:.2f}",
                        f"{dzp:.3f}", f"{dzi:.3f}", f"{100 * f05:.0f}%", f"{ns:,}")
                       for lab, n, s, ps, g, l20, nm, rat, dzp, dzi, f05, ns in rows]:
        out.append("   * - " + r[0])
        out += [f"     - {v}" for v in r[1:]]
    (HERE / "comparison.rst").write_text("\n".join(out) + "\n")
    print("wrote", HERE / "comparison.rst")


if __name__ == "__main__":
    main(sys.argv)
