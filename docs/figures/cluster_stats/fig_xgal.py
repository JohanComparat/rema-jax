"""Cluster x galaxy angular cross-correlation w_x(r_p) in photo-z slices, with the LS DR11
stellar-mass volume-limited galaxy samples of ls11-rmagGAL.

Figure xgal_lambda.png: each richness and redshift bin with the galaxy sample of lowest stellar
mass threshold volume limited to the bin's upper edge (top) and with log M* > 11.25 (bottom);
xgal_external.png: the X-ray and SZ samples. Values xgal_* in values.json.
"""

import numpy as np
from common import EXTERNAL, LABEL, RESULTS, SERIES, Z_EDGES, load, obs_label, plt, record, save, snr, z_label

# The galaxy sample of each redshift bin of the richness samples (scripts/cc_cluster_xgal.sh).
LAMBDA_GAL = {0: "9.75", 1: "10.25", 2: "10.75", 3: "11.25"}


def vec(g):
    h = float(g.attrs["h"])
    x = g["sep_centres"][:] * h
    v, c = g["value"][:], g["cov"][:]
    return x, v, np.sqrt(np.diag(c)), c


def main():
    xdir = RESULTS / "xgal"
    fig, axs = plt.subplots(2, 4, figsize=(11.0, 5.6), sharex=True, sharey="row")
    for zb in range(4):
        for row, thr in enumerate((LAMBDA_GAL[zb], "11.25")):
            ax = axs[row, zb]
            d = load(xdir / f"lambda_vlim_x_M{thr}.h5", "xgal")
            for ob in sorted(o for (z, o) in d if z == zb):
                x, v, e, c = vec(d[zb, ob])
                ax.errorbar(x * (1 + 0.02 * ob), x * v, x * e, fmt="o-", ms=3, color=SERIES[ob],
                            label=obs_label("lambda_vlim", ob))
                record(f"xgal_lambda_vlim_M{thr}_z{zb}_o{ob}", [round(float(snr(v, c, d[zb, ob]["subsamples"].shape[0])), 1),
                                                                int(d[zb, ob].attrs["n_clusters_common"]),
                                                                int(d[zb, ob].attrs["n_gal"])])
            ax.set_xscale("log"), ax.set_yscale("log")
            ax.set_title(z_label("lambda_vlim", zb) + rf", $\log M^*>{thr}$", fontsize=8)
            if row == 1:
                ax.set_xlabel(r"$r_p$ [$h^{-1}$Mpc]")
        axs[0, 0].set_ylabel(r"$r_p\,w_x(r_p)$ [$h^{-1}$Mpc]")
        axs[1, 0].set_ylabel(r"$r_p\,w_x(r_p)$ [$h^{-1}$Mpc]")
    axs[0, -1].legend(loc="lower right", fontsize=6.5)
    save(fig, "xgal_lambda")

    fig, axs = plt.subplots(1, len(EXTERNAL), figsize=(2.9 * len(EXTERNAL), 2.9), sharey=True)
    for ax, s in zip(axs, EXTERNAL):
        i = 0
        for f in sorted(p for p in xdir.glob(f"{s}_x_M*.h5") if "onebin" not in p.stem):
            thr = f.stem.split("_x_M")[1]
            d = load(f, "xgal")
            for (zb, ob) in sorted(d):
                x, v, e, c = vec(d[zb, ob])
                lab = z_label(s, zb) + ("" if s in ("erass1", "psz2_noise") else ", " + obs_label(s, ob)) + f", M*>{thr}"
                ax.errorbar(x * (1 + 0.02 * i), x * v, x * e, fmt="o-" if thr == "11.25" else "s--", ms=3,
                            color=SERIES[i % len(SERIES)], label=lab)
                record(f"xgal_{s}_M{thr}_z{zb}_o{ob}", round(float(snr(v, c, d[zb, ob]["subsamples"].shape[0])), 1))
                i += 1
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_title(LABEL[s])
        ax.set_xlabel(r"$r_p$ [$h^{-1}$Mpc]")
        ax.legend(loc="lower left", fontsize=5.5)
    axs[0].set_ylabel(r"$r_p\,w_x(r_p)$ [$h^{-1}$Mpc]")
    save(fig, "xgal_external")


if __name__ == "__main__":
    main()
