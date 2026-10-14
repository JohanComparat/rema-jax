"""Excess surface density ΔΣ(R) around the clusters, with DES Y3, HSC Y3 and KiDS-1000 sources.

Figures esd_lambda.png and esd_external.png: the surveys that overlap a sample combined,
inverse-variance weighted per bin with the diagonal jackknife errors; values esd_* in values.json.
ΔΣ is comoving, in h M_sun pc^-2, and R in comoving h^-1 Mpc.
"""

import numpy as np
from common import LABEL, RESULTS, SERIES, Z_EDGES, load, obs_label, plt, record, save, snr, z_label

SURVEYS = ("esd_des", "esd_hsc", "esd_kids")


def esd(g):
    """R [h^-1 Mpc], ΔΣ [h M_sun pc^-2], its error and covariance."""
    h = float(g["cosmology/h"][()])
    x = g["sep_centres"][:] * h
    v, c = g["value"][:] / h, g["cov"][:] / h ** 2
    return x, v, np.sqrt(np.diag(c)), c


def combined(groups):
    """Inverse-variance combination of the surveys present (diagonal errors)."""
    xs, vs, ws = [], [], []
    for sv in SURVEYS:
        if sv in groups:
            x, v, e, _ = esd(groups[sv])
            xs.append(x), vs.append(v), ws.append(1.0 / e ** 2)
    v, w = np.array(vs), np.array(ws)
    return xs[0], (v * w).sum(0) / w.sum(0), 1.0 / np.sqrt(w.sum(0))


def groups_of(path):
    out = {}
    for sv in SURVEYS:
        for k, g in load(path, sv).items():
            out.setdefault(k, {})[sv] = g
    return out


def main():
    lam = groups_of(RESULTS / "lambda_vlim_esd.h5")
    if lam:
        nz = len(Z_EDGES["lambda_vlim"]) - 1
        fig, axs = plt.subplots(1, nz, figsize=(2.6 * nz, 2.9), sharey=True)
        for zb, ax in enumerate(axs):
            for ob in sorted(o for (z, o) in lam if z == zb):
                x, v, e = combined(lam[zb, ob])
                ok = v > 0
                ax.errorbar(x[ok] * (1 + 0.02 * ob), v[ok], e[ok], fmt="o-", ms=3, color=SERIES[ob],
                            label=obs_label("lambda_vlim", ob))
                i = int(np.argmin(abs(x - 1.0)))
                record(f"esd_lambda_vlim_z{zb}_o{ob}", [round(float(v[i]), 1), round(float(e[i]), 1)])
                for sv, g in lam[zb, ob].items():
                    _, vv, _, cc = esd(g)
                    record(f"esd_snr_{sv}_lambda_vlim_z{zb}_o{ob}", snr(vv, cc, g["subsamples"].shape[0]))
            ax.set_xscale("log"), ax.set_yscale("log")
            ax.set_title(z_label("lambda_vlim", zb))
            ax.set_xlabel(r"$R$ [$h^{-1}$Mpc]")
        axs[0].set_ylabel(r"$\Delta\Sigma$ [$h\,M_\odot\,{\rm pc}^{-2}$]")
        axs[0].set_ylim(0.3, 300)
        axs[-1].legend(loc="lower left")
        save(fig, "esd_lambda")

    ext = [s for s in ("erass1", "act_dr6", "psz2_noise") if (RESULTS / f"{s}_esd.h5").exists()]
    fig, axs = plt.subplots(1, len(ext), figsize=(3.2 * len(ext), 3.0), sharey=True)
    for ax, s in zip(np.atleast_1d(axs), ext):
        d = groups_of(RESULTS / f"{s}_esd.h5")
        for i, k in enumerate(sorted(d)):
            x, v, e = combined(d[k])
            ok = v > 0
            lab = z_label(s, k[0]) + ("" if s in ("erass1", "psz2_noise") else ", " + obs_label(s, k[1]))
            ax.errorbar(x[ok] * (1 + 0.03 * i), v[ok], e[ok], fmt="o-", ms=3, color=SERIES[i % len(SERIES)], label=lab)
            j = int(np.argmin(abs(x - 1.0)))
            record(f"esd_{s}_z{k[0]}_o{k[1]}", [round(float(v[j]), 1), round(float(e[j]), 1), "+".join(sorted(d[k]))])
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_title(LABEL[s])
        ax.set_xlabel(r"$R$ [$h^{-1}$Mpc]")
        ax.legend(loc="lower left", fontsize=6.5)
    np.atleast_1d(axs)[0].set_ylabel(r"$\Delta\Sigma$ [$h\,M_\odot\,{\rm pc}^{-2}$]")
    np.atleast_1d(axs)[0].set_ylim(0.3, 1000)
    save(fig, "esd_external")


if __name__ == "__main__":
    main()
