"""Cluster auto-correlation w_p(r_p) of every sample, and its large-scale bias.

Figures wp_lambda.png, wp_external.png and bias_lambda.png; values wp_* and bias_* in values.json.
The bias is b^2 = w_p / w_p,lin with w_p,lin the linear matter w_p (colossus, the catalogue's
Omega_m = 0.3, h = 0.7, with Omega_b = 0.049, sigma_8 = 0.81, n_s = 0.965) integrated to the same
pi_max, fitted with the jackknife covariance over 5 <= r_p <= 30 h^-1 Mpc.
"""

import numpy as np
from astropy.io import fits
from common import (EXTERNAL, LABEL, RESULTS, SAMPLES, SERIES, Z_EDGES, load, obs_label, plt, record, save, snr, vector,
                    z_label)

RP_FIT = (5.0, 30.0)
PI_MAX = 100.0   # h^-1 Mpc, as measured


def wp_linear(rp, z):
    """Linear matter w_p(r_p) [h^-1 Mpc] at redshift z, integrated to PI_MAX."""
    from colossus.cosmology import cosmology

    cosmo = cosmology.setCosmology("rema", {"flat": True, "H0": 70.0, "Om0": 0.3, "Ob0": 0.049,
                                            "sigma8": 0.81, "ns": 0.965}, persistence="")
    pi = np.linspace(0.0, PI_MAX, 401)
    r = np.sqrt(rp[:, None] ** 2 + pi[None, :] ** 2)
    xi = cosmo.correlationFunction(r.ravel(), z).reshape(r.shape)
    return 2.0 * np.trapezoid(xi, pi, axis=1)


def bias(g):
    """Large-scale bias and its error from the w_p group ``g`` (generalised least squares on b^2)."""
    x, v, e, c = vector(g)
    k = (x >= RP_FIT[0]) & (x <= RP_FIT[1]) & np.isfinite(v) & np.isfinite(np.diag(c))
    if k.sum() < 2 or not np.isfinite(c[np.ix_(k, k)]).all():
        return np.nan, np.nan
    m = wp_linear(x[k], float(g.attrs["z_mean"]))
    ci = np.linalg.pinv(c[np.ix_(k, k)])
    n = g["subsamples"].shape[0]
    ci *= (n - k.sum() - 2) / (n - 1)
    var = 1.0 / (m @ ci @ m)
    b2 = var * (m @ ci @ v[k])
    if b2 <= 0:
        return np.nan, np.nan
    return float(np.sqrt(b2)), float(0.5 * np.sqrt(var) / np.sqrt(b2))


def plot_bins(ax, data, sample, zb, scale=True):
    for ob in sorted(o for (z, o) in data if z == zb):
        x, v, e, _ = vector(data[zb, ob])
        f = x if scale else 1.0
        ok = np.isfinite(v) & np.isfinite(e)
        ax.errorbar(x[ok] * (1 + 0.02 * ob), (f * v)[ok], (f * e)[ok], fmt="o-", ms=3,
                    color=SERIES[ob % len(SERIES)], label=obs_label(sample, ob))
    ax.set_xscale("log")


def main():
    lam = load(RESULTS / "lambda_vlim_wp.h5", "wp")
    if lam:
        nz = len(Z_EDGES["lambda_vlim"]) - 1
        fig, axs = plt.subplots(1, nz, figsize=(2.6 * nz, 2.9), sharey=True)
        for zb, ax in enumerate(axs):
            plot_bins(ax, lam, "lambda_vlim", zb)
            ax.set_title(z_label("lambda_vlim", zb))
            ax.set_xlabel(r"$r_p$ [$h^{-1}$Mpc]")
        axs[0].set_ylabel(r"$r_p\,w_p(r_p)$ [$h^{-2}$Mpc$^2$]")
        axs[0].set_yscale("log")
        axs[0].set_ylim(100, 2e4)
        axs[-1].legend(loc="upper left")
        save(fig, "wp_lambda")

    # Samples with at least one finite w_p (SPT: too few pairs at every r_p).
    ext = [s for s in EXTERNAL if any(np.isfinite(g["value"][:]).any() for g in load(RESULTS / f"{s}_wp.h5", "wp").values())]
    for s in set(EXTERNAL) - set(ext):
        record(f"wp_none_{s}", "no finite w_p")
    fig, axs = plt.subplots(1, len(ext), figsize=(3.2 * len(ext), 2.9), sharex=True)
    for ax, s in zip(axs, ext):
        d = load(RESULTS / f"{s}_wp.h5", "wp")
        for i, (zb, ob) in enumerate(sorted(d)):
            x, v, e, c = vector(d[zb, ob])
            ok = np.isfinite(v) & np.isfinite(e)
            lab = z_label(s, zb) + ("" if s in ("erass1", "psz2_noise") else ", " + obs_label(s, ob))
            ax.errorbar(x[ok] * (1 + 0.03 * i), (x * v)[ok], (x * e)[ok], fmt="o-", ms=3,
                        color=SERIES[i % len(SERIES)], label=lab)
            record(f"wp_snr_{s}_z{zb}_o{ob}", snr(v[ok], c[np.ix_(ok, ok)], d[zb, ob]["subsamples"].shape[0]))
        ax.set_xscale("log")
        ax.set_xlim(0.8, 60)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(LABEL[s])
        ax.set_xlabel(r"$r_p$ [$h^{-1}$Mpc]")
        ax.legend(loc="upper left", fontsize=6.5)
    axs[0].set_ylabel(r"$r_p\,w_p(r_p)$ [$h^{-2}$Mpc$^2$]")
    save(fig, "wp_external")

    # Large-scale bias against the mean richness of each bin.
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    rows = {}
    for s, mk in (("lambda_vlim", "o"), ("lambda_vlim_s", "s")):
        d = load(RESULTS / f"{s}_wp.h5", "wp")
        lam_mean = {}
        cat = SAMPLES / f"{s}.fits"
        if cat.exists():
            t = fits.getdata(cat, "CLUSTERS")
            for k in d:
                sel = (t["BIN_Z"] == k[0]) & (t["BIN_OBS"] == k[1])
                lam_mean[k] = float(np.mean(t["LAMBDA"][sel]))
        for zb in sorted({k[0] for k in d}):
            pts = [(lam_mean.get((zb, ob), np.nan), *bias(d[zb, ob])) for ob in sorted(o for (z, o) in d if z == zb)]
            pts = np.array(pts)
            rows[s, zb] = pts
            if s == "lambda_vlim":
                ax.errorbar(pts[:, 0] * (1 + 0.01 * zb), pts[:, 1], pts[:, 2], fmt=mk + "-", ms=3.5,
                            color=SERIES[zb], label=z_label(s, zb))
            for ob, (lm, b, eb) in enumerate(pts):
                record(f"bias_{s}_z{zb}_o{ob}", [round(lm, 2), round(b, 3), round(eb, 3)])
    ax.set_xscale("log")
    ax.set_xlabel(r"$\langle\lambda\rangle$")
    ax.set_ylabel(r"$b$ ($5\leq r_p\leq30\,h^{-1}$Mpc)")
    ax.legend(loc="upper left")
    save(fig, "bias_lambda")



if __name__ == "__main__":
    main()
