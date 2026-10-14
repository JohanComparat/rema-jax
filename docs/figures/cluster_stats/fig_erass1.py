"""Figures of docs/erass1_dr11.rst: the eRASS1 clusters confirmed on DR11, binned in L500.

erass1_sample.png   L500 against z_lambda with the bins, lambda against L500, the redshifts
erass1_catalogue.png  z_lambda and lambda of DR11 against the eRASS1 catalogue (DR10 redMaPPer)
erass1_qa.png       data / weighted randoms against the exposure, N_H, the depth and z_vlim
erass1_wp.png       w_p(r_p) per redshift and L500 bin, and the large-scale bias
erass1_esd.png      Delta Sigma(R) per redshift and L500 bin (DES Y3, HSC Y3, KiDS-1000 combined)
erass1_xgal.png     w_x(r_p) with the LS DR11 galaxies in photo-z slices
erass1_lx.png       3D w_p(r_p) of the L500 bins at 0.1 < z < 0.3 with G1075 (all, RS, BC)

Inputs: the sample file and QA of ``rema cluster-randoms`` (``$REMA_DATA/rema/rema_dr11_v0.2.0_randoms``),
the measurements in ``$REMA_WORK/cluster_sumstat`` (``erass1_wp.h5``, ``erass1_esd.h5``,
``xgal/erass1_x_M*.h5``, ``c25/e1lx*.h5``, ``c25/g1075*.h5``) and, for the comparison, ``$CXG_DATA``.
"""

import json

import h5py
import numpy as np
from astropy.io import fits
from common import (INK2, ORANGE, RESULTS, SAMPLES, SERIES, Z_EDGES, load, obs_label, plt, record, save, snr,
                    vector, z_label)

S = "erass1"
LOG_EDGES = [43.1, 43.7, 44.3]          # log10 L500 [erg/s]; the catalogue's L500 is in 1e42 erg/s
RED, BLUECLOUD, ALL = "#b2182b", "#2166ac", "#0b0b0b"


def sample_figure():
    f = SAMPLES / f"{S}.fits"
    if not f.exists():
        print(f"missing {f}: skipped")
        return
    c = fits.getdata(f, "CLUSTERS")
    r = fits.getdata(f, "RANDOMS")
    logl = np.log10(c["EXT_L500"]) + 42
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3))
    ax = axs[0]
    ax.scatter(c["Z_LAMBDA"], logl, s=2, c=np.log10(c["LAMBDA"]), cmap="viridis", rasterized=True)
    for e in LOG_EDGES:
        ax.axhline(e, color=INK2, lw=0.7, ls="--")
    for z in Z_EDGES[S]:
        ax.axvline(z, color=INK2, lw=0.7, ls=":")
    ax.set_xlabel(r"$z_\lambda$"), ax.set_ylabel(r"$\log_{10} L_{500}$ [erg s$^{-1}$]")
    ax.set_title(f"{len(logl)} clusters (colour: log $\\lambda$)")
    ax = axs[1]
    for zb in range(len(Z_EDGES[S]) - 1):
        m = c["BIN_Z"] == zb
        ax.scatter(logl[m], c["LAMBDA"][m], s=2, color=SERIES[zb], label=z_label(S, zb), rasterized=True)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10} L_{500}$ [erg s$^{-1}$]"), ax.set_ylabel(r"$\lambda$")
    ax.legend(markerscale=4, loc="upper left")
    ax = axs[2]
    zb = np.linspace(0.1, 0.8, 36)
    for ob in range(4):
        md, mr = c["BIN_OBS"] == ob, r["BIN_OBS"] == ob
        if md.sum() == 0:
            continue
        nd, _ = np.histogram(c["Z_LAMBDA"][md], zb)
        nr, _ = np.histogram(r["Z"][mr], zb, weights=r["WEIGHT"][mr])
        nr *= md.sum() / max(r["WEIGHT"][mr].sum(), 1e-30)
        ax.step(zb[:-1], nd, where="post", color=SERIES[ob], label=obs_label(S, ob) + f" ({md.sum()})")
        ax.step(zb[:-1], nr, where="post", color=SERIES[ob], ls=":", lw=1)
    ax.set_xlabel(r"$z_\lambda$"), ax.set_ylabel("clusters per bin")
    ax.set_title("data (solid), weighted randoms (dotted)")
    ax.legend(fontsize=6.5)
    save(fig, "erass1_sample")
    counts = {f"z{zb}_o{ob}": int(((c["BIN_Z"] == zb) & (c["BIN_OBS"] == ob)).sum())
              for zb in range(len(Z_EDGES[S]) - 1) for ob in range(4)}
    record("erass1_counts", counts)
    record("erass1_n", int(len(c)))
    record("erass1_lambda_median_per_lx", [round(float(np.median(c["LAMBDA"][c["BIN_OBS"] == ob])), 1) for ob in range(4)])


def catalogue_figure():
    f, fj = SAMPLES / f"{S}.fits", SAMPLES / f"{S}_joined.fits"
    if not (f.exists() and fj.exists()):
        print(f"missing {f} or {fj}: skipped")
        return
    c = fits.getdata(f, "CLUSTERS")
    j = fits.getdata(fj)[c["ROW"]]          # ROW: the row of the joined scan
    dz = (c["Z_LAMBDA"] - j["INPUT_Z"]) / (1 + j["INPUT_Z"])
    ok = j["INPUT_LAMBDA"] > 0
    ratio = c["LAMBDA"][ok] / j["INPUT_LAMBDA"][ok]
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.3))
    ax = axs[0]
    ax.scatter(j["INPUT_Z"], c["Z_LAMBDA"], s=2, color=SERIES[0], rasterized=True)
    ax.plot([0, 0.9], [0, 0.9], color=INK2, lw=0.7)
    ax.set_xlabel("eRASS1 catalogue redshift"), ax.set_ylabel(r"$z_\lambda$ (rema, DR11)")
    ax = axs[1]
    ax.hist(dz, np.linspace(-0.06, 0.06, 61), color=SERIES[0])
    ax.set_xlabel(r"$(z_\lambda - z_{\rm eRASS1})/(1+z_{\rm eRASS1})$"), ax.set_ylabel("clusters")
    ax = axs[2]
    ax.scatter(j["INPUT_LAMBDA"][ok], c["LAMBDA"][ok], s=2, color=SERIES[1], rasterized=True)
    ax.plot([1, 400], [1, 400], color=INK2, lw=0.7)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$ (eRASS1 catalogue, DR10)"), ax.set_ylabel(r"$\lambda$ (rema, DR11)")
    save(fig, "erass1_catalogue")
    med = float(np.median(dz))
    record("erass1_dz", {"median": round(med, 4), "nmad": round(float(1.4826 * np.median(abs(dz - med))), 4),
                         "frac_gt_0.05": round(float(np.mean(abs(dz) > 0.05)), 4)})
    record("erass1_lambda_ratio", [round(float(x), 3) for x in np.percentile(ratio, [16, 50, 84])])
    lum = (c["LAMBDA"] < 10) & (np.log10(c["EXT_L500"]) + 42 > 44)
    record("erass1_low_lambda_luminous", int(lum.sum()))


def qa_figure():
    f = SAMPLES / f"{S}_qa.json"
    if not f.exists():
        print(f"missing {f}: skipped")
        return
    q = json.loads(f.read_text())
    keys = [("EXPOSURE", "eROSITA exposure [s]"), ("NH", r"$N_{\rm H}$ [cm$^{-2}$]"),
            ("DEPTH_Z", "DR11 z depth"), ("ZVLIM", r"$z_{\rm vlim}$"), ("ABS_GLAT", r"$|b|$ [deg]")]
    fig, axs = plt.subplots(1, len(keys), figsize=(13.0, 2.8), sharey=True)
    for ax, (k, lab) in zip(axs, keys):
        if k not in q:
            continue
        for key, mk, col, name in (("all", "o", ALL, "with $P_{\\rm ext}$"),
                                   ("all_without_external", "s", ORANGE, "without")):
            d = q[k].get(key)
            if d is None:
                continue
            x = np.asarray(d["CENTRE"], float)
            ax.errorbar(x, d["RATIO"], d["ERR"], fmt=mk + "-", ms=3, color=col, mfc="none" if key != "all" else col,
                        label=f"{name}: $\\chi^2$ = {d['CHI2']:.0f}/{d['NDOF']}")
            record(f"erass1_qa_{k}_{key}", [round(d["CHI2"], 1), d["NDOF"], round(float(np.min(d["RATIO"])), 2),
                                            round(float(np.max(d["RATIO"])), 2)])
        ax.axhline(1, color=INK2, lw=0.6)
        if k in ("EXPOSURE", "NH"):
            ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.legend(fontsize=6, loc="lower center")
    axs[0].set_ylabel("data / weighted randoms")
    axs[0].set_ylim(0.3, 2.5)
    save(fig, "erass1_qa")


def galaxy_bias(label, z=0.2, rp=(5.0, 15.0)):
    """Large-scale bias of a galaxy auto-correlation of ``c25/`` (w_p in h^-1 Mpc) against the linear
    matter w_p at redshift ``z``."""
    from fig_wp import wp_linear

    f = RESULTS / "c25" / f"{label}.h5"
    if not f.exists():
        return np.nan, np.nan
    g = h5py.File(f, "r")["wp"][label]
    x, v, c = g["sep_centres"][:], g["value"][:], g["cov"][:]
    k = (x > rp[0]) & (x < rp[1])
    m = wp_linear(x[k], z)
    n = g["subsamples"].shape[0]
    ci = np.linalg.pinv(c[np.ix_(k, k)]) * (n - k.sum() - 2) / (n - 1)
    var = 1.0 / (m @ ci @ m)
    b2 = var * (m @ ci @ v[k])
    return float(np.sqrt(b2)), float(0.5 * np.sqrt(var / b2))


def halo_mass(b, z=0.2):
    """log10 M200m [h^-1 M_sun] of the haloes of bias ``b`` (Tinker et al. 2010, colossus)."""
    from colossus.cosmology import cosmology
    from colossus.lss import bias as cbias

    cosmology.setCosmology("rema", {"flat": True, "H0": 70.0, "Om0": 0.3, "Ob0": 0.049, "sigma8": 0.81,
                                    "ns": 0.965}, persistence="")
    m = np.logspace(12.5, 15.8, 400)
    return float(np.log10(np.interp(b, cbias.haloBias(m, z, "200m", model="tinker10"), m)))


def cross_bias():
    """Cluster bias of the L500 bins at 0.1 < z < 0.3 from the 3D cross-correlation with G1075:
    b_C = (w_p^CxG / w_p^GxG) b_G over 3 < r_p < 15 h^-1 Mpc; rows (log <L500>, b_C, error)."""
    from fig_compare_c25 import ours

    bg, ebg = galaxy_bias("g1075")
    record("erass1_bias_g1075", [round(bg, 3), round(ebg, 3)])
    auto = ours("g1075")
    c = fits.getdata(SAMPLES / f"{S}.fits", "CLUSTERS")
    rows = []
    for ob, (lo, hi) in enumerate(zip([0, 12.59, 50.12, 199.5], [12.59, 50.12, 199.5, np.inf])):
        o = ours(f"e1lx{ob}xg1075")
        if o is None or auto is None:
            continue
        r = o[2] / auto[2]
        re = np.abs(r) * np.hypot(o[3] / o[2], auto[3] / auto[2])
        k = (o[1] > 3) & (o[1] < 15) & (r > 0)
        w = 1 / re[k] ** 2
        rat, erat = float(np.sum(w * r[k]) / np.sum(w)), float(1 / np.sqrt(np.sum(w)))
        m = (c["Z_LAMBDA"] >= 0.1) & (c["Z_LAMBDA"] < 0.3) & (c["EXT_L500"] >= lo) & (c["EXT_L500"] < hi)
        lx = float(np.log10(np.mean(c["EXT_L500"][m])) + 42)
        b = rat * bg
        eb = b * np.hypot(erat / rat, ebg / bg)
        rows.append((lx, b, eb))
        record(f"erass1_cross_bias_lx{ob}", [round(lx, 2), round(b, 2), round(eb, 2)])
        record(f"erass1_cross_mass_lx{ob}", [round(halo_mass(b - eb), 2), round(halo_mass(b), 2), round(halo_mass(b + eb), 2)])
    return np.array(rows)


def wp_figure():
    from fig_wp import bias

    d = load(RESULTS / f"{S}_wp.h5", "wp")
    if not d:
        return
    nz = len(Z_EDGES[S]) - 1
    fig, axs = plt.subplots(1, nz + 1, figsize=(3.0 * (nz + 1), 3.0))
    c = fits.getdata(SAMPLES / f"{S}.fits", "CLUSTERS")
    for zb in range(nz):
        ax = axs[zb]
        for ob in sorted(o for (z, o) in d if z == zb):
            x, v, e, cov = vector(d[zb, ob])
            ok = np.isfinite(v) & np.isfinite(e)
            ax.errorbar(x[ok] * (1 + 0.02 * ob), (x * v)[ok], (x * e)[ok], fmt="o-", ms=3, color=SERIES[ob],
                        label=obs_label(S, ob))
            b, eb = bias(d[zb, ob])
            m = (c["BIN_Z"] == zb) & (c["BIN_OBS"] == ob)
            lx = float(np.log10(np.mean(c["EXT_L500"][m])) + 42)
            record(f"erass1_wp_z{zb}_o{ob}", {"n": int(m.sum()), "snr": round(snr(v[ok], cov[np.ix_(ok, ok)],
                                                                                   d[zb, ob]["subsamples"].shape[0]), 1),
                                               "bias": [round(b, 2), round(eb, 2)], "log_lx_mean": round(lx, 2)})
            if np.isfinite(b):
                axs[-1].errorbar(lx, b, eb, fmt="o", color=SERIES[zb], ms=4,
                                 label=z_label(S, zb) if ob == min(o for (z, o) in d if z == zb) else None)
        ax.set_xscale("log"), ax.set_xlim(0.8, 60)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(z_label(S, zb))
        ax.set_xlabel(r"$r_p$ [$h^{-1}$Mpc]")
        ax.legend(fontsize=6.5, loc="upper left")
    axs[0].set_ylabel(r"$r_p\,w_p(r_p)$ [$h^{-2}$Mpc$^2$]")
    axs[-1].set_xlabel(r"$\log_{10}\langle L_{500}\rangle$ [erg s$^{-1}$]")
    axs[-1].set_ylabel(r"$b$ ($5\leq r_p\leq30\,h^{-1}$Mpc)")
    cb = cross_bias()
    if len(cb):
        axs[-1].errorbar(cb[:, 0], cb[:, 1], cb[:, 2], fmt="s-", ms=4, color=ALL, mfc="none",
                         label=r"$0.1\leq z<0.3$, from $w_p^{C\times G}/w_p^{G\times G}$")
    axs[-1].axhspan(2.95 - 0.21, 2.95 + 0.21, color=INK2, alpha=0.15, lw=0)
    axs[-1].axhspan(3.35 - 0.23, 3.35 + 0.23, color=INK2, alpha=0.25, lw=0)
    axs[-1].set_title("bias (grey: Seppi+24 C0, C1)")
    axs[-1].legend(fontsize=6.5)
    save(fig, "erass1_wp")


def esd_figure():
    from fig_esd import combined, groups_of

    d = groups_of(RESULTS / f"{S}_esd.h5")
    if not d:
        return
    nz = len(Z_EDGES[S]) - 1
    fig, axs = plt.subplots(1, nz, figsize=(3.2 * nz, 3.0), sharey=True)
    for zb, ax in enumerate(axs):
        for ob in sorted(o for (z, o) in d if z == zb):
            x, v, e = combined(d[zb, ob])
            ok = v > 0
            ax.errorbar(x[ok] * (1 + 0.03 * ob), v[ok], e[ok], fmt="o-", ms=3, color=SERIES[ob],
                        label=obs_label(S, ob) + " (" + "+".join(s[4:].upper() for s in sorted(d[zb, ob])) + ")")
            j = int(np.argmin(abs(x - 1.0)))
            record(f"erass1_esd_z{zb}_o{ob}", [round(float(v[j]), 1), round(float(e[j]), 1)])
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_title(z_label(S, zb))
        ax.set_xlabel(r"$R$ [$h^{-1}$Mpc]")
        ax.legend(fontsize=6, loc="lower left")
    axs[0].set_ylabel(r"$\Delta\Sigma$ [$h\,M_\odot\,{\rm pc}^{-2}$]")
    axs[0].set_ylim(0.3, 1000)
    save(fig, "erass1_esd")


def xgal_figure():
    files = sorted((RESULTS / "xgal").glob(f"{S}_x_M*.h5"))
    files = [f for f in files if "onebin" not in f.stem]
    if not files:
        return
    nz = 2
    fig, axs = plt.subplots(2, nz, figsize=(7.0, 5.6), sharex=True, sharey=True)
    for f in files:
        thr = f.stem.split("_x_M")[1]
        row = 1 if thr == "11.25" else 0
        d = load(f, "xgal")
        for (zb, ob), g in sorted(d.items()):
            if zb >= nz:
                continue
            h = float(g.attrs["h"])
            x, v, cov = g["sep_centres"][:] * h, g["value"][:], g["cov"][:]
            e = np.sqrt(np.diag(cov))
            ax = axs[row, zb]
            ax.errorbar(x * (1 + 0.02 * ob), x * v, x * e, fmt="o-", ms=3, color=SERIES[ob], label=obs_label(S, ob))
            ax.set_title(z_label(S, zb) + rf", $\log M^*>{thr}$", fontsize=8)
            record(f"erass1_xgal_M{thr}_z{zb}_o{ob}", round(snr(v, cov, g["subsamples"].shape[0]), 1))
    for ax in axs.ravel():
        ax.set_xscale("log"), ax.set_yscale("log")
    for ax in axs[1]:
        ax.set_xlabel(r"$r_p$ [$h^{-1}$Mpc]")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$r_p\,w_x(r_p)$ [$h^{-1}$Mpc]")
    axs[0, 0].legend(fontsize=6.5, loc="lower left")
    save(fig, "erass1_xgal")


def lx_figure():
    from fig_compare_c25 import ours, theirs_auto, theirs_cross

    names = {"": "all galaxies", "RS": "red sequence", "BC": "blue cloud"}
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.4), sharex=True)
    for col, cls in enumerate(("", "RS", "BC")):
        auto = ours("g1075" + cls)
        tc = theirs_cross("c1xg1075", cls)
        if tc is not None:
            axs[0, col].plot(tc[1], tc[1] * tc[2], color=ALL, lw=1.2, label=r"C25 C1 ($\log L_X>43.1$)")
        for ob in range(4):
            o = ours(f"e1lx{ob}xg1075{cls}")
            if o is None:
                continue
            axs[0, col].errorbar(o[1] * (1 + 0.02 * ob), o[1] * o[2], o[1] * o[3], fmt="o-", ms=2.5, lw=0.8,
                                 color=SERIES[ob], label=obs_label(S, ob) + f" ({o[4]['n_clusters']})")
            k = (o[1] > 3) & (o[1] < 15)
            k1 = (o[1] > 0.1) & (o[1] < 1)
            record(f"erass1_lx{ob}{cls}", {"n": int(o[4]["n_clusters"]), "rpwp_large": round(float(np.mean((o[1] * o[2])[k])), 1),
                                           "rpwp_small": round(float(np.nanmean((o[1] * o[2])[k1])), 1)})
            if auto is not None:
                r = o[2] / auto[2]
                re = np.abs(r) * np.hypot(o[3] / o[2], auto[3] / auto[2])
                ok = (r > 0) & np.isfinite(re)
                axs[1, col].errorbar(o[1][ok] * (1 + 0.02 * ob), r[ok], re[ok], fmt="o-", ms=2.5, lw=0.8, color=SERIES[ob])
                kk = k & ok
                record(f"erass1_lx{ob}{cls}_ratio_large", [round(float(np.average(r[kk], weights=1 / re[kk] ** 2)), 2),
                                                           round(float(1 / np.sqrt(np.sum(1 / re[kk] ** 2))), 2)])
        axs[0, col].set_title(f"eRASS1 x G1075 {names[cls]}, $0.1<z<0.3$", fontsize=8.5)
        axs[1, col].set_title(f"ratio to the G1075 {names[cls]} auto-correlation", fontsize=8.5)
        axs[0, col].set_yscale("log"), axs[1, col].set_yscale("log")
        axs[0, col].set_ylim(10, 5000), axs[1, col].set_ylim(0.5, 50)
        axs[1, col].set_xlabel(r"$r_p$ [$h^{-1}$Mpc]")
    for ax in axs.ravel():
        ax.set_xscale("log"), ax.set_xlim(0.05, 50)
        ax.axvspan(15, 60, color="#d0d0d0", alpha=0.5, lw=0)
    axs[0, 0].set_ylabel(r"$r_p\,w_p(r_p)$ [$h^{-2}$Mpc$^2$]")
    axs[1, 0].set_ylabel(r"$w_p^{C\times G}/w_p^{G\times G}$")
    axs[0, 0].legend(fontsize=6, loc="lower left")
    save(fig, "erass1_lx")


if __name__ == "__main__":
    sample_figure()
    catalogue_figure()
    qa_figure()
    wp_figure()
    esd_figure()
    xgal_figure()
    lx_figure()
