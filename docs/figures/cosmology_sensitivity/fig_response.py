"""Tier A: the richness of catalogued clusters re-measured at fixed centres in other cosmologies
(rema remeasure). Reads $REMA_WORK/cosmo_sens/tierA/*.fits: <name>.fits (grid, autodiff, finite
differences, members' free fractions) and <name>_mstar.fits (m* following D_L)."""

from pathlib import Path

import numpy as np

from common import (AQUA, BLUE, INK2, MUTED, ORANGE, RESULTS, need, panel_label, plt, record,
                    save)
from rema.config import CosmologyConfig
from rema.model.cosmo import CosmoTable
from rema.modes.remeasure import read_remeasure

files = sorted(p for p in (RESULTS / "tierA").glob("*.fits") if not p.stem.endswith(("_mstar", "_one")))
if not need(*(files or [RESULTS / "tierA" / "<region>.fits"])):
    raise SystemExit(0)


def load(paths):
    """Clusters of several files, on the cosmologies they share (in the first file's order)."""
    cats = [read_remeasure(p) for p in paths]
    labels = [lab for lab in cats[0][1] if all(lab in c[1] for c in cats)]
    keys = set.intersection(*(set(c[0]) for c in cats))
    out = {}
    for k in keys:
        parts = []
        for c, labs, _ in cats:
            a = np.asarray(c[k])
            if a.ndim == 2 and a.shape[1] == len(labs):
                a = a[:, [labs.index(lab) for lab in labels]]
            parts.append(a)
        out[k] = np.concatenate(parts)
    return out, labels


cat, labels = load(files)
lam, z = cat["LAMBDA"], cat["Z_LAMBDA_CAT"]
good = np.all(lam > 0, axis=1) & np.isfinite(z)
col = {lab: j for j, lab in enumerate(labels)}


def d1(param, lo, hi):
    """Per-cluster d ln(lambda)/d(param) from the symmetric pair (z_lambda iterated)."""
    return (np.log(lam[:, col[f"{param}={hi:g}"]]) - np.log(lam[:, col[f"{param}={lo:g}"]])) / (hi - lo)


_, der = CosmoTable.jvp(CosmologyConfig(), ("Omega_m", "w0"))
fid = CosmoTable.from_config(CosmologyConfig())
zg = np.linspace(0.05, 0.95, 91)
# (a derivative table carries the tangent of the grid, zero: read it on the table's grid)
dlnda = {p: np.interp(zg, np.asarray(fid.z), np.asarray(der[p].da_tab)) / np.asarray(fid.da(zg))
         for p in ("Omega_m", "w0")}
edges = np.array([0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
zc = 0.5 * (edges[1:] + edges[:-1])

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
for ax, (p, lo, hi) in zip(axes[:2], (("Omega_m", 0.25, 0.35), ("w0", -1.2, -0.8))):
    d = d1(p, lo, hi)
    ad = cat[f"DLNLAMBDA_D{p.upper()}"]
    ax.scatter(z[good], d[good], s=3, c=np.log10(lam[good, 0]), cmap="viridis", alpha=0.5, lw=0)
    k = np.digitize(z, edges) - 1
    med = np.array([np.median(d[good & (k == i)]) if np.sum(good & (k == i)) > 10 else np.nan for i in range(zc.size)])
    mad = np.array([np.nanmedian(ad[good & (k == i)]) if np.sum(good & (k == i)) > 10 else np.nan for i in range(zc.size)])
    ax.plot(zc, med, "o-", color=ORANGE, label="median (z_λ iterated)")
    ax.plot(zc, mad, "s--", color=BLUE, ms=4, label="autodiff, fixed z")
    # The elasticity d ln(lambda) / d ln(D_A) that fits the medians.
    ok = np.isfinite(med)
    dl = np.interp(zc, zg, dlnda[p])
    eps = float(np.sum(med[ok] * dl[ok]) / np.sum(dl[ok] ** 2))
    ax.plot(zg, eps * dlnda[p], color=INK2, lw=1, ls=":", label=f"{eps:.2f} × d ln D_A / d{p}")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("z_λ")
    ax.set_ylabel(f"d ln λ / d {p}")
    ax.set_ylim(np.nanpercentile(d[good], 1) - 0.05, np.nanpercentile(d[good], 99) + 0.05)
    ax.legend(loc="lower right")
    record(f"elasticity_{p}", eps)
    record(f"dlnlam_d{p}_median", float(np.median(d[good])))
    for i, zz in enumerate(zc):
        record(f"dlnlam_d{p}_z{zz:.3f}", med[i] if np.isfinite(med[i]) else None)
ad, fd = cat["DLNLAMBDA_DOMEGA_M"], cat.get("DLNLAMBDA_DOMEGA_M_FD")
ax = axes[2]
if fd is not None:
    s = good & np.isfinite(ad) & np.isfinite(fd)
    ax.scatter(fd[s], ad[s], s=4, color=BLUE, alpha=0.5, lw=0)
    lim = np.nanpercentile(np.r_[fd[s], ad[s]], [0.5, 99.5])
    ax.plot(lim, lim, color=INK2, lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("finite differences (Ω_m ± 0.01)")
    ax.set_ylabel("autodiff")
    panel_label(ax, f"d ln λ / dΩ_m at fixed z; median |Δ| = {np.nanmedian(np.abs(ad[s] - fd[s])):.4f}")
    record("ad_fd_median_absdiff", float(np.nanmedian(np.abs(ad[s] - fd[s]))))
save(fig, "response_lambda")

# Other parameters, the redshift, and the reproduction of the catalogue.
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
ax = axes[0]
rows = []
for j, lab in enumerate(labels[1:], start=1):
    dl = np.log(lam[good, j] / lam[good, 0])
    rows.append((lab, np.median(dl), 1.4826 * np.median(np.abs(dl - np.median(dl)))))
    record(f"dlnlam_{lab}", float(np.median(dl)))
y = np.arange(len(rows))
ax.barh(y, [100 * r[1] for r in rows], xerr=[100 * r[2] for r in rows], color=AQUA, ecolor=INK2)
ax.set_yticks(y, [r[0] for r in rows])
ax.set_xlabel("median Δ ln λ [%] (bars: NMAD)")
ax.axvline(0, color=INK2, lw=0.6)
ax = axes[1]
zl = cat["Z_LAMBDA"]
for j, lab in enumerate(labels[1:5], start=1):
    ax.hist(zl[good, j] - zl[good, 0], bins=np.linspace(-0.004, 0.004, 41), histtype="step", label=lab)
    record(f"dz_{lab}", float(np.median(zl[good, j] - zl[good, 0])))
ax.set_xlabel("z_λ(θ) − z_λ(fiducial)")
ax.legend()
ax = axes[2]
r = np.log(lam[good, 0] / cat["LAMBDA_CAT"][good])
ax.hist(r, bins=np.linspace(-0.2, 0.2, 81), color=BLUE)
ax.set_xlabel("ln(λ re-measured, fiducial / λ catalogue)")
panel_label(ax, f"median {np.median(r):+.4f}, NMAD {1.4826 * np.median(np.abs(r - np.median(r))):.4f}")
record("checkpointB_median", float(np.median(r)))
record("checkpointB_nmad", float(1.4826 * np.median(np.abs(r - np.median(r)))))
record("n_tierA", int(good.sum()))
save(fig, "response_checks")

# m* following the luminosity distance.
mfiles = [p.with_name(p.stem + "_mstar.fits") for p in files]
mfiles = [p for p in mfiles if p.exists()]
if mfiles:
    mcat, mlabels = load(mfiles)
    mcol = {lab: j for j, lab in enumerate(mlabels)}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for ax, (p, lo, hi) in zip(axes, (("Omega_m", 0.25, 0.35), ("w0", -1.2, -0.8))):
        ml = mcat["LAMBDA"]
        mg = np.all(ml > 0, axis=1)
        dm = (np.log(ml[:, mcol[f"{p}={hi:g}"]]) - np.log(ml[:, mcol[f"{p}={lo:g}"]])) / (hi - lo)
        mz = mcat["Z_LAMBDA_CAT"]
        k = np.digitize(mz, edges) - 1
        med_m = [np.median(dm[mg & (k == i)]) if np.sum(mg & (k == i)) > 10 else np.nan for i in range(zc.size)]
        d = d1(p, lo, hi)
        k2 = np.digitize(z, edges) - 1
        med = [np.median(d[good & (k2 == i)]) if np.sum(good & (k2 == i)) > 10 else np.nan for i in range(zc.size)]
        ax.plot(zc, med, "o-", color=ORANGE, label="m* fixed (as the catalogue)")
        ax.plot(zc, med_m, "s-", color=AQUA, label="m* follows D_L")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xlabel("z_λ")
        ax.set_ylabel(f"median d ln λ / d {p}")
        ax.legend()
        record(f"dlnlam_d{p}_mstar_median", float(np.median(dm[mg])))
    save(fig, "response_mstar")

# The response against the depth: the z-band 10 sigma depth of the z_vlim map at each cluster.
try:
    zmap = __import__("cosmo_common").load_map()
except SystemExit:
    zmap = None
if zmap is not None and "RA" in cat:
    depth = zmap.value_at(cat["RA"], cat["DEC"], "DEPTH_Z10")
    d = d1("Omega_m", 0.25, 0.35)
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    dedges = np.arange(21.8, 23.21, 0.2)
    dc = 0.5 * (dedges[1:] + dedges[:-1])
    for (z0, z1), col in (((0.2, 0.45), BLUE), ((0.45, 0.7), ORANGE)):
        s = good & np.isfinite(depth) & (z >= z0) & (z < z1) & (lam[:, 0] >= 20)
        k = np.digitize(depth, dedges) - 1
        med = [np.median(d[s & (k == i)]) if np.sum(s & (k == i)) > 20 else np.nan for i in range(dc.size)]
        ax.plot(dc, med, "o-", color=col, label=f"{z0} < z_λ < {z1}, λ ≥ 20")
        for i in range(dc.size):
            if np.isfinite(med[i]):
                record(f"dlnlam_dOm_depth{dc[i]:.1f}_z{z0}", float(med[i]))
    ax.axvline(22.5, color=MUTED, lw=0.6)
    ax.set_xlabel("z-band 10σ depth [mag] (DECaLS left, DES right of 22.5)")
    ax.set_ylabel("median d ln λ / d Ω_m")
    ax.legend()
    save(fig, "response_depth")
