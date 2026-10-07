"""Photometric-redshift figures (Rykoff et al. 2014 Figs. 7, 9-12; Rykoff et al. 2016 Figs. 4-5;
Kluge et al. 2024 Figs. 16-17).

    python docs/figures/redmapper_dr11/fig_redshifts.py
"""

# %% [script-only]
from common import *  # noqa: F403

# %% [markdown]
# ## Cluster redshifts
#
# z_λ against spectroscopic redshifts. The spectroscopic redshifts come with the DR11
# photo-z sweeps (the compilation of their `Z_SPEC` column: SDSS, BOSS, eBOSS, DESI, GAMA,
# ...). For a cluster they are either the redshift of the central galaxy (`CG_SPEC_Z`) or the
# biweight mean of the members kept by the velocity clipping (`SPEC_Z_BOOT`, at least 3
# members). The calibration area (RA 160–180° and 190–210°, Dec −10–10°) trained the
# red-sequence model and the z_λ correction with these redshifts, so the statistics are given
# for the rest of the sky ("independent") and for the calibration area separately.

# %%
cl = load("clusters")
lam, zl, zle = np.asarray(cl["LAMBDA"]), np.asarray(cl["Z_LAMBDA"]), np.asarray(cl["Z_LAMBDA_E"])
cgz = np.asarray(cl["CG_SPEC_Z"])
calib = np.asarray(cl["IN_CALIB"])
base = (lam >= 20) & (cgz > 0.03) & (zl > 0.05)
print(f"{(lam >= 20).sum():,} clusters with lambda >= 20, {base.sum():,} with a central spectroscopic "
      f"redshift ({(base & ~calib).sum():,} outside the calibration area)")

# Clusters with >= 2 other members (P >= 0.8) with a spectroscopic redshift within 1000 km/s of
# the central's (Rykoff et al. 2014 Fig. 10).
mem = load("members", ["CL_ROW", "P", "ZSPEC", "CENT_RANK"])
mrow, mp, mzs = np.asarray(mem["CL_ROW"]), np.asarray(mem["P"]), np.asarray(mem["ZSPEC"])
use = (mp >= 0.8) & (mzs > 0) & (np.asarray(mem["CENT_RANK"]) != 0)
use &= base[mrow]
dv = C_KMS * np.abs(mzs[use] - cgz[mrow[use]]) / (1 + cgz[mrow[use]])
n_close = np.bincount(mrow[use][dv < 1000.0], minlength=lam.size)
clean = base & (n_close >= 2)
print(f"{clean.sum():,} of them with >= 2 spectroscopic members within 1000 km/s of the central")

ZEDGES = np.arange(0.05, 0.851, 0.05)


def zstats(sel):
    """Per z_spec bin: bias, NMAD/(1+z), mean sigma_z/(1+z), 4 sigma outlier fraction."""
    dz = (zl[sel] - cgz[sel]) / (1 + cgz[sel])
    b = binned(cgz[sel], dz, ZEDGES, min_n=20)
    b["err"] = binned(cgz[sel], zle[sel] / (1 + cgz[sel]), ZEDGES, min_n=20)["mean"]
    out4 = np.abs(zl[sel] - cgz[sel]) > 4 * zle[sel]
    b["out4"] = binned(cgz[sel], out4.astype(float), ZEDGES, min_n=20)["mean"]
    return b


def zl_panel(ax_top, ax_bot, sel, title):
    from matplotlib.colors import LogNorm

    ax_top.hist2d(zl[sel], cgz[sel], bins=[np.linspace(0.05, 0.9, 171)] * 2, cmap=RAMP, norm=LogNorm(vmin=1),
                  rasterized=True)
    out4 = sel & (np.abs(zl - cgz) > 4 * zle)
    ax_top.plot(zl[out4], cgz[out4], ".", ms=1.5, color=ORANGE, rasterized=True,
                label=f"4σ outliers ({out4.sum() / sel.sum():.1%})")
    ax_top.plot([0, 1], [0, 1], color=INK2, lw=0.6)
    ax_top.set_xlim(0.05, 0.9)
    ax_top.set_ylim(0.05, 0.9)
    ax_top.set_xlabel("z_λ")
    ax_top.set_ylabel("z_spec of the central galaxy")
    ax_top.legend(loc="upper left")
    panel_label(ax_top, "")
    ax_top.set_title(title, loc="left")
    for part, ls, lab in ((~calib, "-", "independent"), (calib, (0, (4, 2)), "calibration area")):
        b = zstats(sel & part)
        ax_bot.plot(b["x"], b["med"], color=BLUE, ls=ls, label=f"median Δz/(1+z), {lab}")
        ax_bot.plot(b["x"], b["nmad"], color=ORANGE, ls=ls, label=f"NMAD, {lab}")
        ax_bot.plot(b["x"], b["err"], color=AQUA, ls=ls, label=f"mean σ_zλ/(1+z), {lab}")
    ax_bot.axhline(0, color=INK2, lw=0.6)
    ax_bot.axhline(LIT["R14_sigz"], color=MUTED, lw=0.8, ls=":")
    ax_bot.set_ylim(-0.02, 0.04)
    ax_bot.set_xlim(0.05, 0.9)
    ax_bot.set_xlabel("z_spec")
    ax_bot.set_ylabel("Δz/(1+z)")


fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True,
                         gridspec_kw={"height_ratios": [2.2, 1]})
zl_panel(axes[0, 0], axes[1, 0], base, f"λ ≥ 20 with a central spec-z ({base.sum():,})")
zl_panel(axes[0, 1], axes[1, 1], clean, f"… and ≥ 2 spectroscopic members ({clean.sum():,})")
axes[1, 0].legend(ncol=2, fontsize=7, loc="upper left")
save(fig, "zl_vs_zspec")

for name, sel in (("all", base & ~calib), ("clean", clean & ~calib), ("calib", base & calib)):
    low = sel & (cgz < 0.6)
    dz = (zl[low] - cgz[low]) / (1 + cgz[low])
    o4 = np.mean(np.abs(zl[sel] - cgz[sel]) > 4 * zle[sel])
    print(f"{name:6s}: z < 0.6 bias {np.median(dz):+.4f}, NMAD {nmad(dz):.4f}, "
          f"mean sigma_zl/(1+z) {np.mean(zle[low] / (1 + cgz[low])):.4f}; 4 sigma outliers {o4:.2%} "
          f"({sel.sum():,} clusters)")
    record(f"zl_{name}_bias", np.median(dz))
    record(f"zl_{name}_nmad", nmad(dz))
    record(f"zl_{name}_err", np.mean(zle[low] / (1 + cgz[low])))
    record(f"zl_{name}_out4", o4)
    record(f"zl_{name}_n", int(sel.sum()))

# %% [markdown]
# ### Outlier fractions (Rykoff et al. 2014 Fig. 11)
#
# Fraction of clusters (λ ≥ 20, central spectroscopic redshift) with |z_λ − z_spec| larger
# than 3, 4 or 5 σ_zλ, in windows of z_λ ± 0.025, outside the calibration area. A Gaussian
# would give 0.27%, 0.006% and 0.00006%. The clean sample (≥ 2 spectroscopic members near
# the central) removes most of the outliers: they are mostly miscentred clusters or
# projections, as in Rykoff et al. (2014).

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
zc = np.arange(0.075, 0.826, 0.025)
for sel, ls, lab in ((base & ~calib, "-", "λ ≥ 20"), (clean & ~calib, (0, (4, 2)), "clean")):
    for k, (n, col) in enumerate(((3, BLUE), (4, ORANGE), (5, AQUA))):
        frac = []
        for z0 in zc:
            w = sel & (np.abs(zl - z0) < 0.025)
            frac.append(np.mean(np.abs(zl[w] - cgz[w]) > n * zle[w]) if w.sum() > 30 else np.nan)
        frac = np.where(np.asarray(frac) > 0, frac, np.nan)          # no outlier: off the log axis
        ax.plot(zc, frac, color=col, ls=ls, label=f"> {n}σ, {lab}")
ax.axhline(0.0027, color=MUTED, lw=0.8, ls=":", label="Gaussian, > 3σ")
ax.set_yscale("log")
ax.set_ylim(1e-3, 0.5)
ax.set_xlabel("z_λ")
ax.set_ylabel("outlier fraction")
ax.legend(ncol=2)
save(fig, "zl_outliers")

# %% [markdown]
# ### Redshift distribution (Rykoff et al. 2014 Fig. 12)
#
# For the clusters with λ ≥ 20 and a central spectroscopic redshift: the histogram of z_spec,
# of z_λ, and the sum of the P(z) of each cluster (with its Poisson ±1σ band). If P(z) is a
# fair description of the errors, the sum follows the z_spec histogram.

# %%
pzb, pz = np.asarray(cl["PZBINS"]), np.asarray(cl["PZ"])
sel = np.flatnonzero(base)
edges = np.arange(0.05, 0.9001, 0.01)
fine = np.arange(0.0, 1.3, 0.001)                 # P(z) is normalised over its whole range
ptot = np.zeros(fine.size)
for i in sel:
    p = np.interp(fine, pzb[i], pz[i], left=0.0, right=0.0)
    s = p.sum()
    if s > 0:
        ptot += p / s
summed = np.histogram(fine, edges, weights=ptot)[0]
fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
centres = 0.5 * (edges[1:] + edges[:-1])
ax.fill_between(centres, summed - np.sqrt(summed), summed + np.sqrt(summed), color=YELLOW, alpha=0.35, lw=0,
                step="mid", label="Σ P(z) ± 1σ")
ax.step(centres, summed, where="mid", color=YELLOW, lw=1.2)
ax.step(centres, np.histogram(cgz[sel], edges)[0], where="mid", color=INK, lw=1.4, label="z_spec (central)")
ax.step(centres, np.histogram(zl[sel], edges)[0], where="mid", color=BLUE, lw=1.0, ls=(0, (4, 2)), label="z_λ")
ax.set_xlabel("redshift")
ax.set_ylabel("clusters per Δz = 0.01")
ax.legend()
save(fig, "zl_nz")
h_spec = np.histogram(cgz[sel], edges)[0]
ok = summed > 5
chi2 = np.sum((h_spec[ok] - summed[ok]) ** 2 / summed[ok])
print(f"sum P(z) vs N(z_spec): chi2 = {chi2:.0f} for {ok.sum()} bins")
record("pz_chi2", [round(float(chi2), 1), int(ok.sum())])

# %% [markdown]
# ### Bias and uncertainty against spectroscopic cluster redshifts (Kluge et al. 2024 Figs. 16-17)
#
# Clusters with a velocity-clipped spectroscopic redshift (`SPEC_Z_BOOT`) from at least 10
# members at z < 0.3, 7 at 0.3–0.6 and 3 above, as in Kluge et al. (2024), λ ≥ 10, outside the
# calibration area. Left: running median and 16th–84th percentiles of z_λ − z_spec, with the
# bias and the empirical uncertainty of Kluge et al. in their four redshift ranges (boxes).
# Right: the mean formal error Z_LAMBDA_E against the empirical uncertainty (half the 16–84
# range) and the absolute bias.

# %%
zsb, nmem = np.asarray(cl["SPEC_Z_BOOT"]), np.asarray(cl["N_MEMBERS"])
need = np.where(zsb < 0.3, 10, np.where(zsb < 0.6, 7, 3))
ksel = (lam >= 10) & np.isfinite(zsb) & (zsb > 0.03) & (nmem >= need) & ~calib & (zl > 0.05)
d = zl[ksel] - zsb[ksel]
edges = np.arange(0.05, 0.901, 0.025)
b = binned(zsb[ksel], d, edges, min_n=15)
e = binned(zsb[ksel], zle[ksel], edges, min_n=15)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
ax = axes[0]
ax.plot(zsb[ksel], d, ".", ms=1, color=MUTED, alpha=0.4, rasterized=True)
ax.fill_between(b["x"], b["p16"], b["p84"], color=BLUE, alpha=0.25, lw=0, label="rema 16–84%")
ax.plot(b["x"], b["med"], color=BLUE, label="rema median")
for (z0, z1), bias, hi, lo in zip(LIT["K24_zbins"], LIT["K24_bias"], LIT["K24_dz_hi"], LIT["K24_dz_lo"]):
    ax.add_patch(plt.Rectangle((z0, bias + lo), z1 - z0, hi - lo, fill=False, ec=ORANGE, lw=1.0))
    ax.plot([z0, z1], [bias, bias], color=ORANGE, lw=1.2)
ax.plot([], [], color=ORANGE, label="Kluge+24 bias and ±δz")
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xlim(0.05, 0.9)
ax.set_ylim(-0.06, 0.06)
ax.set_xlabel("z_spec (members)")
ax.set_ylabel("z_λ − z_spec")
ax.legend(loc="lower left")
panel_label(ax, f"{ksel.sum():,} clusters")
ax = axes[1]
emp = 0.5 * (b["p84"] - b["p16"])
ax.plot(e["x"], e["mean"], "o", ms=3, color=AQUA, label="mean formal σ_zλ")
ax.plot(b["x"], emp, color=BLUE, ls=(0, (4, 2)), label="empirical (16–84%)/2")
ax.plot(b["x"], np.abs(b["med"]), color=ORANGE, label="|bias|")
ax.set_xlim(0.05, 0.9)
ax.set_ylim(0, 0.04)
ax.set_xlabel("z_spec (members)")
ax.set_ylabel("Δz")
ax.legend(loc="upper left")
save(fig, "zl_bias_error")
ratio = np.nanmedian(emp / np.interp(b["x"], e["x"], e["mean"]))
print(f"Kluge-like sample: {ksel.sum():,} clusters; empirical / formal error = {ratio:.2f} (median over z)")
record("zl_k24_n", int(ksel.sum()))
record("zl_k24_emp_over_formal", ratio)
for (z0, z1) in LIT["K24_zbins"]:
    w = (zsb[ksel] >= z0) & (zsb[ksel] < z1)
    if w.sum() > 20:
        record(f"zl_k24_bias_{z0}_{z1}", np.median(d[w]))
        record(f"zl_k24_dz_{z0}_{z1}", [float(np.percentile(d[w], 84) - np.median(d[w])),
                                         float(np.percentile(d[w], 16) - np.median(d[w]))])

# %% [markdown]
# ### Member zred (Rykoff et al. 2014 Fig. 7)
#
# zred of the members with P > 0.9 against the spectroscopic redshift of their cluster
# (central galaxy), outside the calibration area: the zred are those of the catalogue, after
# the zred correction ("afterburner") of the calibration. Bottom: mean offset, rms, mean
# zred error and the fraction of 4σ outliers per bin.

# %%
mem = load("members", ["CL_ROW", "P", "ZRED", "ZRED_E", "CENT_RANK"])
mrow = np.asarray(mem["CL_ROW"])
s = (np.asarray(mem["P"]) > 0.9) & (cgz[mrow] > 0.03) & ~calib[mrow]
zr, zre, zc = np.asarray(mem["ZRED"])[s], np.asarray(mem["ZRED_E"])[s], cgz[mrow[s]]
from matplotlib.colors import LogNorm  # noqa: E402

fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.5), constrained_layout=True, sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
ax = axes[0]
ax.hist2d(zc, zr, bins=[np.linspace(0.05, 0.9, 171), np.linspace(0.0, 1.0, 201)], cmap=RAMP,
          norm=LogNorm(vmin=1), rasterized=True)
ax.plot([0, 1], [0, 1], color=INK2, lw=0.6)
ax.set_ylabel("zred of members (P > 0.9)")
panel_label(ax, f"{zr.size:,} members")
ax = axes[1]
dz = zr - zc
b = binned(zc, dz, ZEDGES, min_n=50)
ax.plot(b["x"], b["mean"], color=BLUE, label="mean offset")
ax.plot(b["x"], b["std"], color=ORANGE, label="rms")
ax.plot(b["x"], binned(zc, zre, ZEDGES, min_n=50)["mean"], color=AQUA, label="mean zred error")
ax.plot(b["x"], binned(zc, (np.abs(dz) > 4 * zre).astype(float), ZEDGES, min_n=50)["mean"], color=MAGENTA,
        label="4σ outlier fraction")
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xlim(0.05, 0.9)
ax.set_ylim(-0.03, 0.12)
ax.set_xlabel("z_spec of the central galaxy")
ax.legend(ncol=2)
save(fig, "zred_members")
