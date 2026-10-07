"""Richness figures (Rykoff et al. 2014 Figs. 19, 27; Kluge et al. 2024 Figs. 11, 18, B.2).

    python docs/figures/redmapper_dr11/fig_richness.py
"""

# %% [script-only]
from common import *  # noqa: F403

# %% [markdown]
# ## Richness
#
# λ is the sum of the membership probabilities of the galaxies brighter than 0.2 L*
# (m*(z) + 1.75), corrected for the masked and too shallow part of the cluster aperture:
# SCALEVAL = 1/(completeness) multiplies the observed sum, and MASKFRAC is the masked fraction
# of the aperture (clusters with MASKFRAC > 0.2 are not kept). z_vlim is the redshift where the
# local 10σ z-band depth reaches m* + 1.75: below it the catalogue is volume limited and
# SCALEVAL ≈ 1.

# %%
cl = load("clusters")
lam, zl = np.asarray(cl["LAMBDA"]), np.asarray(cl["Z_LAMBDA"])
scale, mfrac, zv = np.asarray(cl["SCALEVAL"]), np.asarray(cl["MASKFRAC"]), np.asarray(cl["ZVLIM"])
mp = load("maps", ["AREA", "ZVLIM"])
area_tot = float(np.sum(mp["AREA"]))
print(f"{lam.size:,} clusters (lambda >= 3) over {area_tot:.0f} deg2; median z_vlim {np.nanmedian(zv):.3f}")

# %% [markdown]
# ### λ against z_λ, SCALEVAL and MASKFRAC (Rykoff et al. 2014 Figs. 19 and 27; Kluge et al. 2024 Fig. B.2)

# %%
from matplotlib.colors import LogNorm  # noqa: E402

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
ax = axes[0]
h = ax.hist2d(zl, np.log10(lam), bins=[np.linspace(0.05, 0.95, 181), np.linspace(np.log10(3), 2.5, 120)],
              cmap=RAMP, norm=LogNorm(vmin=1), rasterized=True)
ax.axhline(np.log10(20), color=ORANGE, lw=1.0, label="λ = 20")
ax.axvline(np.nanmedian(zv), color=INK2, lw=0.8, ls=(0, (4, 2)), label="median z_vlim")
ax.set_xlabel("z_λ")
ax.set_ylabel("log10 λ")
ax.legend(loc="upper right")
fig.colorbar(h[3], ax=ax, label="clusters")
ax = axes[1]
x = zl - zv
b = binned(x, np.log10(scale), np.arange(-0.6, 0.31, 0.02), min_n=50)
ax.hist2d(x[np.isfinite(x)], np.log10(scale[np.isfinite(x)]), bins=[np.linspace(-0.6, 0.3, 91), np.linspace(-0.02, 0.6, 63)],
          cmap=RAMP, norm=LogNorm(vmin=1), rasterized=True)
ax.plot(b["x"], b["med"], color=ORANGE, label="median")
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("z_λ − z_vlim")
ax.set_ylabel("log10 SCALEVAL")
ax.legend(loc="upper left")
ax = axes[2]
for lo, col in ((3, BLUE), (20, ORANGE)):
    ax.hist(mfrac[lam >= lo], bins=np.linspace(0, 0.2, 41), histtype="step", color=col, lw=1.4, density=True,
            label=f"λ ≥ {lo}")
ax.set_yscale("log")
ax.set_xlabel("MASKFRAC")
ax.set_ylabel("probability density")
ax.legend()
save(fig, "richness_z")
record("frac_scaleval_gt_1p1_below_zvlim", np.mean(scale[(zl < zv) & (lam >= 20)] > 1.1))
record("median_zvlim", np.nanmedian(zv))

# %% [markdown]
# ### Richness function (cumulative counts per deg²)
#
# N(> λ) per deg² in four bins of z_λ, in the part of the sky where z_vlim is above the bin
# (the volume-limited area, given in the legend). SDSS DR8
# (Rykoff et al. 2016, 10,134 deg², λ ≥ 20) and DES Y1 (McClintock et al. 2019, 1,437 deg²,
# λ ≥ 20) are drawn for the bins they cover; rema λ is not normalised to their filter sets.

# %%
ZB = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]
grid = np.logspace(np.log10(5), np.log10(300), 60)
sdss, desy1 = external("sdss_dr8"), external("des_y1")
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True, sharey=True)
for ax, (z0, z1) in zip(axes, ZB):
    a = float(np.sum(np.asarray(mp["AREA"])[np.asarray(mp["ZVLIM"]) >= z1]))
    s = (zl >= z0) & (zl < z1) & (zv >= z1)                     # same sky as the area a
    if a > 0 and s.sum():
        n = np.array([(lam[s] >= g).sum() for g in grid]) / a
        ax.plot(grid, n, color=BLUE, label=f"rema DR11 ({a:,.0f} deg²)")
        record(f"n_lam20_z{z0}_{z1}", float((lam[s] >= 20).sum() / a))
    for ref, area, col, lab in ((sdss, LIT["R16_dr8_area"], ORANGE, "SDSS DR8"), (desy1, LIT["DESY1_area"], AQUA, "DES Y1")):
        if ref is None:
            continue
        rs = (ref["Z"] >= z0) & (ref["Z"] < z1)
        if lab == "SDSS DR8" and z1 > 0.35:
            continue                                     # SDSS DR8 is volume limited to z ~ 0.35
        if rs.sum() > 20:
            g = grid[grid >= 20]
            ax.plot(g, np.array([(ref["LAMBDA"][rs] >= x).sum() for x in g]) / area, color=col, ls=(0, (4, 2)),
                    label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("λ")
    panel_label(ax, f"{z0} < z_λ < {z1}")
    ax.legend(loc="lower left")
axes[0].set_ylabel("N(> λ) per deg²")
save(fig, "richness_function")

# %% [markdown]
# ### Richness against other redMaPPer catalogues (Kluge et al. 2024 Fig. 11)
#
# One-to-one matches within 1.5′ and |Δz|/(1+z) < 0.02, with MASKFRAC < 0.1 in rema: DES Y1
# redMaPPer (deeper DES photometry, griz), SDSS DR8 redMaPPer v6.3 (shallower, ugriz) and the
# eROMaPPer λ_norm of Kluge et al. (2024) at the eRASS1 cluster positions (LS DR10, run at
# the X-ray positions, normalised to grz). Bottom: the ratio against redshift (running median
# and 16–84%); dotted: the ratio expected from Ider Chitham et al. (2020) for SDSS and 0.79 of
# Kluge et al. (2024) for DES Y1.

# %%
refs = [("des_y1", "DES Y1 redMaPPer", desy1), ("sdss_dr8", "SDSS DR8 redMaPPer", sdss),
        ("erass1", "Kluge+24 eRASS1 λ_norm", external("erass1"))]
refs = [r for r in refs if r[2] is not None]
fig, axes = plt.subplots(2, len(refs), figsize=(4.6 * len(refs), 7.5), constrained_layout=True, squeeze=False,
                         gridspec_kw={"height_ratios": [1.6, 1]})
ok_cl = mfrac < 0.1
for k, (key, lab, ref) in enumerate(refs):
    zr = ref["Z_LAMBDA"] if "Z_LAMBDA" in ref else ref["Z"]
    rra, rdec = (ref["RA_OPT"], ref["DEC_OPT"]) if "RA_OPT" in ref else (ref["RA"], ref["DEC"])
    good = np.isfinite(zr) & np.isfinite(ref["LAMBDA"]) & (ref["LAMBDA"] > 0)
    idx_ref = np.flatnonzero(good)
    sub = np.flatnonzero(ok_cl & (lam >= 5))
    i1, i2 = match_within(np.asarray(cl["RA"])[sub], np.asarray(cl["DEC"])[sub], zl[sub],
                          rra[idx_ref], rdec[idx_ref], zr[idx_ref], 1.5, dz_max=0.02, rank2=ref["LAMBDA"][idx_ref])
    a, r = sub[i1], idx_ref[i2]
    la, lr, zz = lam[a], ref["LAMBDA"][r], zl[a]
    ax = axes[0, k]
    sc = ax.scatter(lr, la, s=3, c=zz, cmap=RAMP, vmin=0.05, vmax=0.8, rasterized=True, lw=0)
    ax.plot([5, 400], [5, 400], color=INK2, lw=0.8, label="1:1")
    big = lr >= 20
    ratio = np.exp(np.median(np.log(la[big] / lr[big]))) if big.sum() > 10 else np.nan
    ax.plot([5, 400], [5 * ratio, 400 * ratio], color=ORANGE, lw=1.2, label=f"median ratio {ratio:.2f} (λ_ref ≥ 20)")
    if key == "des_y1":
        ax.plot([5, 400], [5 * LIT["K24_lam_ratio_desy1"], 400 * LIT["K24_lam_ratio_desy1"]], color=INK2, lw=0.8,
                ls=":", label="Kluge+24: 0.79")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5, 400)
    ax.set_ylim(3, 400)
    ax.set_xlabel(f"λ, {lab}")
    ax.set_ylabel("λ, rema DR11")
    ax.legend(loc="upper left")
    panel_label(ax, "")
    ax.set_title(f"{a.size:,} matches", loc="right")
    record(f"lam_ratio_{key}", ratio)
    record(f"lam_ratio_{key}_n", int(big.sum()))
    ax = axes[1, k]
    lrat = np.log(la / lr)
    b = binned(zz[big], lrat[big], np.arange(0.05, 0.91, 0.05), min_n=15)
    ax.plot(zz[big], np.exp(lrat[big]), ".", ms=2, color=MUTED, alpha=0.5, rasterized=True)
    ax.fill_between(b["x"], np.exp(b["p16"]), np.exp(b["p84"]), color=BLUE, alpha=0.25, lw=0)
    ax.plot(b["x"], np.exp(b["med"]), color=BLUE, label="median, λ_ref ≥ 20")
    if key == "sdss_dr8":
        zg = np.linspace(0.05, 0.6, 50)
        ax.plot(zg, ider_chitham_lambda(1.0, zg), color=ORANGE, ls=":", label="Ider Chitham+20 (LS/SDSS)")
    if key == "des_y1":
        ax.axhline(LIT["K24_lam_ratio_desy1"], color=ORANGE, ls=":", label="Kluge+24 (LS DR10/DES Y1)")
    ax.axhline(1, color=INK2, lw=0.6)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 3)
    ax.set_xlim(0.05, 0.9)
    ax.set_xlabel("z_λ")
    ax.set_ylabel("λ_rema / λ_ref")
    ax.legend(loc="upper left")
fig.colorbar(sc, ax=axes[0, :], label="z_λ", shrink=0.8)
save(fig, "richness_external")

# %% [markdown]
# ### Richness against velocity dispersion (Kluge et al. 2024 Fig. 18)
#
# Clusters with at least 15 spectroscopic members after the velocity clipping and an
# unflagged dispersion (bootstrap mean). The line is an orthogonal-distance (major-axis) fit of
# log λ = a log σ_v + b; dotted: the relation of Kluge et al. (2024) for λ_norm (eRASS1),
# multiplied by 1.039, their griz-to-grz normalisation.

# %%
def orthogonal_fit(x, y):
    """Slope and intercept of the orthogonal-distance (major-axis) line through (x, y)."""
    sxx, syy, sxy = np.var(x), np.var(y), np.cov(x, y, bias=True)[0, 1]
    slope = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy**2)) / (2 * sxy)
    return slope, np.mean(y) - slope * np.mean(x)


nm, vd, vflag = np.asarray(cl["N_MEMBERS"]), np.asarray(cl["VDISP_BOOT"]), np.asarray(cl["VDISP_FLAG"])
s = (nm >= 15) & np.isfinite(vd) & (vd > 50) & ~vflag.astype(bool)
x, y = np.log10(vd[s]), np.log10(lam[s])
a_, b_ = orthogonal_fit(x, y)
resid = y - (a_ * x + b_)
fig, ax = plt.subplots(figsize=(6, 4.6), constrained_layout=True)
sc = ax.scatter(vd[s], lam[s], s=4, c=zl[s], cmap=RAMP, vmin=0.05, vmax=0.6, lw=0, rasterized=True)
fig.colorbar(sc, ax=ax, label="z_λ", shrink=0.8)
xs = np.linspace(150, 1600, 50)
ax.plot(xs, 10 ** (a_ * np.log10(xs) + b_), color=ORANGE, lw=1.4,
        label=f"rema: log λ = {a_:.2f} log σ_v {b_:+.2f}, scatter {np.std(resid):.2f} dex")
ka, kb = LIT["K24_lam_sig"]
ax.plot(xs, LIT["K24_snorm_griz"] * 10 ** (ka * np.log10(xs) + kb), color=INK2, ls=":", label="Kluge+24 (× 1.039)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("σ_v [km/s]")
ax.set_ylabel("λ")
ax.legend(loc="upper left")
panel_label(ax, "")
ax.set_title(f"{s.sum():,} clusters with ≥ 15 spectroscopic members", loc="right")
save(fig, "richness_vdisp")
record("lam_sigma_fit", [round(float(a_), 3), round(float(b_), 3), round(float(np.std(resid)), 3), int(s.sum())])
