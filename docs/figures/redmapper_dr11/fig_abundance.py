"""Depth, volume limit, abundance and sky-distribution figures (Rykoff et al. 2014 Figs. 17-18;
Rykoff et al. 2016 Figs. 1-3, 6; Kluge et al. 2024 Figs. 2-3, 12-13, B.1, B.3; Baxter et al. 2016
Fig. 2).

    python docs/figures/redmapper_dr11/fig_abundance.py
"""

# %% [script-only]
from common import *  # noqa: F403

# %% [markdown]
# ## Depth, volume limit and abundance
#
# The maps come from the DR11 randoms (one file, 2500 per deg²) cut like the run: mask bits,
# at least one exposure in g, r, i and z, E(B−V) < 0.2, |b| ≥ 15° and the own boxes of the
# regions that finished.

# %%
mp = load("maps")
cl = load("clusters")
M = meta()
NSIDE = M["nside"]
pix = np.asarray(mp["PIX"])
area = np.asarray(mp["AREA"])
zvmap = np.asarray(mp["ZVLIM"])
lam, zl, zv = np.asarray(cl["LAMBDA"]), np.asarray(cl["Z_LAMBDA"]), np.asarray(cl["ZVLIM"])
print(f"run area {area.sum():,.0f} deg2 in {pix.size:,} pixels of NSIDE {NSIDE}: "
      + ", ".join(f"{k} {v:,.0f} deg2" for k, v in M["area"].items()))
record("area_total", float(area.sum()))

# %% [markdown]
# ### 10σ depth maps (Rykoff et al. 2016 Fig. 1; Kluge et al. 2024 Figs. 2 and B.1)
#
# Mean 10σ galaxy (`GALDEPTH`) depth per pixel, corrected for Galactic extinction, in the four
# bands of the run.

# %%
fig = plt.figure(figsize=(13, 7.2), constrained_layout=True)
for k, b in enumerate(["G", "R", "I", "Z"]):
    v = np.asarray(mp[f"DEPTH_{b}10"])
    lo, hi = np.nanpercentile(v, [2, 98])
    ax = sky_axes(fig, 221 + k)
    im = sky_image(ax, NSIDE, pix, v, vmin=lo, vmax=hi)
    fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.6, pad=0.04, label=f"{b.lower()}-band 10σ depth [mag]")
    print(f"{b.lower()}: median 10 sigma depth {np.median(v):.2f} (2-98%: {lo:.2f}-{hi:.2f})")
    record(f"depth_{b.lower()}10", float(np.median(v)))
save(fig, "depth_maps")

# %% [markdown]
# ### z_vlim (Kluge et al. 2024 Figs. 3 and B.3)
#
# Left: the z_vlim map. Middle: m*(z) + 1.75 in the z band, which turns the 10σ depth into
# z_vlim, and the area per z_vlim. Right: the z_vlim of this map at the positions of the eRASS1
# clusters against the `ZVLIM_02` of Kluge et al. (2024), computed from the LS DR10 depth.

# %%
from rema.calibration import Calibration  # noqa: E402
from rema.model.profiles import MStar, maxmag_from_mstar  # noqa: E402

cfg = Calibration.read(CALIB).config
ms = MStar(cfg.model.mstar)
fig = plt.figure(figsize=(15, 4.2), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1])
ax = sky_axes(fig, gs[0])
im = sky_image(ax, NSIDE, pix, zvmap, vmin=0.5, vmax=0.95)
fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.6, pad=0.04, label="z_vlim (0.2 L* at 10σ)")
ax = fig.add_subplot(gs[1])
zg = np.asarray(ms.z)
mlim = np.asarray(maxmag_from_mstar(np.asarray(ms.m), cfg.model.lval_reference))
ax.plot(zg, mlim, color=BLUE, label="m*(z) + 1.75 (z band)")
ax.set_xlim(0.3, 1.2)
ax.set_ylim(19.5, 23.0)
ax.set_xlabel("z_vlim")
ax.set_ylabel("10σ z-band depth [mag]")
ax2 = ax.twinx()
ax2.hist(zvmap, bins=np.arange(0.3, 1.2, 0.01), weights=area, color=ORANGE, alpha=0.45)
ax2.set_ylabel("area per Δz_vlim = 0.01 [deg²]", color=INK2)
ax2.grid(False)
ax2.spines["right"].set_visible(True)
ax.legend(loc="upper left")
ax = fig.add_subplot(gs[2])
er = external("erass1")
if er is not None:
    i = hp.ang2pix(NSIDE, er["RA"], er["DEC"], lonlat=True, nest=True)
    j = np.searchsorted(pix, i).clip(0, pix.size - 1)
    found = (pix[j] == i) & np.isfinite(er["ZVLIM_02"]) & (er["ZVLIM_02"] > 0)
    ax.plot(er["ZVLIM_02"][found], zvmap[j[found]], ".", ms=2, color=BLUE, alpha=0.5, rasterized=True)
    ax.plot([0.3, 1.3], [0.3, 1.3], color=INK2, lw=0.8)
    dzv = zvmap[j[found]] - er["ZVLIM_02"][found]
    panel_label(ax, f"{found.sum():,} eRASS1 clusters: median Δ = {np.median(dzv):+.3f}")
    record("zvlim_minus_k24", float(np.median(dzv)))
    ax.set_xlim(0.4, 1.2)
    ax.set_ylim(0.4, 1.2)
ax.set_xlabel("ZVLIM_02, Kluge+24 (LS DR10)")
ax.set_ylabel("z_vlim, this map (DR11)")
save(fig, "zvlim")

# %% [markdown]
# ### Effective area against z_vlim (Kluge et al. 2024 Fig. 3)
#
# Area of the run where z_vlim exceeds a given redshift, in total and in the half of the sky
# covered by eRASS1 (Galactic longitude > 180°).

# %%
lon = hp.pix2ang(NSIDE, pix, nest=True, lonlat=True)
gl = SkyCoord(lon[0] * u.deg, lon[1] * u.deg).galactic.l.deg
zgrid = np.linspace(0.3, 1.2, 181)
fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
for sel, col, lab in ((np.ones(pix.size, bool), BLUE, "run"), (gl > 180, ORANGE, "run ∩ eRASS1-DE (l > 180°)")):
    ax.plot(zgrid, [area[sel & (zvmap >= z)].sum() for z in zgrid], color=col, label=lab)
ax.set_xlabel("z_vlim")
ax.set_ylabel("area with z_vlim above [deg²]")
ax.legend()
save(fig, "area_zvlim")
for z in (0.6, 0.7, 0.8):
    record(f"area_zvlim_{z}", float(area[zvmap >= z].sum()))

# %% [markdown]
# ### Cluster density against redshift (Rykoff et al. 2014 Fig. 18; Rykoff et al. 2016 Fig. 6; Kluge et al. 2024 Fig. 12)
#
# Clusters in the sky where z_vlim is above the upper edge of their z_λ bin, divided by the area
# of that sky. Left: per deg² and per Δz = 0.05. Right: comoving density, with the abundance of halos
# above M500c = 0.7, 1.0 and 1.3 × 10¹⁴ M☉ (Tinker et al. 2008 mass function, colossus,
# Ωm = 0.3, h = 0.7, σ8 = 0.8), the levels of SDSS DR8 (Rykoff et al. 2014, z < 0.35) and the
# densities of SDSS DR8 v6.3 and DES Y1 redMaPPer (λ ≥ 20) with their published areas.

# %%
from colossus.cosmology import cosmology  # noqa: E402
from colossus.lss import mass_function  # noqa: E402

cosmology.setCosmology("rema", {"flat": True, "H0": 70.0, "Om0": 0.3, "Ob0": 0.045, "sigma8": 0.8, "ns": 0.96})
DZ = 0.05
zedges = np.arange(0.05, 0.951, DZ)
zc = 0.5 * (zedges[1:] + zedges[:-1])
sr_per_deg2 = (np.pi / 180.0) ** 2
dvol = np.array([COSMO.comoving_volume(z1).value - COSMO.comoving_volume(z0).value
                 for z0, z1 in zip(zedges[:-1], zedges[1:])]) / (4 * np.pi) * sr_per_deg2 * H ** 3  # (Mpc/h)^3 per deg2
a_z = np.array([area[zvmap >= z1].sum() for z1 in zedges[1:]])


def densities(sel):
    """Clusters per deg2 and per (h^-1 Mpc)^3 in each z bin, in the sky where z_vlim is above the bin."""
    k = np.digitize(zl, zedges) - 1
    inside = (k >= 0) & (k < zc.size)
    deep = inside & (zv >= zedges[np.clip(k + 1, 0, zedges.size - 1)])
    n = np.bincount(k[sel & deep], minlength=zc.size)[:zc.size]
    with np.errstate(divide="ignore", invalid="ignore"):
        na = np.where(a_z > 100, n / a_z, np.nan)
    return na, na / dvol, np.where(a_z > 100, np.sqrt(n) / a_z, np.nan)


fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
for lmin, col in ((10, AQUA), (20, BLUE), (40, ORANGE)):
    na, nv, ena = densities(lam >= lmin)
    axes[0].errorbar(zc, na, yerr=ena, color=col, marker="o", ms=3, label=f"rema λ ≥ {lmin}")
    axes[1].errorbar(zc, nv, yerr=ena / dvol, color=col, marker="o", ms=3, label=f"rema λ ≥ {lmin}")
    axes[1].hlines(LIT["R14_nV"][lmin], 0.08, 0.35, color=col, lw=4, alpha=0.3)
    sel_mid = (zc > 0.15) & (zc < 0.55)
    record(f"nV_lam{lmin}", float(np.nanmedian(nv[sel_mid])))
    record(f"nA_lam{lmin}", float(np.sum((lam >= lmin) & (zl < zv) & (zl > 0.05)) / area.sum()))
axes[1].plot([], [], color=MUTED, lw=4, alpha=0.5, label="Rykoff+14, SDSS DR8 (z < 0.35)")
for ref_name, area_ref, z0, z1, mk in (("sdss_dr8", LIT["R16_dr8_area"], 0.08, 0.35, "s"),
                                      ("des_y1", LIT["DESY1_area"], 0.2, 0.65, "^")):
    ref = external(ref_name)
    if ref is None:
        continue
    s = (ref["LAMBDA"] >= 20) & (ref["Z"] >= z0) & (ref["Z"] < z1)
    n = np.histogram(ref["Z"][s], zedges)[0].astype(float)
    inside = (zc > z0) & (zc < z1)
    axes[1].plot(zc[inside], (n / area_ref / dvol)[inside], mk, ms=5, mfc="none", color=INK2,
                 label=f"{ref_name.replace('_', ' ').upper()} λ ≥ 20")
zf = np.linspace(0.05, 0.95, 37)
for m14, ls in ((0.7, ":"), (1.0, (0, (4, 2))), (1.3, "-.")):
    lnm = np.linspace(np.log(m14 * 1e14 * H), np.log(3e15), 200)       # Msun/h
    nh = [np.trapezoid(mass_function.massFunction(np.exp(lnm), z, mdef="500c", model="tinker08",
                                                  q_out="dndlnM"), lnm) for z in zf]
    axes[1].plot(zf, nh, color=INK2, lw=1.0, ls=ls, label=f"halos M500c > {m14} × 10¹⁴ M☉")
axes[0].set_ylabel("clusters per deg² per Δz = 0.05")
axes[1].set_ylabel("comoving density [h³ Mpc⁻³]")
for ax in axes:
    ax.set_yscale("log")
    ax.set_xlabel("z_λ")
    ax.set_xlim(0.05, 0.95)
axes[1].legend(fontsize=7, ncol=2, loc="lower left")
axes[0].legend(loc="lower left")
save(fig, "density_z")

# %% [markdown]
# ### Density against depth (Kluge et al. 2024 Fig. 13)
#
# Clusters per deg² in the plane of z_vlim (the local depth) and λ, in five bins of z_λ,
# divided at each λ by the density in the volume-limited sky (z_vlim above the bin). The
# vertical line marks z_vlim = z_λ: to its right the density should not depend on z_vlim (ratio
# 1); to its left λ is extrapolated with SCALEVAL, and clusters scatter across the λ threshold.

# %%
ZB = [(0.2, 0.3), (0.35, 0.45), (0.5, 0.6), (0.65, 0.75), (0.8, 0.9)]
zvb = np.arange(0.55, 0.951, 0.025)
lb = np.logspace(np.log10(5), np.log10(150), 16)
a_zv = np.histogram(zvmap, zvb, weights=area)[0]
fig, axes = plt.subplots(1, len(ZB), figsize=(16, 3.6), constrained_layout=True, sharey=True)
for ax, (z0, z1) in zip(axes, ZB):
    s = (zl >= z0) & (zl < z1)
    h = np.histogram2d(zv[s], lam[s], [zvb, lb])[0]
    vl = zvb[:-1] >= z1                                           # volume-limited columns
    ref = h[vl].sum(axis=0) / a_zv[vl].sum() if vl.any() else np.full(lb.size - 1, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((a_zv[:, None] > 50) & (h >= 3), h / a_zv[:, None] / ref[None, :], np.nan)
    im = ax.pcolormesh(zvb, lb, ratio.T, cmap=DIVERGE, vmin=0.5, vmax=1.5, rasterized=True)
    ax.axvline(0.5 * (z0 + z1), color=INK, lw=1.2)
    ax.set_xlim(zvb[0], zvb[-1])
    ax.set_yscale("log")
    ax.set_xlabel("z_vlim")
    ax.set_title(f"{z0} < z_λ < {z1}", loc="left")
axes[0].set_ylabel("λ")
fig.colorbar(im, ax=axes, label="density / volume-limited density", shrink=0.9)
save(fig, "density_depth")

# %% [markdown]
# ### Density-contrast maps (Rykoff et al. 2014 Fig. 17; Rykoff et al. 2016 Figs. 2-3)
#
# δ = n/⟨n⟩ − 1 of clusters with λ ≥ 5 (as in Rykoff et al. 2014) in three bins of z_λ, on
# NSIDE 32 pixels (3.4 deg²) with at least half of their area in the run and z_vlim above the
# bin. The titles compare the rms of δ with the Poisson expectation.

# %%
NS_LO = 32
down = (NSIDE // NS_LO) ** 2
lo_pix = pix // down
ZB = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7)]
fig = plt.figure(figsize=(15, 3.6), constrained_layout=True)
cl_lo = np.asarray(cl["PIX"]) // down
full_area = hp.nside2pixarea(NS_LO, degrees=True)
for k, (z0, z1) in enumerate(ZB):
    deep = zvmap >= z1
    a = np.bincount(lo_pix[deep], weights=area[deep], minlength=hp.nside2npix(NS_LO))
    s = (lam >= 5) & (zl >= z0) & (zl < z1) & (zv >= z1)
    n = np.bincount(cl_lo[s], minlength=hp.nside2npix(NS_LO)).astype(float)
    good = a > 0.5 * full_area
    mean = n[good].sum() / a[good].sum()
    delta = n[good] / a[good] / mean - 1
    poisson = np.sqrt(np.mean(1 / (mean * a[good])))
    ax = sky_axes(fig, 131 + k)
    im = sky_image(ax, NS_LO, np.flatnonzero(good), delta, cmap=DIVERGE, vmin=-0.5, vmax=0.5)
    ax.set_title(f"{z0} < z_λ < {z1}: {mean:.1f} per deg², rms δ {np.std(delta):.2f} (Poisson {poisson:.2f})",
                 fontsize=8)
    record(f"density_contrast_rms_z{z0}_{z1}", [float(np.std(delta)), float(poisson)])
fig.colorbar(im, ax=fig.axes, orientation="horizontal", shrink=0.4, label="δ = n/⟨n⟩ − 1 (λ ≥ 5)")
save(fig, "density_maps")

# %% [markdown]
# ### Density against observing conditions (Baxter et al. 2016 Fig. 2)
#
# Clusters with λ ≥ 20 and 0.1 < z_λ < 0.5 in pixels where z_vlim > 0.5 (so depth alone should
# not change their number), in ten quantiles of each pixel property, normalised by the mean
# density; the error bars are Poisson. The last panel is the density against the distance to
# the edge of the region (own box) that found the cluster, from the QA file of the merge: the
# regions are run separately and their clusters are merged by own box.

# %%
props = [("DEPTH_Z10", "z-band 10σ depth [mag]"), ("EBV", "E(B−V)"), ("PSFSIZE_Z", "z-band PSF FWHM [″]"),
         ("MASKED_FRAC", "masked fraction of the pixel")]
deep = zvmap > 0.5
s = (lam >= 20) & (zl > 0.1) & (zl < 0.5)
cpix = np.asarray(cl["PIX"])
j = np.searchsorted(pix, cpix[s]).clip(0, pix.size - 1)
ok = (pix[j] == cpix[s]) & deep[j]
ncl = np.bincount(j[ok], minlength=pix.size).astype(float)
mean = ncl[deep].sum() / area[deep].sum()
fig, axes = plt.subplots(1, len(props) + 1, figsize=(17, 3.4), constrained_layout=True, sharey=True)
for ax, (name, label) in zip(axes, props):
    v = np.asarray(mp[name])[deep]
    q = np.unique(np.nanpercentile(v, np.linspace(0, 100, 11)))
    k = np.digitize(v, q[1:-1])
    nb = np.bincount(k, weights=ncl[deep], minlength=q.size - 1)
    ab = np.bincount(k, weights=area[deep], minlength=q.size - 1)
    xb = np.array([np.median(v[k == i]) for i in range(q.size - 1)])
    ax.errorbar(xb, nb / ab / mean, yerr=np.sqrt(nb) / ab / mean, color=BLUE, marker="o", ms=3)
    ax.axhline(1, color=INK2, lw=0.6)
    ax.set_xlabel(label)
axes[0].set_ylabel("n / ⟨n⟩")
axes[0].set_ylim(0.7, 1.3)
ax = axes[-1]
for k, run in enumerate(RUNS):
    f = run / "clusters_dr11_qa.json"
    if f.exists():
        ep = json.loads(f.read_text())["edge_profile"]
        x = np.asarray(ep["d_lo"]) + 0.125
        ax.errorbar(x, ep["ratio"], yerr=np.asarray(ep["ratio"]) / np.sqrt(ep["n"]), color=SERIES[k], marker="o",
                    ms=3, label=run.name.replace("rema_dr11_v0.2.0_", ""))
ax.axhline(1, color=INK2, lw=0.6)
ax.set_xlabel("distance to the own-box edge [deg]")
ax.legend()
save(fig, "density_systematics")
