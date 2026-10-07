"""Comparison with external cluster catalogues (Kluge et al. 2024 Figs. 7-8, 10; Rozo & Rykoff 2014
Fig. 4; Rozo et al. 2015 Fig. 2; Saro et al. 2015; Bleem et al. 2020; Wen & Han 2024 Fig. 10;
Zou et al. 2021 Fig. 9).

    python docs/figures/redmapper_dr11/fig_external.py
"""

# %% [script-only]
from common import *  # noqa: F403

# %% [markdown]
# ## Comparison with SZ, X-ray and optical catalogues
#
# External clusters are counted when they fall in the run footprint (a map pixel at least
# half covered) at 0.05 < z < 0.9. A cluster is recovered when a rema cluster with λ ≥ 5 lies
# within 1 h⁻¹ Mpc (projected at the external redshift) with |z_λ − z|/(1+z) < 0.05; the
# richest such cluster is the counterpart.

# %%
cl = load("clusters")
mp = load("maps", ["PIX", "AREA"])
NSIDE = meta()["nside"]
lam, zl = np.asarray(cl["LAMBDA"]), np.asarray(cl["Z_LAMBDA"])
ra, dec = np.asarray(cl["RA"]), np.asarray(cl["DEC"])
pix, parea = np.asarray(mp["PIX"]), np.asarray(mp["AREA"])
full = hp.nside2pixarea(NSIDE, degrees=True)


def in_run(r, d):
    p = hp.ang2pix(NSIDE, r, d, lonlat=True, nest=True)
    j = np.searchsorted(pix, p).clip(0, pix.size - 1)
    return (pix[j] == p) & (parea[j] > 0.5 * full)


def counterparts(ref, lam_min=5.0, dz_max=0.05, rmax=1.0):
    """Indices (external, rema) of the recovered clusters and the mask of those considered."""
    zz = ref["Z"]
    consider = np.isfinite(zz) & (zz > 0.05) & (zz < 0.9) & in_run(ref["RA"], ref["DEC"])
    if "PCONT" in ref:
        consider &= ref["PCONT"] < 0.5
    ie = np.flatnonzero(consider)
    sub = np.flatnonzero(lam >= lam_min)
    a, b, _ = match_physical(ref["RA"][ie], ref["DEC"][ie], zz[ie], ra[sub], dec[sub], zl[sub], rmax=rmax,
                             dz_max=dz_max, rank2=lam[sub])
    return ie[a], sub[b], consider


CATS = [("act_dr6", "ACT DR6 (SZ)"), ("act_dr5", "ACT DR5 (SZ)"), ("spt_2500d", "SPT-SZ 2500d"),
        ("psz2", "Planck PSZ2"), ("erass1", "eRASS1 (PCONT < 0.5)"), ("mcxc", "MCXC (X-ray)")]
REFS = {k: external(k) for k, _ in CATS}
CATS = [(k, lab) for k, lab in CATS if REFS[k] is not None]

# %% [markdown]
# ### Recovery of SZ- and X-ray-selected clusters (Kluge et al. 2024 Fig. 10; Rozo & Rykoff 2014 Fig. 4)
#
# Top: redshift distribution of the external clusters in the footprint (lines) and of the
# recovered ones (filled). Bottom: the recovered fraction against redshift (left) and against
# the external mass M500 (right; each catalogue's own mass scale).

# %%
fig = plt.figure(figsize=(13, 7.5), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
axes = np.array([[fig.add_subplot(gs[0, :]), None], [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
zb = np.arange(0.05, 0.901, 0.05)
mb = np.logspace(np.log10(0.5), np.log10(20), 14)
for k, (name, lab) in enumerate(CATS):
    ref = REFS[name]
    ie, ir, consider = counterparts(ref)
    col = SERIES[k]
    rec = np.zeros(ref["RA"].size, bool)
    rec[ie] = True
    axes[0, 0].hist(ref["Z"][consider], zb, histtype="step", color=col, lw=1.2, label=f"{lab}: {consider.sum():,}")
    axes[0, 0].hist(ref["Z"][rec], zb, color=col, alpha=0.12)
    hz_all = np.histogram(ref["Z"][consider], zb)[0]
    hz_rec = np.histogram(ref["Z"][rec], zb)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(hz_all >= 5, hz_rec / hz_all, np.nan)
    axes[1, 0].plot(0.5 * (zb[1:] + zb[:-1]), f, color=col, marker="o", ms=3, label=lab)
    if "M500" in ref:
        m = ref["M500"]
        okm = consider & np.isfinite(m) & (m > 0)
        hm_all = np.histogram(m[okm], mb)[0]
        hm_rec = np.histogram(m[okm & rec], mb)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            axes[1, 1].plot(np.sqrt(mb[1:] * mb[:-1]), np.where(hm_all >= 5, hm_rec / hm_all, np.nan), color=col,
                            marker="o", ms=3, label=lab)
    frac = rec[consider].mean()
    print(f"{lab}: {consider.sum():,} in the footprint at 0.05 < z < 0.9, recovered {frac:.1%}")
    record(f"recovery_{name}", [float(frac), int(consider.sum())])
axes[0, 0].set_xlabel("external redshift")
axes[0, 0].set_ylabel("clusters per Δz = 0.05")
axes[0, 0].set_yscale("log")
axes[0, 0].legend(fontsize=7, ncol=2)
axes[1, 0].set_xlabel("external redshift")
axes[1, 0].set_ylabel("recovered fraction")
axes[1, 0].set_ylim(0, 1.05)
axes[1, 1].set_xscale("log")
plain_log_ticks(axes[1, 1])
axes[1, 1].set_xlabel("M500 of the external catalogue [10¹⁴ M☉]")
axes[1, 1].set_ylim(0, 1.05)
axes[1, 1].legend(fontsize=7, loc="lower right")
save(fig, "external_recovery")

# %% [markdown]
# ### Redshift consistency (Kluge et al. 2024 Fig. 7)
#
# The external clusters with a spectroscopic or literature redshift (ACT DR6, SPT, MCXC), paired
# by position only (1 h⁻¹ Mpc, the richest rema cluster with |Δz|/(1+z) < 0.3): fraction whose
# z_λ agrees within 0.02, 0.03, 0.05 or 0.10 (in |Δz|/(1+z)), against λ.

# %%
pairs = []
for name in ("act_dr6", "spt_2500d", "mcxc"):
    ref = REFS.get(name)
    if ref is None:
        continue
    ie, ir, _ = counterparts(ref, dz_max=0.3)
    pairs.append((ref["Z"][ie], ir))
zref = np.concatenate([p[0] for p in pairs])
irem = np.concatenate([p[1] for p in pairs])
dz = np.abs(zl[irem] - zref) / (1 + zref)
lb = np.logspace(np.log10(5), np.log10(200), 12)
fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
for k, tol in enumerate((0.02, 0.03, 0.05, 0.10)):
    b = binned(lam[irem], (dz < tol).astype(float), lb, min_n=10)
    ax.plot(b["x"], b["mean"], color=SERIES[k], marker="o", ms=3, label=f"|Δz|/(1+z) < {tol}")
    record(f"zcons_{tol}_lam20", float(np.mean(dz[lam[irem] >= 20] < tol)))
ax.set_xscale("log")
ax.set_xlabel("λ (rema)")
ax.set_ylabel("fraction with consistent redshift")
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right")
panel_label(ax, f"{irem.size:,} ACT DR6, SPT and MCXC clusters")
save(fig, "external_zconsistency")

# %% [markdown]
# ### Richness against mass proxies (Rozo et al. 2015 Fig. 2; Saro et al. 2015 Fig. 4; Bleem et al. 2020 Fig. 9)
#
# λ of the counterparts against the SZ masses of ACT DR6 and SPT (M500c), the eRASS1 masses
# (from the count rate, Bulbul et al. 2024) and the eRASS1 0.2–2.3 keV luminosity L500. The
# line is a least-squares fit of ln λ on ln M (or ln L) for clusters at 0.1 < z < 0.6; the
# scatter quoted is the rms of ln λ about it.

# %%
panels = [("act_dr6", "M500", "ACT DR6 M500c [10¹⁴ M☉]"), ("spt_2500d", "M500", "SPT M500 [10¹⁴ h70⁻¹ M☉]"),
          ("erass1", "M500", "eRASS1 M500 [10¹⁴ M☉]"), ("erass1", "LX", "eRASS1 L500 [10⁴⁴ erg/s]")]
panels = [p for p in panels if REFS.get(p[0]) is not None]
fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8), constrained_layout=True, squeeze=False)
for ax, (name, col, label) in zip(axes[0], panels):
    ref = REFS[name]
    ie, ir, _ = counterparts(ref)
    x = ref[col][ie]
    ok = np.isfinite(x) & (x > 0) & (ref["Z"][ie] > 0.1) & (ref["Z"][ie] < 0.6)
    lx, ly = np.log(x[ok]), np.log(lam[ir][ok])
    slope, icpt = np.polyfit(lx, ly, 1)
    rms = np.std(ly - (slope * lx + icpt))
    sc = ax.scatter(x[ok], lam[ir][ok], s=4, c=ref["Z"][ie][ok], cmap=RAMP, vmin=0.1, vmax=0.6, lw=0,
                    rasterized=True)
    xs = np.logspace(np.log10(np.percentile(x[ok], 1)), np.log10(np.percentile(x[ok], 99)), 20)
    ax.plot(xs, np.exp(icpt) * xs ** slope, color=ORANGE, lw=1.4,
            label=f"λ ∝ X^{slope:.2f}, σ_lnλ = {rms:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_ticks(ax, "x", subs=(1.0,) if col == "LX" else (1.0, 2.0, 5.0))
    plain_log_ticks(ax, "y")
    ax.set_xlabel(label)
    ax.set_ylabel("λ")
    ax.legend(loc="upper left")
    ax.set_title(f"{ok.sum():,} pairs", loc="right")
    record(f"lam_{name}_{col}", [round(float(slope), 3), round(float(rms), 3), int(ok.sum())])
fig.colorbar(sc, ax=axes[0, :], label="external redshift", shrink=0.8)
save(fig, "external_mass")

# %% [markdown]
# ### Wen & Han (2024) clusters (Wen & Han 2024 Fig. 10; Zou et al. 2021 Fig. 9)
#
# Wen & Han (2024) found 1.58 million clusters in the Legacy Surveys DR9 and DR10 from the
# stellar mass of galaxies in photometric-redshift slices. Left: fraction of rema clusters
# with a Wen & Han counterpart (1 h⁻¹ Mpc, |Δz|/(1+z) < 0.05) against λ, in bins of z_λ. Right:
# fraction of Wen & Han clusters in the footprint with a rema counterpart (λ ≥ 5) against their
# mass M500, in bins of redshift.

# %%
wh = external("wen_han_2024")
if wh is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    inside = in_run(wh["RA"], wh["DEC"]) & np.isfinite(wh["Z"]) & (wh["Z"] > 0.05) & (wh["Z"] < 0.9)
    iw = np.flatnonzero(inside)
    sub = np.flatnonzero(lam >= 5)
    a, b, _ = match_physical(ra[sub], dec[sub], zl[sub], wh["RA"][iw], wh["DEC"][iw], wh["Z"][iw], rmax=1.0,
                             dz_max=0.05, rank2=wh["M500"][iw])
    has_wh = np.zeros(lam.size, bool)
    has_wh[sub[a]] = True
    has_rm = np.zeros(wh["RA"].size, bool)
    has_rm[iw[b]] = True
    lb = np.logspace(np.log10(5), np.log10(200), 14)
    mb = np.logspace(np.log10(0.47), np.log10(10), 12)
    for k, (z0, z1) in enumerate(((0.05, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9))):
        s = (zl >= z0) & (zl < z1) & (lam >= 5)
        bb = binned(lam[s], has_wh[s].astype(float), lb, min_n=20)
        axes[0].plot(bb["x"], bb["mean"], color=SERIES[k], marker="o", ms=3, label=f"{z0} < z < {z1}")
        s = inside & (wh["Z"] >= z0) & (wh["Z"] < z1)
        bb = binned(wh["M500"][s], has_rm[s].astype(float), mb, min_n=20)
        axes[1].plot(bb["x"], bb["mean"], color=SERIES[k], marker="o", ms=3, label=f"{z0} < z < {z1}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("λ (rema)")
    axes[0].set_ylabel("fraction with a Wen & Han counterpart")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("M500, Wen & Han [10¹⁴ M☉]")
    axes[1].set_ylabel("fraction with a rema counterpart")
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right")
    save(fig, "external_wenhan")
    record("wh_frac_lam20", float(has_wh[lam >= 20].mean()))
    record("wh_frac_m2", float(has_rm[inside & (wh["M500"] >= 2)].mean()))
    print(f"rema lambda >= 20 with a Wen & Han counterpart: {has_wh[lam >= 20].mean():.1%}; "
          f"Wen & Han M500 >= 2e14 with a rema counterpart: {has_rm[inside & (wh['M500'] >= 2)].mean():.1%}")
