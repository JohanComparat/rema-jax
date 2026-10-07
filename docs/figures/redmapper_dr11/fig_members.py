"""Member figures (Rykoff et al. 2014 Fig. 16; Kluge et al. 2024 Fig. 4; Rozo et al. 2015b Fig. 8;
Rines et al. 2018 Fig. 14; Myles et al. 2021 Fig. 3; Tomooka et al. 2020 Fig. 2).

    python docs/figures/redmapper_dr11/fig_members.py
"""

# %% [script-only]
from common import *  # noqa: F403

# %% [markdown]
# ## Members
#
# The member catalogue lists every galaxy with a membership probability P ≥ 0.01 (P = PMEM ×
# θ_L × θ_R: colour, luminosity and radial weights) and the centre candidates. The reduced
# table of `prepare.py` keeps P ≥ 0.05 and every member with a spectroscopic redshift.
# `VEL` is the rest-frame velocity relative to the velocity-clipped cluster redshift, and
# `ISMEMBER_SPEC` flags the members kept by the 3σ clipping (Clerc et al. 2016).

# %%
cl = load("clusters")
mem = load("members")
lam, zl = np.asarray(cl["LAMBDA"]), np.asarray(cl["Z_LAMBDA"])
row, P = np.asarray(mem["CL_ROW"]), np.asarray(mem["P"])
zsp, vel, kept = np.asarray(mem["ZSPEC"]), np.asarray(mem["VEL"]), np.asarray(mem["ISMEMBER_SPEC"]).astype(bool)
rlam = np.asarray(cl["R_LAMBDA"])
x = np.asarray(mem["R"]) / rlam[row]
print(f"{P.size:,} members in the reduced table, {np.sum(zsp > 0):,} with a spectroscopic redshift")

# %% [markdown]
# ### An example cluster (Rykoff et al. 2014 Fig. 16; Kluge et al. 2024 Fig. 4)
#
# The cluster with the most spectroscopic members among those with λ ≥ 80 at 0.1 < z_λ < 0.3,
# outside the calibration area. (a) members on the sky, coloured by P, with the R_λ circle
# and the centre candidates labelled with P_cen; (b) P(z) with z_λ, the velocity-clipped
# spectroscopic redshift and that of the central; (c) distribution of P; (d) Σ P within R
# against the NFW filter of the richness (normalised to λ at R_λ); (e) rest-frame velocities
# of the spectroscopic members with the clipping limits ±3σ_v.

# %%
nmem = np.asarray(cl["N_MEMBERS"])
cand = np.flatnonzero((lam >= 80) & (zl > 0.1) & (zl < 0.3) & ~np.asarray(cl["IN_CALIB"]))
c = cand[np.argmax(nmem[cand])]
ra0, dec0, z0, rl = float(cl["RA"][c]), float(cl["DEC"][c]), float(zl[c]), float(rlam[c])
name = f"J{ra0:07.3f}{dec0:+07.3f}"
print(f"example: rema {name}, lambda = {lam[c]:.1f} ± {cl['LAMBDA_E'][c]:.1f}, z_lambda = {z0:.4f}, "
      f"{nmem[c]} spectroscopic members, sigma_v = {cl['VDISP_BOOT'][c]:.0f} km/s")
mine = np.flatnonzero(row == c)


def nfw_sigma_np(r, rs=0.15, rcore=0.1):
    """Projected NFW shape F(x) of the richness filter (flat inside rcore), r in h^-1 Mpc (numpy
    version of rema.model.profiles.nfw_sigma)."""
    x = np.maximum(np.asarray(r, np.float64), rcore) / rs
    out = np.full(x.shape, 1.0 / 3.0)
    lo, hi = x < 0.999, x > 1.001
    out[lo] = (1 - np.arccosh(1 / x[lo]) / np.sqrt(1 - x[lo] ** 2)) / (x[lo] ** 2 - 1)
    out[hi] = (1 - np.arccos(1 / x[hi]) / np.sqrt(x[hi] ** 2 - 1)) / (x[hi] ** 2 - 1)
    return out


def nfw_enclosed_np(r, rs=0.15, rcore=0.1):
    """Integral of 2 pi R nfw_sigma_np(R) from 0 to r (numerical)."""
    g = np.linspace(0, np.max(r), 4001)
    cum = np.concatenate([[0], np.cumsum(0.5 * np.diff(g) * (2 * np.pi * g * nfw_sigma_np(g, rs, rcore))[1:]
                                          + 0.5 * np.diff(g) * (2 * np.pi * g * nfw_sigma_np(g, rs, rcore))[:-1])])
    return np.interp(r, g, cum)


fig = plt.figure(figsize=(15, 7.6), constrained_layout=True)
gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 1, 1, 1])
ax = fig.add_subplot(gs[:, 0])
mra, mdec, mp = np.asarray(mem["RA"])[mine], np.asarray(mem["DEC"])[mine], P[mine]
dx = (mra - ra0) * np.cos(np.radians(dec0)) * 60
dy = (mdec - dec0) * 60
order = np.argsort(mp)
sc = ax.scatter(dx[order], dy[order], c=mp[order], s=8 + 40 * mp[order], cmap=RAMP, vmin=0, vmax=1, lw=0)
mpc_arcmin = np.radians(1 / 60) * COSMO.angular_diameter_distance(z0).value * H
t = np.linspace(0, 2 * np.pi, 200)
ax.plot(rl / mpc_arcmin * np.cos(t), rl / mpc_arcmin * np.sin(t), color=INK2, lw=0.8)
rac, decc, pc = np.asarray(cl["RA_CENT"])[c], np.asarray(cl["DEC_CENT"])[c], np.asarray(cl["P_CEN"])[c]
for k in range(len(pc)):
    if pc[k] > 0.01:
        cx, cy = (rac[k] - ra0) * np.cos(np.radians(dec0)) * 60, (decc[k] - dec0) * 60
        ax.plot(cx, cy, "o", mfc="none", mec=ORANGE, ms=12, mew=1.4)
        ax.annotate(f"{pc[k]:.2f}", (cx, cy), xytext=(9, 9), textcoords="offset points", fontsize=8, color=INK)
ax.set_aspect("equal")
lim = 1.15 * rl / mpc_arcmin
ax.set_xlim(lim, -lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("ΔRA [arcmin]")
ax.set_ylabel("ΔDec [arcmin]")
fig.colorbar(sc, ax=ax, shrink=0.6, label="P")
ax.set_title(f"(a) {name}: λ = {lam[c]:.0f}, z_λ = {z0:.3f}", loc="left")
ax = fig.add_subplot(gs[0, 1])
pzb, pz = np.asarray(cl["PZBINS"])[c], np.asarray(cl["PZ"])[c]
ax.plot(pzb, pz / np.trapezoid(pz, pzb), color=BLUE)
ax.axvline(z0, color=BLUE, lw=0.8, ls=(0, (4, 2)), label="z_λ")
ax.axvline(cl["SPEC_Z_BOOT"][c], color=ORANGE, lw=1.0, label="z_spec (members)")
ax.axvline(cl["CG_SPEC_Z"][c], color=AQUA, lw=1.0, ls=":", label="z_spec (central)")
ax.set_xlabel("z")
ax.set_ylabel("P(z)")
ax.legend(fontsize=7, loc="upper right")
panel_label(ax, "(b)")
ax = fig.add_subplot(gs[0, 2])
ax.hist(mp, bins=np.linspace(0, 1, 21), color=BLUE, alpha=0.6)
ax.set_xlabel("P")
ax.set_ylabel("members")
panel_label(ax, "(c)")
ax = fig.add_subplot(gs[0, 3])
rm = np.asarray(mem["R"])[mine]
rg = np.linspace(0.02, rl, 60)
ax.plot(rg, [mp[rm < r].sum() for r in rg], color=BLUE, label="Σ P(< R)")
nfw = nfw_enclosed_np(rg) / nfw_enclosed_np(np.array([rl]))[0] * lam[c]
ax.plot(rg, nfw, color=INK2, ls=(0, (4, 2)), label="NFW filter × λ")
ax.set_xlabel("R [h⁻¹ Mpc]")
ax.legend(fontsize=7)
panel_label(ax, "(d)")
ax = fig.add_subplot(gs[1, 1:])
spec = mine[zsp[mine] > 0]
v = vel[spec]
sig = float(cl["VDISP_BOOT"][c])
bins = np.arange(-5000, 5001, 250)
ax.hist(v[np.isfinite(v)], bins, color=MUTED, alpha=0.7, label="spectroscopic members (P ≥ 0.01)")
ax.hist(v[kept[spec]], bins, color=BLUE, alpha=0.7, label="kept by the clipping")
for s in (-3, 3):
    ax.axvline(s * sig, color=ORANGE, lw=1.0, ls=(0, (4, 2)))
vs = np.linspace(-5000, 5000, 400)
ax.plot(vs, kept[spec].sum() * 250 / (np.sqrt(2 * np.pi) * sig) * np.exp(-0.5 * (vs / sig) ** 2), color=INK,
        lw=1.2, label=f"Gaussian σ_v = {sig:.0f} km/s")
ax.set_xlabel("rest-frame velocity [km/s]")
ax.set_ylabel("members")
ax.legend(fontsize=7)
panel_label(ax, "(e)")
save(fig, "member_example")

# %% [markdown]
# ### Membership probability against spectroscopic membership (Rozo et al. 2015b Fig. 8; Rines et al. 2018 Fig. 14)
#
# Members with a spectroscopic redshift, in clusters with a velocity-clipped redshift and
# dispersion (≥ 10 spectroscopic members, unflagged), excluding the central: the fraction kept
# by the 3σ clipping in bins of P. If P is calibrated, the fraction follows the diagonal. The
# spectroscopic targets are the bright galaxies of SDSS, BOSS and DESI, so the test covers the
# bright end of the members.

# %%
vflag = np.asarray(cl["VDISP_FLAG"]).astype(bool)
good_cl = (nmem >= 10) & ~vflag & np.isfinite(np.asarray(cl["VDISP_BOOT"]))
s = (zsp > 0) & good_cl[row] & (np.asarray(mem["CENT_RANK"]) != 0) & np.isfinite(vel)
pb = np.linspace(0, 1, 11)
fig, ax = plt.subplots(figsize=(5.5, 4.6), constrained_layout=True)
for k, (z0_, z1_) in enumerate(((0.05, 0.2), (0.2, 0.35), (0.35, 0.6))):
    w = s & (zl[row] >= z0_) & (zl[row] < z1_)
    b = binned(P[w], kept[w].astype(float), pb, min_n=30)
    ax.plot(b["x"], b["mean"], color=SERIES[k], marker="o", ms=4, label=f"{z0_} < z_λ < {z1_} ({w.sum():,})")
ax.plot([0, 1], [0, 1], color=INK2, lw=0.8)
ax.set_xlabel("P (photometric)")
ax.set_ylabel("fraction kept by the velocity clipping")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(loc="upper left")
save(fig, "pmem_spec")
b = binned(P[s], kept[s].astype(float), pb, min_n=30)
record("pmem_spec_frac", [round(float(f), 3) for f in b["mean"]])

# %% [markdown]
# ### Stacked velocities and projected fraction (Myles et al. 2021 Fig. 3)
#
# P-weighted rest-frame velocities of the spectroscopic members (not the central) of the
# clusters above, stacked in units of each cluster's σ_v, in bins of λ. A narrow Gaussian
# (the cluster) plus a flat background within |v| < 5000 km/s is fitted to each stack; the
# P-weighted fraction in the flat component estimates the fraction of members that are
# projected along the line of sight.

# %%
from scipy.optimize import minimize  # noqa: E402

VMAX = 5000.0
w = s & (np.abs(vel) < VMAX)
sv = np.asarray(cl["VDISP_BOOT"])[row]
lbins = [(10, 20), (20, 40), (40, 80), (80, 400)]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
fproj = []
for k, (l0, l1) in enumerate(lbins):
    q = w & (lam[row] >= l0) & (lam[row] < l1)
    vq, pq, sq = vel[q], P[q], sv[q]

    def nll(p, vq=vq, pq=pq, sq=sq):
        f, a = p
        if not (0 < f < 1 and 0.5 < a < 2):
            return 1e30
        g = np.exp(-0.5 * (vq / (a * sq)) ** 2) / (np.sqrt(2 * np.pi) * a * sq)
        return -np.sum(pq * np.log((1 - f) * g + f / (2 * VMAX)))

    res = minimize(nll, [0.2, 1.0], method="Nelder-Mead")
    f, a = res.x
    fproj.append((np.sqrt(l0 * l1), f, q.sum()))
    u_ = vq / sq
    axes[0].hist(u_, bins=np.linspace(-8, 8, 81), weights=pq, density=True, histtype="step", color=SERIES[k], lw=1.3,
                 label=f"{l0} ≤ λ < {l1}: f_proj = {f:.2f}")
axes[0].set_yscale("log")
axes[0].set_xlabel("v / σ_v")
axes[0].set_ylabel("P-weighted density")
axes[0].legend(fontsize=7)
fp = np.array(fproj)
axes[1].plot(fp[:, 0], fp[:, 1], color=BLUE, marker="o")
axes[1].set_xscale("log")
plain_log_ticks(axes[1])
axes[1].set_xlabel("λ")
axes[1].set_ylabel("projected fraction f_proj")
axes[1].set_ylim(0, 0.5)
save(fig, "member_projection")
record("fproj", [[round(float(a_), 1), round(float(b_), 3)] for a_, b_, _ in fproj])

# %% [markdown]
# ### Radial profiles (Rykoff et al. 2014 Fig. 16; Tomooka et al. 2020 Fig. 2)
#
# Left: stacked surface density of membership probability, Σ P per unit area in units of
# R_λ, divided by λ, in bins of λ, with the NFW filter of the richness (r_s = 0.15 h⁻¹ Mpc,
# for the median R_λ of each bin). Right: fraction of the spectroscopic members (P ≥ 0.05)
# kept by the velocity clipping against R/R_λ.

# %%
xb = np.linspace(0.0, 1.2, 25)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
for k, (l0, l1) in enumerate(lbins):
    cls = (lam >= l0) & (lam < l1) & (zl > 0.1) & (zl < 0.6)
    q = cls[row]
    ssum = np.histogram(x[q], xb, weights=P[q] / lam[row[q]])[0]
    ann = np.pi * (xb[1:] ** 2 - xb[:-1] ** 2)
    prof = ssum / ann / cls.sum()
    xc = 0.5 * (xb[1:] + xb[:-1])
    axes[0].plot(xc, prof, color=SERIES[k], marker="o", ms=3, label=f"{l0} ≤ λ < {l1}")
    rl_med = np.median(rlam[cls])
    rr = xc * rl_med
    nf = nfw_sigma_np(rr)
    inner = xc < 1.0                                  # same integral inside R_lambda
    norm = np.sum((nf * ann)[inner]) / np.sum((prof * ann)[inner])
    axes[0].plot(xc, nf / norm, color=SERIES[k], lw=0.8, ls=(0, (4, 2)))
    q2 = q & (zsp > 0) & good_cl[row] & (np.asarray(mem["CENT_RANK"]) != 0)
    b = binned(x[q2], kept[q2].astype(float), xb, min_n=30)
    axes[1].plot(b["x"], b["mean"], color=SERIES[k], marker="o", ms=3, label=f"{l0} ≤ λ < {l1}")
axes[0].set_yscale("log")
axes[0].set_xlabel("R / R_λ")
axes[0].set_ylabel("Σ P / λ per (R/R_λ)² per cluster")
axes[0].legend(fontsize=7)
axes[1].set_xlabel("R / R_λ")
axes[1].set_ylabel("fraction kept by the velocity clipping")
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=7)
save(fig, "member_profiles")
