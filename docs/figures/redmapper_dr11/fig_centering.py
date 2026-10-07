"""Centring figures (Rykoff et al. 2016 Figs. 11-12; Zhang et al. 2019 Figs. 1, 3; Saro et al. 2015;
Bleem et al. 2020; Seppi et al. 2023).

    python docs/figures/redmapper_dr11/fig_centering.py
"""

# %% [script-only]
from common import *  # noqa: F403

# %% [markdown]
# ## Centring
#
# rema keeps redMaPPer's centring: up to five candidate central galaxies per cluster, each with
# a probability P_CEN from the wcen likelihood (luminosity, zred and local density). The
# catalogue position is the most likely central.

# %%
cl = load("clusters")
lam, zl = np.asarray(cl["LAMBDA"]), np.asarray(cl["Z_LAMBDA"])
pcen = np.asarray(cl["P_CEN"])
p0 = pcen[:, 0]
rlam = np.asarray(cl["R_LAMBDA"])                    # h^-1 Mpc
ra, dec = np.asarray(cl["RA"]), np.asarray(cl["DEC"])

# %% [markdown]
# ### Centring probabilities (Rykoff et al. 2016 Sect. 8)
#
# Distribution of the probability of the most likely central, its mean against λ and z_λ, and
# the fraction of clusters with P_cen > 0.9.
# Rykoff et al. (2016) predicted ⟨P_cen⟩ = 0.82 for DES SV (λ ≥ 20) and measured a well-centred
# fraction ρ0 = 0.78 ± 0.11 from X-ray and SZ centres.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)
ax = axes[0]
for (l0, l1), col in (((5, 20), BLUE), ((20, 1e4), ORANGE)):
    s = (lam >= l0) & (lam < l1)
    lab = f"λ ≥ {l0}" if l1 > 1e3 else f"{l0} ≤ λ < {l1}"
    ax.hist(p0[s], bins=np.linspace(0, 1, 51), histtype="step", density=True, color=col, lw=1.4,
            label=f"{lab}: ⟨P_cen⟩ = {p0[s].mean():.2f}")
ax.set_xlabel("P_cen of the most likely central")
ax.set_ylabel("probability density")
ax.legend(loc="upper left")
ax = axes[1]
lb = np.logspace(np.log10(5), np.log10(200), 16)
for k, (z0, z1) in enumerate(((0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9))):
    s = (zl >= z0) & (zl < z1)
    b = binned(lam[s], p0[s], lb, min_n=30)
    ax.plot(b["x"], b["mean"], color=SERIES[k], marker="o", ms=3, label=f"{z0} < z_λ < {z1}")
ax.axhline(LIT["R16_pcen"], color=INK2, ls=":", lw=0.8, label="Rykoff+16 prediction (0.82)")
ax.set_xscale("log")
ax.set_xlabel("λ")
ax.set_ylabel("⟨P_cen⟩")
ax.set_ylim(0.4, 1.0)
ax.legend(loc="lower left")
ax = axes[2]
s = lam >= 20
b = binned(zl[s], p0[s], np.arange(0.05, 0.951, 0.05), min_n=30)
ax.plot(b["x"], b["mean"], color=ORANGE, marker="o", ms=3, label="⟨P_cen⟩, λ ≥ 20")
b2 = binned(zl[s], (p0[s] > 0.9).astype(float), np.arange(0.05, 0.951, 0.05), min_n=30)
ax.plot(b2["x"], b2["mean"], color=BLUE, marker="o", ms=3, label="fraction with P_cen > 0.9, λ ≥ 20")
ax.axhline(LIT["R16_pcen"], color=INK2, ls=":", lw=0.8)
ax.set_xlabel("z_λ")
ax.set_ylim(0.0, 1.0)
ax.legend(loc="lower left")
save(fig, "centering_pcen")
record("pcen_mean_lam20", float(p0[lam >= 20].mean()))

# %% [markdown]
# ### Offsets from X-ray and SZ centres (Rykoff et al. 2016 Figs. 11-12; Zhang et al. 2019 Fig. 3)
#
# Each external cluster (redshift > 0.05) is paired with the richest rema cluster (λ ≥ 20)
# within 10′ and |z_λ − z|/(1+z) < 0.05; the offset is in units of R_λ at z_λ. The offsets
# are fitted with the two-component model of Rykoff et al. (2016): a fraction ρ0 of well
# centred clusters, whose offsets follow a Rayleigh distribution of width σ0 (the positional
# errors), and miscentred ones, Rayleigh of width σ1 (in R_λ). The external centres are the
# eRASS1 X-ray positions (Bulbul et al. 2024), ACT DR6 SZ positions (beam 1.4′) and SPT-SZ
# 2500d positions (Bocquet et al. 2019).

# %%
from scipy.optimize import minimize  # noqa: E402



def offsets(ref, xmax=1.5):
    """R/R_lambda of the rema clusters paired with the external clusters (and their indices)."""
    good = np.isfinite(ref["Z"]) & (ref["Z"] > 0.05)
    sub = np.flatnonzero(lam >= 20)
    ir = np.flatnonzero(good)
    a, b = match_within(ref["RA"][ir], ref["DEC"][ir], ref["Z"][ir], ra[sub], dec[sub], zl[sub], 10.0,
                        dz_max=0.05, rank2=lam[sub])
    i, r = sub[b], ir[a]
    sep = SkyCoord(ra[i] * u.deg, dec[i] * u.deg).separation(SkyCoord(ref["RA"][r] * u.deg, ref["DEC"][r] * u.deg))
    rmpc = sep.radian * COSMO.angular_diameter_distance(zl[i]).value * H        # h^-1 Mpc
    x = rmpc / rlam[i]
    keep = x < xmax
    return x[keep], i[keep], r[keep]


def fit_offsets(x, xmax=1.5):
    """Maximum-likelihood rho0, sigma0, sigma1 of the two-Rayleigh model, truncated at xmax."""
    def ray(x, s):
        return x / s**2 * np.exp(-0.5 * (x / s) ** 2) / (1 - np.exp(-0.5 * (xmax / s) ** 2))

    def nll(p):
        r0, s0, s1 = p
        if not (0 < r0 < 1 and 0.005 < s0 < s1 < 2):
            return 1e30
        return -np.sum(np.log(r0 * ray(x, s0) + (1 - r0) * ray(x, s1) + 1e-300))

    best = min((minimize(nll, p0_, method="Nelder-Mead") for p0_ in ([0.8, 0.05, 0.3], [0.6, 0.1, 0.5])),
               key=lambda r: r.fun)
    return best.x, ray


fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)
fits = {}
for ax, (name, lab) in zip(axes, (("erass1", "eRASS1 X-ray"), ("act_dr6", "ACT DR6 SZ"), ("spt_2500d", "SPT-SZ 2500d"))):
    ref = external(name)
    if ref is None:
        ax.set_visible(False)
        continue
    x, i, r = offsets(ref)
    (r0, s0, s1), ray = fit_offsets(x)
    xs = np.linspace(0.002, 1.5, 300)
    ax.hist(x, bins=np.linspace(0, 1.5, 61), density=True, color=BLUE, alpha=0.45, lw=0)
    ax.plot(xs, r0 * ray(xs, s0) + (1 - r0) * ray(xs, s1), color=INK, lw=1.4,
            label=f"ρ0 = {r0:.2f}, σ0 = {s0:.3f}, σ1 = {s1:.2f}")
    ax.plot(xs, (1 - r0) * ray(xs, s1), color=ORANGE, lw=1.0, ls=(0, (4, 2)), label="miscentred part")
    ax.set_xlabel("R / R_λ")
    ax.set_ylabel("probability density")
    ax.legend(loc="upper right")
    panel_label(ax, f"{lab}: {x.size:,} pairs")
    ax.set_xlim(0, 1.5)
    fits[name] = (r0, s0, s1, x.size)
    record(f"centring_{name}", [round(float(r0), 3), round(float(s0), 4), round(float(s1), 3), int(x.size)])
    print(f"{lab}: {x.size} pairs, rho0 = {r0:.2f}, sigma0 = {s0:.3f}, sigma1 = {s1:.2f}; "
          f"<P_cen> of the paired clusters {p0[i].mean():.2f}")
save(fig, "centering_offsets")

# %% [markdown]
# ### Offsets from other redMaPPer centres (Zhang et al. 2019 Fig. 1)
#
# The same pairing with the DES Y1 and SDSS DR8 redMaPPer catalogues: the fraction of clusters
# whose central galaxy is the same (offset below 0.02 R_λ, a few arcseconds) measures how
# reproducible the centring is between surveys of different depth and filters.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for ax, (name, lab) in zip(axes, (("des_y1", "DES Y1 redMaPPer"), ("sdss_dr8", "SDSS DR8 redMaPPer"))):
    ref = external(name)
    if ref is None:
        ax.set_visible(False)
        continue
    sel = ref["LAMBDA"] >= 20
    ref = {k: v[sel] for k, v in ref.items()}
    x, i, r = offsets(ref)
    same = np.mean(x < 0.02)
    ax.hist(x, bins=np.r_[0, 0.02, np.arange(0.05, 1.501, 0.05)], color=BLUE, alpha=0.6, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("R / R_λ (first bin: < 0.02, the same central galaxy)")
    ax.set_ylabel("pairs")
    panel_label(ax, f"{lab}: {x.size:,} pairs, same central {same:.0%}")
    record(f"centring_same_{name}", [float(same), int(x.size)])
save(fig, "centering_optical")
