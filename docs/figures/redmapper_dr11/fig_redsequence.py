"""Red-sequence calibration figures (Rykoff et al. 2014 Figs. 1, 3-5; Kluge et al. 2024 Fig. C.1).

    python docs/figures/redmapper_dr11/fig_redsequence.py
"""

# %% [script-only]
from common import *  # noqa: F403

# %% [markdown]
# ## Red-sequence calibration
#
# The red-sequence model of the production calibration (RA 160–180° and 190–210°,
# Dec −10–10°): mean colours at the pivot magnitude, slopes and intrinsic scatters, as cubic
# splines in redshift. It is compared with the spectroscopic training galaxies, with the
# members of the final catalogue, with the DR10 grz model of Kluge et al. (2024) and with the
# g − r red sequence that Comparat et al. (2025, Table 2) measured from cluster–galaxy
# cross-correlations.

# %%
from rema.calibration import Calibration
from rema.io.tables import read_table
from rema.model.redsequence import RSModel

CAL = Calibration.read(CALIB)
RS = CAL.rs
KLUGE_PARS = Path(os.environ.get("REMA_KLUGE_PARS", Path.home() / "data" / "eromapper" / "data" / "cal" /
                                 "legacy_dr10_grz_z_v0.3" / "legacy_dr10_south_v0.3_grz_z_cal_iter1_pars.fit"))
KLUGE = RSModel.from_redmapper_pars(str(KLUGE_PARS)) if KLUGE_PARS.exists() else None
# Comparat et al. (2025), Table 2: z, g-r, its error, scatter, its error.
COMPARAT = np.array([
    [0.156, 1.0618, 0.0007, 0.0461, 0.0007], [0.165, 1.0981, 0.0006, 0.0450, 0.0006],
    [0.175, 1.1364, 0.0009, 0.0450, 0.0010], [0.185, 1.1712, 0.0005, 0.0468, 0.0005],
    [0.195, 1.2036, 0.0007, 0.0440, 0.0007], [0.205, 1.2488, 0.0009, 0.0438, 0.0009],
    [0.215, 1.2837, 0.0008, 0.0466, 0.0008], [0.225, 1.3194, 0.0007, 0.0495, 0.0007],
    [0.235, 1.3498, 0.0008, 0.0518, 0.0008], [0.245, 1.3832, 0.0009, 0.0556, 0.0010],
    [0.255, 1.4114, 0.0011, 0.0576, 0.0011], [0.266, 1.4409, 0.0009, 0.0482, 0.0011],
    [0.275, 1.4752, 0.0010, 0.0590, 0.0010], [0.285, 1.5062, 0.0007, 0.0546, 0.0008],
    [0.295, 1.5373, 0.0008, 0.0555, 0.0009], [0.305, 1.5566, 0.0012, 0.0552, 0.0013],
    [0.315, 1.5766, 0.0009, 0.0587, 0.0010], [0.325, 1.6148, 0.0008, 0.0548, 0.0008],
    [0.334, 1.6405, 0.0014, 0.0539, 0.0014], [0.344, 1.6762, 0.0010, 0.0417, 0.0011]])
COLOURS = ["g − r", "r − i", "i − z"]
ZR = np.linspace(0.05, 0.9, 341)
at = RS.at(ZR)
MEAN, SLOPE, PIVOT = np.asarray(at.mean), np.asarray(at.slope), np.asarray(at.pivot)
SIG = np.sqrt(np.einsum("...ii->...i", np.asarray(at.cint)))
if KLUGE is not None:
    k = KLUGE.at(ZR)
    K_MEAN, K_SIG = np.asarray(k.mean), np.sqrt(np.einsum("...ii->...i", np.asarray(k.cint)))
print(f"bands {RS.bands}, reference {RS.ref_band}; {len(RS.z_mean)} colour nodes, "
      f"{len(RS.z_sigma)} scatter nodes; Kluge+24 model {'found' if KLUGE is not None else 'missing'}")


def colour_at_pivot(colour, refmag, z, j):
    """Colour j moved along the red-sequence slope to the pivot magnitude at redshift z."""
    a = RS.at(np.asarray(z, np.float64))
    return colour - np.asarray(a.slope)[:, j] * (refmag - np.asarray(a.pivot))


# %% [markdown]
# ### Colour against redshift (Rykoff et al. 2014 Figs. 1 and 4; Kluge et al. 2024 Fig. C.1)
#
# Top: the spectroscopic training galaxies of the calibration area (m < m* + 1, χ² < 20 against
# the final model), at their spectroscopic redshift. Bottom: members with P > 0.9 of the whole
# catalogue, at the z_λ of their cluster. Colours are moved along the red-sequence slope to
# the pivot magnitude. The training sample (`rs_specz.fits`) comes from the calibration
# directory, when it is there.

# %%
mem = load("members", ["P", "GR", "RI", "IZ", "REFMAG", "Z"])
sel = np.flatnonzero(np.asarray(mem["P"]) > 0.9)
mz = np.asarray(mem["Z"])[sel]
mref = np.asarray(mem["REFMAG"])[sel]
mcol = np.stack([np.asarray(mem[c])[sel] for c in ("GR", "RI", "IZ")], axis=1)
ok = np.all(np.isfinite(mcol), axis=1) & (mz > 0.05) & (mz < 0.9)
mz, mref, mcol = mz[ok], mref[ok], mcol[ok]
mpiv = np.stack([colour_at_pivot(mcol[:, j], mref, mz, j) for j in range(3)], axis=1)
print(f"{mz.size:,} members with P > 0.9 and positive fluxes in g, r, i, z")

spec_file = RUNS[0] / "calib" / "plots" / "rs_specz.fits"
if spec_file.exists():
    d = read_table(spec_file)
    red = (np.asarray(d["CHISQ"]) < 20.0) & np.all(np.isfinite(d["COLOR_PIV"]), axis=1)
    sz, sc = np.asarray(d["Z"])[red], np.asarray(d["COLOR_PIV"])[red]
    print(f"{red.sum():,} training galaxies with chi2 < 20 (of {red.size:,} brighter than m* + 1)")
else:
    sz = sc = None
    print("rs_specz.fits not found: the top row is left out")

from matplotlib.colors import LogNorm  # noqa: E402

YLIM = [(0.6, 2.3), (0.2, 1.5), (0.1, 0.95)]
rows = 2 if sz is not None else 1
fig, axes = plt.subplots(rows, 3, figsize=(12, 3.6 * rows), constrained_layout=True, squeeze=False)
for r in range(rows):
    z_, c_ = (sz, sc) if (r == 0 and sz is not None) else (mz, mpiv)
    for j, ax in enumerate(axes[r]):
        ax.hist2d(z_, c_[:, j], bins=[np.linspace(0.05, 0.9, 171), np.linspace(*YLIM[j], 141)],
                  cmap=RAMP, norm=LogNorm(vmin=1), rasterized=True)
        ax.plot(ZR, MEAN[:, j], color=INK, lw=1.4, label="rema DR11 model")
        for s in (-3, 3):
            ax.plot(ZR, MEAN[:, j] + s * SIG[:, j], color=INK, lw=0.8, ls=(0, (4, 2)),
                    label="±3σ_int" if s > 0 else None)
        ax.plot(np.asarray(RS.z_mean), np.interp(RS.z_mean, ZR, MEAN[:, j]), "o", ms=3, color=INK)
        if KLUGE is not None and j == 0:
            ax.plot(ZR, K_MEAN[:, 0], color=ORANGE, lw=1.4, label="Kluge+24 DR10 grz")
        ax.set_ylim(*YLIM[j])
        ax.set_xlim(0.05, 0.9)
        ax.set_ylabel(f"{COLOURS[j]} at the pivot [mag]")
        ax.set_xlabel("spectroscopic redshift" if (r == 0 and sz is not None) else "z_λ")
    panel_label(axes[r, 0], "training galaxies (calibration area)" if (r == 0 and sz is not None)
                else "members, P > 0.9")
axes[0, 0].legend(loc="lower right")
save(fig, "rs_colour_z")

# %% [markdown]
# ### Red-sequence parameters (Rykoff et al. 2014 Fig. 5)
#
# Mean colour at the pivot, slope dc/dm and intrinsic scatter σ_int of each colour, and the
# pivot magnitude against m*(z). Kluge et al. (2024) fitted g − r and r − z (grz bands): their
# g − r is drawn with the rema g − r, their r − z with the rema (r − i) + (i − z).

# %%
from rema.model.profiles import MStar  # noqa: E402

rz = MEAN[:, 1] + MEAN[:, 2]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), constrained_layout=True)
ax = axes[0]
for j in range(3):
    ax.plot(ZR, MEAN[:, j], color=SERIES[j], label=f"{COLOURS[j]}")
ax.plot(ZR, rz, color=SERIES[3], label="r − z = (r − i) + (i − z)")
if KLUGE is not None:
    ax.plot(ZR, K_MEAN[:, 0], color=INK2, lw=1.0, ls=(0, (4, 2)), label="Kluge+24 (g − r, r − z)")
    ax.plot(ZR, K_MEAN[:, 1], color=INK2, lw=1.0, ls=(0, (4, 2)))
ax.errorbar(COMPARAT[:, 0], COMPARAT[:, 1], yerr=COMPARAT[:, 2], fmt="o", ms=3, color=INK,
            label="Comparat+25 g − r")
ax.set_ylabel("mean colour at the pivot [mag]")
ax.legend(loc="lower right", fontsize=7)
ax = axes[1]
for j in range(3):
    ax.plot(ZR, SLOPE[:, j], color=SERIES[j], label=COLOURS[j])
ax.axhline(0, color=INK2, lw=0.8)
ax.set_ylabel("slope dc/dm_z")
ax = axes[2]
for j in range(3):
    ax.plot(ZR, SIG[:, j], color=SERIES[j], label=COLOURS[j])
if KLUGE is not None:
    ax.plot(ZR, K_SIG[:, 0], color=INK2, lw=1.0, ls=(0, (4, 2)), label="Kluge+24 σ(g − r)")
ax.errorbar(COMPARAT[:, 0], COMPARAT[:, 3], yerr=COMPARAT[:, 4], fmt="o", ms=3, color=INK,
            label="Comparat+25 σ(g − r)")
ax.set_ylim(0, 0.2)
ax.set_ylabel("intrinsic scatter σ_int [mag]")
ax.legend(loc="upper left", fontsize=7)
ax = axes[3]
ms = np.asarray(MStar(CAL.config.model.mstar)(ZR))
ax.plot(ZR, PIVOT, color=BLUE, label="pivot magnitude")
ax.plot(ZR, ms, color=INK2, ls=(0, (4, 2)), label="m*(z)")
ax.set_ylabel("z-band magnitude")
ax.legend(loc="upper left")
for ax in axes:
    ax.set_xlabel("redshift")
    ax.set_xlim(0.05, 0.9)
save(fig, "rs_parameters")

for zz in (0.2, 0.4, 0.6, 0.8):
    i = np.argmin(np.abs(ZR - zz))
    print(f"z = {zz}: sigma_int(g-r, r-i, i-z) = {SIG[i, 0]:.3f}, {SIG[i, 1]:.3f}, {SIG[i, 2]:.3f}")
    record(f"rs_sig_z{zz}", [round(float(s), 4) for s in SIG[i]])

# %% [markdown]
# ### Composite red sequence (Rykoff et al. 2014 Fig. 3)
#
# Members with P > 0.9 in four thin slices of z_λ, in the colour that straddles the 4000 Å
# break at that redshift, against the reference (z-band) magnitude, with the model mean and
# ±1σ_int.

# %%
SLICES = [(0.15, 0.16, 0), (0.30, 0.31, 0), (0.50, 0.51, 1), (0.75, 0.76, 2)]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), constrained_layout=True)
for ax, (z0, z1, j) in zip(axes, SLICES):
    s = (mz >= z0) & (mz < z1)
    zc = 0.5 * (z0 + z1)
    a = RS.at(np.array([zc]))
    m0, sl, pv = float(np.asarray(a.mean)[0, j]), float(np.asarray(a.slope)[0, j]), float(np.asarray(a.pivot)[0])
    sg = float(np.sqrt(np.asarray(a.cint)[0, j, j]))
    xm = np.linspace(np.percentile(mref[s], 0.5), np.percentile(mref[s], 99.5), 50)
    ylo, yhi = m0 - 0.6, m0 + 0.6
    ax.hist2d(mref[s], mcol[s, j], bins=[np.linspace(xm[0], xm[-1], 80), np.linspace(ylo, yhi, 80)],
              cmap=RAMP, norm=LogNorm(vmin=1), rasterized=True)
    mod = m0 + sl * (xm - pv)
    ax.plot(xm, mod, color=INK, lw=1.4)
    ax.plot(xm, mod - sg, color=INK, lw=0.8, ls=(0, (4, 2)))
    ax.plot(xm, mod + sg, color=INK, lw=0.8, ls=(0, (4, 2)))
    ax.set_xlabel("m_z [mag]")
    ax.set_ylabel(f"{COLOURS[j]} [mag]")
    panel_label(ax, f"{z0:.2f} < z_λ < {z1:.2f}: {s.sum():,} members")
save(fig, "rs_cmd")
