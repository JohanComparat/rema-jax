"""What the finder sees of the cosmology: D_A(z) [h^-1 Mpc] and E(z)/E(z_ref) for one-at-a-time
changes of the parameters, relative to the fiducial (Omega_m = 0.3, h = 0.7, w0 = -1)."""

import numpy as np

from common import BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, INK2, plt, record, save, panel_label
from rema.config import CosmologyConfig
from rema.model.cosmo import CosmoTable

z = np.linspace(0.05, 0.95, 91)
fid = CosmoTable.from_config(CosmologyConfig())
var = [("Omega_m", 0.35, BLUE), ("Omega_m", 0.25, BLUE), ("w0", -0.8, ORANGE), ("w0", -1.2, ORANGE),
       ("wa", 0.3, AQUA), ("h", 0.75, YELLOW), ("sum_mnu", 0.24, MAGENTA), ("Omega_b", 0.06, GREEN)]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for p, v, col in var:
    t = CosmoTable.from_config(CosmologyConfig(**{p: v}))
    dd = np.asarray(t.da(z)) / np.asarray(fid.da(z)) - 1
    de = (np.asarray(t.volume_factor(z, 0.9)) / np.asarray(fid.volume_factor(z, 0.9))) - 1
    ls = "-" if v > getattr(CosmologyConfig(), p) else "--"
    axes[0].plot(z, 100 * dd, color=col, ls=ls, label=f"{p} = {v:g}")
    axes[1].plot(z, 100 * de, color=col, ls=ls)
    if p in ("Omega_m", "w0") and v in (0.35, -0.8):
        for zz in (0.3, 0.5):
            record(f"dDA_{p}_{v:g}_z{zz}", float(np.interp(zz, z, 100 * dd)))
axes[0].set_ylabel("ΔD_A / D_A [%]   (h⁻¹ Mpc)")
axes[1].set_ylabel("Δ[E(z) / E(0.9)] / [E(z) / E(0.9)] [%]")
for ax in axes:
    ax.set_xlabel("z")
    ax.axhline(0, color=INK2, lw=0.6)
axes[0].legend(ncol=2)
panel_label(axes[0], "apertures, background, mask")
panel_label(axes[1], "zred volume factor")
save(fig, "distances")
