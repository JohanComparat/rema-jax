"""Diagnostic plots of a calibration (matplotlib, optional dependency)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..calib.init import observed_colours
from ..model.redsequence import RSModel


def plot_redsequence(rs: RSModel, flux, ivar, z, w, refmag, path, zrange=(0.05, 0.9)):
    """Members' colours (at the pivot magnitude) vs redshift, with the model mean and scatter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col, _ = observed_colours(np.asarray(flux, float), np.asarray(ivar, float), rs.iref)
    at = rs.at(jnp.asarray(np.asarray(z, np.float32)))
    col_piv = col - np.asarray(at.slope) * (np.asarray(refmag) - np.asarray(at.pivot))[:, None]
    zz = np.linspace(*zrange, 200)
    m = rs.at(jnp.asarray(zz))
    mean = np.asarray(m.mean)
    sig = np.sqrt(np.diagonal(np.asarray(m.cint), axis1=1, axis2=2))
    names = [f"{a}-{b}" for a, b in zip(rs.bands[:-1], rs.bands[1:])]
    fig, axes = plt.subplots(rs.ncol, 1, figsize=(6, 2.6 * rs.ncol), sharex=True)
    axes = np.atleast_1d(axes)
    for j, ax in enumerate(axes):
        ax.scatter(z, col_piv[:, j], s=2, c=np.clip(w, 0, 1), cmap="Greys", vmin=0, vmax=1, rasterized=True)
        ax.plot(zz, mean[:, j], color="C3")
        ax.fill_between(zz, mean[:, j] - sig[:, j], mean[:, j] + sig[:, j], color="C3", alpha=0.25)
        ax.plot(np.asarray(rs.z_mean), np.asarray(rs.mean)[:, j], "o", color="C3", ms=3)
        lo, hi = np.percentile(mean[:, j], [0, 100])
        ax.set_ylim(lo - 0.6, hi + 0.6)
        ax.set_ylabel(names[j])
    axes[-1].set_xlabel("z")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def plot_zlambda(zspec, zlambda, zlambda_e, path):
    """z_lambda vs spectroscopic redshift of the calibration clusters."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zs, zl, ze = (np.asarray(a, float) for a in (zspec, zlambda, zlambda_e))
    ok = (zs > 0) & (zl > 0)
    dz = (zl - zs) / (1 + zs)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].errorbar(zs[ok], zl[ok], yerr=ze[ok], fmt=".", ms=2, alpha=0.5)
    axes[0].plot([0, 1], [0, 1], "k-", lw=0.5)
    axes[0].set_ylabel("z_lambda")
    axes[1].plot(zs[ok], dz[ok], ".", ms=2, alpha=0.5)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylim(-0.05, 0.05)
    axes[1].set_xlabel("z_spec")
    axes[1].set_ylabel("dz/(1+z)")
    nmad = 1.4826 * np.median(np.abs(dz[ok] - np.median(dz[ok]))) if ok.any() else np.nan
    axes[0].set_title(f"N={ok.sum()}  bias={np.median(dz[ok]):+.4f}  NMAD={nmad:.4f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)
