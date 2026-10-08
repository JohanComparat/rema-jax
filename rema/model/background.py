"""Galaxy background of the matched filter, Sigma_g(z, chi^2, m).

Sigma_g is the surface density of galaxies per deg^2, per unit reference magnitude and per unit
chi^2, where chi^2 is computed against the red sequence at redshift z (redMaPPer CHISQBKG). It
is factorised as

    Sigma_g(z, chi^2, m) = N(m) P(chi^2 | z, m),

N(m) = counts(m) / (A_eff(m) dm), the galaxy density per magnitude (independent of z), and
P(chi^2 | z, m) the chi^2 distribution of those galaxies at redshift z, per unit chi^2. Sparse
(bright) magnitude bins borrow from their neighbours: P is a kernel-weighted ratio of counts
whose magnitude kernel widens until it holds ``min_counts`` galaxies. Galaxies enter the
histogram at z only if m < m*(z) - 2.5 log10(lmax_faint) (redMaPPer: L > 0.1 L*).

The table is looked up with trilinear interpolation (differentiable); chi^2 or m outside the
table give +inf (no background, i.e. not a member candidate).

:class:`ZredBkg` is redMaPPer's second background (ZREDBKG), Sigma_g(zred, m) per deg^2, per mag
and per unit zred, of all galaxies with a usable zred fit. The wcen centring model uses it for
the chance that a candidate central is a foreground or background galaxy.

The photo-z filter and ``rema pscd`` use a third table of the same shape, Sigma_pz(z, m)
(:func:`build_photoz_bkg`): the galaxies' photo-z distributions stacked, per deg^2, per mag and per
unit (true) z. It is the noise term N(m, z) of the AMICO matched filter (Bellagamba et al. 2018).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

from .likelihood import chisq as chisq_fn
from .redsequence import RSAt, RSModel


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ChisqBkg:
    z: jnp.ndarray            # [nz] grid points
    chisq: jnp.ndarray        # [nc] bin centres
    mag: jnp.ndarray          # [nm] bin centres
    sigma_g: jnp.ndarray      # [nz, nc, nm]  per deg^2 per mag per unit chi^2
    nm: jnp.ndarray           # [nm] N(m) per deg^2 per mag

    def lookup(self, z, chisq, mag):
        """Sigma_g at (z, chi^2, m) (broadcast); +inf outside the table."""
        z, chisq, mag = jnp.broadcast_arrays(jnp.asarray(z), jnp.asarray(chisq), jnp.asarray(mag))
        fz = (z - self.z[0]) / (self.z[1] - self.z[0])
        fc = (chisq - self.chisq[0]) / (self.chisq[1] - self.chisq[0])
        fm = (mag - self.mag[0]) / (self.mag[1] - self.mag[0])
        val = map_coordinates(self.sigma_g, [fz, fc, fm], order=1, mode="nearest")
        dc = 0.5 * (self.chisq[1] - self.chisq[0])
        dm = 0.5 * (self.mag[1] - self.mag[0])
        inside = ((chisq >= self.chisq[0] - dc) & (chisq < self.chisq[-1] + dc)
                  & (mag >= self.mag[0] - dm) & (mag < self.mag[-1] + dm))
        return jnp.where(inside & (val > 0), val, jnp.inf)

    def to_hdus(self):
        from ..io.tables import table_hdu
        from astropy.io import fits

        img = fits.ImageHDU(np.asarray(self.sigma_g, np.float32), name="BKG_CHISQ")
        img.header["BUNIT"] = "1/(deg2 mag chi2)"
        axes = table_hdu({"Z": np.asarray(self.z)}, extname="BKG_Z")
        axc = table_hdu({"CHISQ": np.asarray(self.chisq)}, extname="BKG_C")
        axm = table_hdu({"MAG": np.asarray(self.mag), "NM": np.asarray(self.nm)}, extname="BKG_M")
        return [img, axes, axc, axm]

    @classmethod
    def from_fits(cls, path) -> "ChisqBkg":
        from astropy.io import fits

        with fits.open(path) as h:
            sg = np.asarray(h["BKG_CHISQ"].data, np.float64)
            z = np.asarray(h["BKG_Z"].data["Z"], np.float64)
            c = np.asarray(h["BKG_C"].data["CHISQ"], np.float64)
            m = np.asarray(h["BKG_M"].data["MAG"], np.float64)
            nm = np.asarray(h["BKG_M"].data["NM"], np.float64)
        return cls(jnp.asarray(z), jnp.asarray(c), jnp.asarray(m), jnp.asarray(sg), jnp.asarray(nm))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ZredBkg:
    """Sigma_g(zred, m): galaxies per deg^2, per mag and per unit zred (redMaPPer ZREDBKG)."""

    zred: jnp.ndarray         # [nz] bin centres
    mag: jnp.ndarray          # [nm] bin centres
    sigma_g: jnp.ndarray      # [nz, nm]

    def lookup(self, zred, mag):
        """Sigma_g at (zred, m), bilinear (broadcast).

        As redMaPPer: +inf below the table (zred or m under the first bin edge, which includes a
        failed zred of -1) or for non-finite input; values above the table use its last bins.
        """
        zred, mag = jnp.broadcast_arrays(jnp.asarray(zred), jnp.asarray(mag))
        dz = self.zred[1] - self.zred[0]
        dm = self.mag[1] - self.mag[0]
        val = map_coordinates(self.sigma_g, [(zred - self.zred[0]) / dz, (mag - self.mag[0]) / dm],
                              order=1, mode="nearest")
        ok = (jnp.isfinite(zred) & jnp.isfinite(mag) & (zred >= self.zred[0] - 0.5 * dz)
              & (mag >= self.mag[0] - 0.5 * dm))
        return jnp.where(ok, val, jnp.inf)

    def to_hdus(self):
        from ..io.tables import table_hdu
        from astropy.io import fits

        img = fits.ImageHDU(np.asarray(self.sigma_g, np.float32), name="BKG_ZRED")
        img.header["BUNIT"] = "1/(deg2 mag zred)"
        axz = table_hdu({"ZRED": np.asarray(self.zred)}, extname="BKG_ZRED_Z")
        axm = table_hdu({"MAG": np.asarray(self.mag)}, extname="BKG_ZRED_M")
        return [img, axz, axm]

    @classmethod
    def from_fits(cls, path) -> "ZredBkg":
        from astropy.io import fits

        with fits.open(path) as h:
            sg = np.asarray(h["BKG_ZRED"].data, np.float64)
            z = np.asarray(h["BKG_ZRED_Z"].data["ZRED"], np.float64)
            m = np.asarray(h["BKG_ZRED_M"].data["MAG"], np.float64)
        return cls(jnp.asarray(z), jnp.asarray(m), jnp.asarray(sg))


def _cic_1d(x, centres):
    """Cloud-in-cell counts of ``x`` on regularly spaced ``centres`` (weight off the grid is lost)."""
    f = (np.asarray(x, np.float64) - centres[0]) / (centres[1] - centres[0])
    i0 = np.floor(f).astype(np.int64)
    w1 = f - i0
    out = np.zeros(centres.size)
    for i, w in ((i0, 1 - w1), (i0 + 1, w1)):
        ok = (i >= 0) & (i < centres.size)
        out += np.bincount(i[ok], weights=w[ok], minlength=centres.size)
    return out


def _cic_2d(x, y, cx, cy):
    """Cloud-in-cell counts of points (x, y) on the regular grid cx x cy: [nx, ny]."""
    fx = (np.asarray(x, np.float64) - cx[0]) / (cx[1] - cx[0])
    fy = (np.asarray(y, np.float64) - cy[0]) / (cy[1] - cy[0])
    ix, iy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    wx, wy = fx - ix, fy - iy
    out = np.zeros(cx.size * cy.size)
    for i, a in ((ix, 1 - wx), (ix + 1, wx)):
        for j, b in ((iy, 1 - wy), (iy + 1, wy)):
            ok = (i >= 0) & (i < cx.size) & (j >= 0) & (j < cy.size)
            out += np.bincount(i[ok] * cy.size + j[ok], weights=(a * b)[ok], minlength=out.size)
    return out.reshape(cx.size, cy.size)


def build_zred_bkg(zred, zred_chisq, refmag, area_fn, *, zred_range=(0.01, 1.0),
                   zredbinsize: float = 0.01, magbinsize: float = 0.2, mag_min: float = 12.0,
                   mag_max: float = 24.0, chisq_max: float = 100.0,
                   count_floor: float = 0.1) -> ZredBkg:
    """Zred background from a galaxy table (redMaPPer's ZredBackgroundGenerator).

    Galaxies with a valid zred fit (0 <= zred chi^2 < ``chisq_max``) are counted by cloud in cell
    in (zred, m), with no luminosity cut. Every cell is floored at ``count_floor`` galaxies and
    divided by the effective area ``area_fn(m)`` [deg^2] and the bin widths. Bins are labelled
    by their centres (redMaPPer labels them by their left edges, half a bin off).
    """
    zred = np.asarray(zred, np.float64)
    zred_chisq = np.asarray(zred_chisq, np.float64)
    refmag = np.asarray(refmag, np.float64)
    zedges = np.arange(zred_range[0], zred_range[1] + zredbinsize / 2, zredbinsize)
    medges = np.arange(mag_min, mag_max + magbinsize / 2, magbinsize)
    zc = 0.5 * (zedges[1:] + zedges[:-1])
    mc = 0.5 * (medges[1:] + medges[:-1])
    use = (np.isfinite(zred) & np.isfinite(refmag) & (zred_chisq >= 0) & (zred_chisq < chisq_max)
           & (zred > 0))
    counts = _cic_2d(zred[use], refmag[use], zc, mc)
    area = np.asarray([float(area_fn(m)) for m in mc])
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_g = np.where(area[None, :] > 0,
                           np.maximum(counts, count_floor) / (area[None, :] * magbinsize * zredbinsize),
                           np.inf)
    return ZredBkg(jnp.asarray(zc), jnp.asarray(mc), jnp.asarray(sigma_g))


def _kernel_ratio(num: np.ndarray, den: np.ndarray, centres: np.ndarray, sigmas, min_counts: float):
    """Kernel-smoothed num/den along the last axis, widening the kernel where den is sparse.

    num: [..., nm], den: [nm]. Returns num_s / den_s evaluated at each centre.
    """
    nm = centres.size
    out = np.zeros_like(num, dtype=np.float64)
    done = np.zeros(nm, bool)
    for s in sigmas:
        K = np.exp(-0.5 * ((centres[:, None] - centres[None, :]) / s) ** 2)     # [nm(target), nm(src)]
        dens = K @ den
        ok = (dens >= min_counts) & ~done
        if np.any(ok):
            out[..., ok] = (num @ K.T)[..., ok] / dens[ok]
            done |= ok
    if not np.all(done):
        K = np.ones((nm, nm))
        dens = K @ den
        rest = ~done & (dens > 0)
        out[..., rest] = (num @ K.T)[..., rest] / dens[rest]
    return out


def build_chisq_bkg(flux, ivar, refmag, rs: RSModel, mstar, area_fn, *, zrange, iref: int,
                    zbinsize: float = 0.02, chisqbinsize: float = 0.5, chisq_max: float = 20.0,
                    magbinsize: float = 0.2, mag_min: float = 12.0, mag_max: float = 24.0,
                    lmax_faint: float = 0.1, min_counts: float = 50.0,
                    sigmas=(0.2, 0.4, 0.8, 1.6, 3.2), smooth_z: float = 0.0,
                    mode: str = "lupt", eps: float = 0.015, chunk: int = 65536) -> ChisqBkg:
    """Histogram the galaxies' chi^2 against the red sequence on a grid of redshifts.

    ``area_fn(m)`` returns the effective area in deg^2 over which a galaxy of magnitude m is in
    the catalogue (see :meth:`rema.sky.maps.Footprint.effective_area`).
    """
    flux = np.asarray(flux, np.float32)
    ivar = np.asarray(ivar, np.float32)
    refmag = np.asarray(refmag, np.float64)
    zg = np.arange(zrange[0], zrange[1] + zbinsize / 2, zbinsize)
    cedges = np.arange(0.0, chisq_max + chisqbinsize / 2, chisqbinsize)
    medges = np.arange(mag_min, mag_max + magbinsize / 2, magbinsize)
    cc = 0.5 * (cedges[1:] + cedges[:-1])
    mc = 0.5 * (medges[1:] + medges[:-1])
    nc, nmb = cc.size, mc.size

    # N(m): all galaxies, linear (cloud-in-cell) assignment in magnitude.
    counts_m = _cic_1d(refmag, mc)
    area = np.asarray([float(area_fn(m)) for m in mc])
    nm = np.where(area > 0, counts_m / np.maximum(area, 1e-12) / magbinsize, 0.0)

    mstar_g = np.asarray(mstar(jnp.asarray(zg)))
    at_all = rs.at(jnp.asarray(zg))
    hist = np.zeros((zg.size, nc, nmb))
    den = np.zeros((zg.size, nmb))

    @jax.jit
    def chi2_at(f, iv, mean, slope, cint, pivot):
        at = RSAt(mean[None], slope[None], cint[None], pivot[None])
        return chisq_fn(f, iv, at, iref, mode, eps)[0]

    for k in range(zg.size):
        mlim = mstar_g[k] - 2.5 * np.log10(lmax_faint)
        use = np.flatnonzero(refmag < mlim)
        if use.size == 0:
            continue
        chis = np.empty(use.size)
        for lo in range(0, use.size, chunk):
            sl = use[lo:lo + chunk]
            pad = chunk - sl.size
            f = np.concatenate([flux[sl], np.ones((pad, flux.shape[1]), np.float32)])
            iv = np.concatenate([ivar[sl], np.ones((pad, flux.shape[1]), np.float32)])
            c = chi2_at(jnp.asarray(f), jnp.asarray(iv), at_all.mean[k], at_all.slope[k],
                        at_all.cint[k], at_all.pivot[k])
            chis[lo:lo + sl.size] = np.asarray(c)[:sl.size]
        m = refmag[use]
        den[k] = _cic_1d(m, mc)
        inc = chis < chisq_max
        # Bilinear (CIC) assignment in (chi^2, m).
        hist[k] = _cic_2d(chis[inc], m[inc], cc, mc)

    # P(chi^2 | z, m) per unit chi^2, kernel-smoothed in magnitude.
    p = np.zeros_like(hist)
    for k in range(zg.size):
        p[k] = _kernel_ratio(hist[k], den[k], mc, sigmas, min_counts) / chisqbinsize
    if smooth_z > 0:
        K = np.exp(-0.5 * ((zg[:, None] - zg[None, :]) / smooth_z) ** 2)
        K /= K.sum(axis=1, keepdims=True)
        p = np.einsum("kj,jcm->kcm", K, p)
    sigma_g = p * nm[None, None, :]
    return ChisqBkg(jnp.asarray(zg), jnp.asarray(cc), jnp.asarray(mc), jnp.asarray(sigma_g),
                    jnp.asarray(nm))


def build_photoz_bkg(zphot, zphot_e, refmag, area_fn, *, zrange=(0.0, 1.6), zbinsize: float = 0.005,
                     magbinsize: float = 0.2, mag_min: float = 12.0, mag_max: float = 24.0,
                     min_counts: float = 50.0, sigmas=(0.2, 0.4, 0.8, 1.6, 3.2),
                     chunk: int = 65536) -> ZredBkg:
    """Sigma_pz(z, m), the stacked photo-z distributions of a galaxy table.

    Every galaxy with a usable photo-z (``zphot_e`` > 0) contributes its Gaussian
    N(z; zphot, zphot_e), evaluated at the redshifts of the grid (spacing ``zbinsize`` over
    ``zrange``), to the magnitude bins of its reference magnitude (cloud in cell). As for the chi^2
    background, Sigma_pz = N(m) P(z | m): N(m) = counts / (A_eff(m) dm) and the mean distribution
    P(z | m) is a kernel-weighted average whose magnitude kernel widens until it holds
    ``min_counts`` galaxies. The table's ``zred`` axis holds the redshifts, so
    :meth:`ZredBkg.lookup` returns Sigma_pz(z, m) per deg^2 per mag per unit z.
    """
    zphot = np.asarray(zphot, np.float64)
    zphot_e = np.asarray(zphot_e, np.float64)
    refmag = np.asarray(refmag, np.float64)
    zg = np.arange(zrange[0], zrange[1] + zbinsize / 2, zbinsize)
    medges = np.arange(mag_min, mag_max + magbinsize / 2, magbinsize)
    mc = 0.5 * (medges[1:] + medges[:-1])
    use = np.flatnonzero(np.isfinite(zphot) & (zphot_e > 0) & np.isfinite(refmag))
    den = _cic_1d(refmag[use], mc)
    area = np.asarray([float(area_fn(m)) for m in mc])
    nm = np.where(area > 0, den / np.maximum(area, 1e-12) / magbinsize, 0.0)
    # num[k, b] = sum over the galaxies of magnitude bin b (CIC weights) of N(z_k; zphot, s).
    num = np.zeros((zg.size, mc.size))
    zg32 = zg.astype(np.float32)
    f = (refmag[use] - mc[0]) / magbinsize
    i0 = np.floor(f).astype(np.int64)
    w1 = f - i0
    for lo in range(0, use.size, chunk):
        sl = slice(lo, lo + chunk)
        zp = zphot[use][sl].astype(np.float32)
        s = zphot_e[use][sl].astype(np.float32)
        x = (zg32[None, :] - zp[:, None]) / s[:, None]
        pdf = np.exp(-0.5 * x * x) / (np.sqrt(2.0 * np.pi, dtype=np.float32) * s[:, None])
        for ib, w in ((i0[sl], 1.0 - w1[sl]), (i0[sl] + 1, w1[sl])):
            ok = (ib >= 0) & (ib < mc.size)
            # [nm, n] sparse assignment times [n, nz] distributions
            idx = np.flatnonzero(ok)
            order = np.argsort(ib[idx], kind="stable")
            idx = idx[order]
            bins, starts = np.unique(ib[idx], return_index=True)
            wp = pdf[idx] * w[idx, None].astype(np.float32)
            sums = np.add.reduceat(wp, starts, axis=0) if idx.size else np.zeros((0, zg.size))
            num[:, bins] += sums.T
    p = _kernel_ratio(num, den, mc, sigmas, min_counts)
    sigma_g = p * nm[None, :]
    sigma_g = np.where(area[None, :] > 0, sigma_g, np.inf)
    return ZredBkg(jnp.asarray(zg), jnp.asarray(mc), jnp.asarray(sigma_g))
