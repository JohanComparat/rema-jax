"""Red-sequence model: colours as a function of redshift and reference magnitude.

For adjacent colours c_j = m_j - m_{j+1} (bands ordered blue to red) the model is

    c_j(z, m) = mean_j(z) + slope_j(z) (m - pivot(z)),

with Gaussian intrinsic scatter of covariance C_int(z) = D R D, D = diag(sigma_j(z)). Every
quantity is a natural cubic spline through redshift nodes (:mod:`rema.model.splines`). The
parameters are unconstrained so that fits can move them freely:

- ``log_sigma`` (sigma = exp, floored at ``min_sigma``),
- ``corr``: hyperbolic arctangents of the partial correlations of the canonical partial
  correlation parametrisation, so R is a correlation matrix for any parameter value.

:class:`RSModel` is a pytree whose leaves are the node values; node positions are static.

>>> rs = RSModel.from_template(bands=("g", "r", "i", "z"), ref_band="z")
>>> at = rs.at(0.5)
>>> at.mean.shape, at.cint.shape
((3,), (3, 3))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .splines import NaturalSpline


@lru_cache(maxsize=64)
def _spline(nodes: tuple[float, ...]) -> NaturalSpline:
    return NaturalSpline(np.asarray(nodes))


def npairs(ncol: int) -> int:
    return ncol * (ncol - 1) // 2


def corr_from_partial(z):
    """Correlation matrix from partial correlations ``z[..., npairs]`` in (-1, 1).

    Pairs are ordered (1,0), (2,0), (2,1), (3,0), ... (row-major lower triangle).
    """
    z = jnp.asarray(z)
    m = z.shape[-1]
    n = int(round((1 + np.sqrt(1 + 8 * m)) / 2))
    L = [[None] * n for _ in range(n)]
    k = 0
    one = jnp.ones(z.shape[:-1], dtype=z.dtype)
    zero = jnp.zeros(z.shape[:-1], dtype=z.dtype)
    L[0][0] = one
    for i in range(1, n):
        rem = one
        for j in range(i):
            L[i][j] = z[..., k] * jnp.sqrt(rem)
            rem = rem - L[i][j] ** 2
            k += 1
        L[i][i] = jnp.sqrt(jnp.maximum(rem, 1e-12))
    for i in range(n):
        for j in range(i + 1, n):
            L[i][j] = zero
    Lm = jnp.stack([jnp.stack(row, axis=-1) for row in L], axis=-2)
    LT = jnp.swapaxes(Lm, -1, -2)
    return jnp.sum(Lm[..., :, :, None] * LT[..., None, :, :], axis=-2)


def partial_from_corr(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`corr_from_partial` for one correlation matrix (numpy)."""
    L = np.linalg.cholesky(np.asarray(R, dtype=np.float64))
    n = L.shape[0]
    out = []
    for i in range(1, n):
        rem = 1.0
        for j in range(i):
            out.append(L[i, j] / np.sqrt(rem))
            rem -= L[i, j] ** 2
    return np.asarray(out)


class RSAt(NamedTuple):
    """Red-sequence model evaluated at some redshift(s)."""

    mean: jnp.ndarray      # [..., ncol]
    slope: jnp.ndarray     # [..., ncol]
    cint: jnp.ndarray      # [..., ncol, ncol]
    pivot: jnp.ndarray     # [...]

    def colours(self, refmag):
        """Model colours at reference magnitude ``refmag``."""
        return self.mean + self.slope * (refmag - self.pivot)[..., None]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RSModel:
    mean: jnp.ndarray          # [n_mean, ncol]
    slope: jnp.ndarray         # [n_slope, ncol]
    log_sigma: jnp.ndarray     # [n_sigma, ncol]
    corr: jnp.ndarray          # [n_corr, npairs]  (atanh of partial correlations)
    pivot: jnp.ndarray         # [n_pivot]
    z_mean: tuple = field(metadata=dict(static=True))
    z_slope: tuple = field(metadata=dict(static=True))
    z_sigma: tuple = field(metadata=dict(static=True))
    z_corr: tuple = field(metadata=dict(static=True))
    z_pivot: tuple = field(metadata=dict(static=True))
    bands: tuple = field(default=("g", "r", "i", "z"), metadata=dict(static=True))
    ref_band: str = field(default="z", metadata=dict(static=True))
    min_sigma: float = field(default=0.01, metadata=dict(static=True))

    # ------------------------------------------------------------------ shape
    @property
    def nband(self) -> int:
        return len(self.bands)

    @property
    def ncol(self) -> int:
        return self.nband - 1

    @property
    def iref(self) -> int:
        return self.bands.index(self.ref_band)

    def band_offsets_matrix(self) -> tuple[np.ndarray, list[int]]:
        """T with (m_b - m_ref) = T @ c for the non-reference bands, and their indices."""
        return band_offsets_matrix(self.nband, self.iref)

    # ------------------------------------------------------------------ evaluation
    def at(self, z) -> RSAt:
        z = jnp.asarray(z)
        mean = _spline(self.z_mean).eval_jax(z, self.mean)
        slope = _spline(self.z_slope).eval_jax(z, self.slope)
        sig = jnp.maximum(jnp.exp(_spline(self.z_sigma).eval_jax(z, self.log_sigma)), self.min_sigma)
        if self.ncol > 1:
            pc = jnp.tanh(_spline(self.z_corr).eval_jax(z, self.corr))
            R = corr_from_partial(pc)
        else:
            R = jnp.ones(z.shape + (1, 1), dtype=sig.dtype)
        cint = sig[..., :, None] * R * sig[..., None, :]
        pivot = _spline(self.z_pivot).eval_jax(z, self.pivot)
        return RSAt(mean, slope, cint, pivot)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_arrays(cls, nodes: dict, values: dict, bands, ref_band, min_sigma=0.01) -> "RSModel":
        return cls(mean=jnp.asarray(values["mean"]), slope=jnp.asarray(values["slope"]),
                   log_sigma=jnp.asarray(values["log_sigma"]), corr=jnp.asarray(values["corr"]),
                   pivot=jnp.asarray(values["pivot"]),
                   z_mean=tuple(map(float, nodes["mean"])), z_slope=tuple(map(float, nodes["slope"])),
                   z_sigma=tuple(map(float, nodes["sigma"])), z_corr=tuple(map(float, nodes["corr"])),
                   z_pivot=tuple(map(float, nodes["pivot"])), bands=tuple(bands),
                   ref_band=ref_band, min_sigma=float(min_sigma))

    @classmethod
    def from_redmapper_pars(cls, path: str, mean_nodes=None) -> "RSModel":
        """Read a redMaPPer ``*_pars.fit`` file (colour, slope, sigma/correlation, pivot nodes).

        Colours with different node sets are resampled onto common ``mean_nodes`` (default:
        the union of the colour nodes) by evaluating redMaPPer's natural splines there.
        """
        from astropy.io import fits

        with fits.open(path) as h:
            d, hdr = h[1].data, h[1].header
            ncol = int(hdr["NCOL"])
            bands = tuple(b.strip() for b in hdr["BANDS"].split(","))
            ref_band = bands[int(hdr["REF_IND"])]
            names = {c.lower(): c for c in d.columns.names}

            def col(n):
                return np.asarray(d[names[n]][0], dtype=np.float64)

            cz = [col(f"z{j:02d}") for j in range(ncol)]
            cv = [col(f"c{j:02d}") for j in range(ncol)]
            sz = [col(f"zs{j:02d}") for j in range(ncol)]
            sv = [col(f"slope{j:02d}") for j in range(ncol)]
            covz = col("covmat_z")
            sigma = col("sigma")                        # [ncol, ncol, nnode]
            pz, pv = col("pivotmag_z"), col("pivotmag")
        zm = np.unique(np.concatenate(cz)) if mean_nodes is None else np.asarray(mean_nodes)
        mean = np.stack([NaturalSpline(cz[j])(zm, cv[j]) for j in range(ncol)], axis=1)
        zsl = np.unique(np.concatenate(sz))
        slope = np.stack([NaturalSpline(sz[j])(zsl, sv[j]) for j in range(ncol)], axis=1)
        diag = np.stack([sigma[j, j, :] for j in range(ncol)], axis=1)
        log_sigma = np.log(np.maximum(diag, 1e-3))
        corr = np.zeros((covz.size, npairs(ncol)))
        for k in range(covz.size):
            R = np.eye(ncol)
            for i in range(ncol):
                for j in range(i):
                    R[i, j] = R[j, i] = np.clip(sigma[i, j, k], -0.99, 0.99)
            if ncol > 1:
                corr[k] = np.arctanh(partial_from_corr(R))
        return cls.from_arrays(
            nodes={"mean": zm, "slope": zsl, "sigma": covz, "corr": covz, "pivot": pz},
            values={"mean": mean, "slope": slope, "log_sigma": log_sigma, "corr": corr,
                    "pivot": pv}, bands=bands, ref_band=ref_band)

    @classmethod
    def from_template(cls, bands=("g", "r", "i", "z"), ref_band="z", template="bc03_legacy_grizw1",
                      template_bands=("g", "r", "i", "z", "w1"), zrange=(0.05, 0.95),
                      dz_mean=0.05, dz_other=0.1, sigma=0.05, mstar="des_z03",
                      pivot_offset=0.5) -> "RSModel":
        """Initial model from a template of adjacent colours (zero slope, constant scatter).

        The pivot magnitude is m*(z) + ``pivot_offset``.
        """
        from importlib import resources

        from astropy.io import fits

        from .profiles import MStar

        path = template if template.endswith((".fits", ".fit")) else str(
            resources.files("rema.data").joinpath(f"colors_{template}.fits"))
        with fits.open(path) as h:
            tz = np.asarray(h[1].data["Z"], np.float64)
            tc = np.asarray(h[1].data["COLOR"], np.float64)
        # Colours between our adjacent bands from the template's adjacent colours.
        tb = list(template_bands)
        ncol = len(bands) - 1
        cols = np.zeros((tz.size, ncol))
        for j in range(ncol):
            a, b = tb.index(bands[j]), tb.index(bands[j + 1])
            cols[:, j] = tc[:, a:b].sum(axis=1)
        zm = np.arange(zrange[0], zrange[1] + dz_mean / 2, dz_mean)
        zo = np.arange(zrange[0], zrange[1] + dz_other / 2, dz_other)
        mean = np.stack([np.interp(zm, tz, cols[:, j]) for j in range(ncol)], axis=1)
        ms = MStar(mstar)
        return cls.from_arrays(
            nodes={"mean": zm, "slope": zo, "sigma": zo, "corr": zo, "pivot": zo},
            values={"mean": mean, "slope": np.zeros((zo.size, ncol)),
                    "log_sigma": np.full((zo.size, ncol), np.log(sigma)),
                    "corr": np.zeros((zo.size, npairs(ncol))),
                    "pivot": np.asarray(ms(zo)) + pivot_offset},
            bands=bands, ref_band=ref_band)

    # ------------------------------------------------------------------ io
    def to_hdus(self):
        from ..io.tables import table_hdu

        hdus = []
        for name, z, v in (("MEAN", self.z_mean, self.mean), ("SLOPE", self.z_slope, self.slope),
                           ("LOG_SIGMA", self.z_sigma, self.log_sigma),
                           ("CORR", self.z_corr, self.corr), ("PIVOT", self.z_pivot, self.pivot)):
            hdus.append(table_hdu({"Z": np.asarray(z), "VALUE": np.asarray(v, dtype=np.float64)},
                                  {"BANDS": ",".join(self.bands), "REFBAND": self.ref_band,
                                   "MINSIG": self.min_sigma}, f"RS_{name}"))
        return hdus

    @classmethod
    def from_fits(cls, path) -> "RSModel":
        from astropy.io import fits

        nodes, values = {}, {}
        with fits.open(path) as h:
            for key, ext in (("mean", "RS_MEAN"), ("slope", "RS_SLOPE"), ("sigma", "RS_LOG_SIGMA"),
                             ("corr", "RS_CORR"), ("pivot", "RS_PIVOT")):
                d = h[ext].data
                nodes[key] = np.asarray(d["Z"], np.float64)
                values["log_sigma" if key == "sigma" else key] = np.asarray(d["VALUE"], np.float64)
            hdr = h["RS_MEAN"].header
            bands = tuple(hdr["BANDS"].split(","))
            ref, minsig = hdr["REFBAND"], float(hdr["MINSIG"])
        # One column per node comes back flat (two colours: one correlation coefficient).
        ncol = len(bands) - 1
        for key, width in (("mean", ncol), ("slope", ncol), ("log_sigma", ncol),
                           ("corr", ncol * (ncol - 1) // 2)):
            values[key] = values[key].reshape(-1, width)
        return cls.from_arrays(nodes, values, bands, ref, minsig)


def band_offsets_matrix(nband: int, iref: int) -> tuple[np.ndarray, list[int]]:
    """Matrix T [nband-1, ncol] with m_b - m_ref = sum_j T[b, j] c_j for b != ref.

    >>> T, others = band_offsets_matrix(4, 3)
    >>> others, T.tolist()
    ([0, 1, 2], [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    """
    ncol = nband - 1
    others = [b for b in range(nband) if b != iref]
    T = np.zeros((len(others), ncol))
    for row, b in enumerate(others):
        if b < iref:
            T[row, b:iref] = 1.0
        else:
            T[row, iref:b] = -1.0
    return T, others


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ZredCorrection:
    """zred bias corrections: dz = corr(z) + (m - pivot) corr_slope(z); zred_e is scaled by corr_r(z)."""

    corr: jnp.ndarray
    corr_slope: jnp.ndarray
    corr_r: jnp.ndarray
    z_corr: jnp.ndarray
    z_slope: jnp.ndarray

    @classmethod
    def identity(cls) -> "ZredCorrection":
        z = jnp.asarray([0.0, 2.0])
        return cls(jnp.zeros(2), jnp.zeros(2), jnp.ones(2), z, z)

    @classmethod
    def from_redmapper_pars(cls, path: str) -> "ZredCorrection":
        from astropy.io import fits

        with fits.open(path) as h:
            d = h[1].data
            names = {c.lower(): c for c in d.columns.names}

            def col(n):
                return jnp.asarray(np.asarray(d[names[n]][0], np.float64))

            return cls(col("corr"), col("corr_slope"), col("corr_r"), col("corr_z"),
                       col("corr_slope_z"))

    def apply(self, zred_u, zred_u_e, refmag, pivot_fn, niter: int = 5):
        """Corrected (zred, zred_e) from uncorrected values (redMaPPer ``compute_zreds``)."""
        piv = pivot_fn(zred_u)
        dz = jnp.zeros_like(zred_u)
        for _ in range(niter):
            zz = zred_u + dz
            dz = (jnp.interp(zz, self.z_corr, self.corr)
                  + (refmag - piv) * jnp.interp(zz, self.z_slope, self.corr_slope))
        zred = zred_u + dz
        return zred, zred_u_e * jnp.interp(zred, self.z_slope, self.corr_r)
