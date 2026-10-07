"""The finder's response to the cosmology, as a smooth function of (z, lambda).

From re-measurements at fixed centres (``rema remeasure``, :mod:`rema.modes.remeasure`), the
response of a catalogue made in the fiducial cosmology is

    R(theta; z, lambda) = ln lambda_theta - ln lambda_fid
                        = sum_p d1_p (theta_p - fid_p) + 1/2 d2_p (theta_p - fid_p)^2,

and similarly z_theta - z_fid = sum_p dz_p (theta_p - fid_p): the richness (redshift) the same
cluster would have if the catalogue had been made in cosmology theta. The coefficients are medians
over the clusters of each (z, lambda) bin, smoothed by a low-order polynomial in (z, ln lambda)
weighted by the counts, and tabulated on the bin centres. :class:`ResponseTable` is a pytree, so
:func:`delta_lnlam` and :func:`delta_z` can be evaluated inside a jitted likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..config import CosmologyConfig
from ..model.cosmo import PARAMS


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ResponseTable:
    """d1, d2, dz [n_param, n_z, n_lambda] on nodes (z, ln lambda); ``params`` and their fiducial."""

    z: jnp.ndarray
    lnlam: jnp.ndarray
    d1: jnp.ndarray
    d2: jnp.ndarray
    dz: jnp.ndarray
    params: tuple = field(default=(), metadata=dict(static=True))
    fiducial: tuple = field(default=(), metadata=dict(static=True))

    def dtheta(self, theta: Mapping[str, float]) -> jnp.ndarray:
        """theta - fiducial for the table's parameters (missing ones at the fiducial)."""
        return jnp.stack([jnp.asarray(theta.get(p, f), jnp.result_type(float)) - f
                          for p, f in zip(self.params, self.fiducial)]) if self.params else jnp.zeros(0)

    @classmethod
    def zero(cls, params: Sequence[str] = ("Omega_m",), fiducial: CosmologyConfig | None = None):
        """No response (the catalogue does not depend on the cosmology)."""
        fid = fiducial or CosmologyConfig()
        n = len(params)
        z = jnp.asarray([0.0, 2.0])
        ll = jnp.asarray([0.0, 10.0])
        zero = jnp.zeros((n, 2, 2))
        return cls(z, ll, zero, zero, zero, tuple(params), tuple(float(getattr(fid, p)) for p in params))

    @classmethod
    def constant(cls, d1: Mapping[str, float], fiducial: CosmologyConfig | None = None,
                 dz: Mapping[str, float] | None = None):
        """A response independent of (z, lambda), e.g. an analytic estimate."""
        fid = fiducial or CosmologyConfig()
        params = tuple(d1)
        one = jnp.ones((len(params), 2, 2))
        d1a = one * jnp.asarray([d1[p] for p in params])[:, None, None]
        dza = one * jnp.asarray([(dz or {}).get(p, 0.0) for p in params])[:, None, None]
        return cls(jnp.asarray([0.0, 2.0]), jnp.asarray([0.0, 10.0]), d1a, 0.0 * one, dza, params,
                   tuple(float(getattr(fid, p)) for p in params))

    # ------------------------------------------------------------------ FITS
    def write(self, path, header: Mapping | None = None):
        from ..io.tables import table_hdu, write_fits

        hdr = {"PARAMS": ",".join(self.params),
               **{f"FID{i}": v for i, v in enumerate(self.fiducial)}, **(header or {})}
        nodes = table_hdu({"Z": np.asarray(self.z)}, extname="Z_NODES")
        lnodes = table_hdu({"LNLAMBDA": np.asarray(self.lnlam)}, extname="LNLAMBDA_NODES")
        coef = [table_hdu({"D1": np.asarray(self.d1[i]), "D2": np.asarray(self.d2[i]),
                           "DZ": np.asarray(self.dz[i])}, extname=f"RESP_{p.upper()}")
                for i, p in enumerate(self.params)]
        return write_fits(path, [nodes, lnodes, *coef], hdr)

    @classmethod
    def read(cls, path) -> "ResponseTable":
        from astropy.io import fits

        with fits.open(path) as h:
            hdr = h[0].header
            params = tuple(p for p in hdr["PARAMS"].split(",") if p)
            fid = tuple(float(hdr[f"FID{i}"]) for i in range(len(params)))
            z = np.asarray(h["Z_NODES"].data["Z"], np.float64)
            ll = np.asarray(h["LNLAMBDA_NODES"].data["LNLAMBDA"], np.float64)
            arr = {k: np.stack([np.asarray(h[f"RESP_{p.upper()}"].data[k], np.float64) for p in params])
                   if params else np.zeros((0, z.size, ll.size)) for k in ("D1", "D2", "DZ")}
        return cls(jnp.asarray(z), jnp.asarray(ll), jnp.asarray(arr["D1"]), jnp.asarray(arr["D2"]),
                   jnp.asarray(arr["DZ"]), params, fid)


def _bilinear(tab: jnp.ndarray, xn, yn, x, y):
    """Bilinear interpolation of tab [nx, ny] on nodes (xn, yn), clamped at the edges."""
    def frac(nodes, v):
        i = jnp.clip(jnp.searchsorted(nodes, v) - 1, 0, nodes.shape[0] - 2)
        t = jnp.clip((v - nodes[i]) / (nodes[i + 1] - nodes[i]), 0.0, 1.0)
        return i, t

    i, tx = frac(xn, x)
    j, ty = frac(yn, y)
    return ((1 - tx) * (1 - ty) * tab[i, j] + tx * (1 - ty) * tab[i + 1, j]
            + (1 - tx) * ty * tab[i, j + 1] + tx * ty * tab[i + 1, j + 1])


def delta_lnlam(tab: ResponseTable, lnlam, z, dtheta) -> jnp.ndarray:
    """ln lambda_theta - ln lambda_fid at (lnlam, z) (broadcast) for ``dtheta`` = theta - fid [P]."""
    lnlam, z = jnp.broadcast_arrays(jnp.asarray(lnlam), jnp.asarray(z))
    out = jnp.zeros(lnlam.shape)
    for k in range(len(tab.params)):
        d1 = _bilinear(tab.d1[k], tab.z, tab.lnlam, z, lnlam)
        d2 = _bilinear(tab.d2[k], tab.z, tab.lnlam, z, lnlam)
        out = out + d1 * dtheta[k] + 0.5 * d2 * dtheta[k] ** 2
    return out


def delta_z(tab: ResponseTable, lnlam, z, dtheta) -> jnp.ndarray:
    """z_theta - z_fid at (lnlam, z) for ``dtheta`` [P]."""
    lnlam, z = jnp.broadcast_arrays(jnp.asarray(lnlam), jnp.asarray(z))
    out = jnp.zeros(lnlam.shape)
    for k in range(len(tab.params)):
        out = out + _bilinear(tab.dz[k], tab.z, tab.lnlam, z, lnlam) * dtheta[k]
    return out


# --------------------------------------------------------------------------- from re-measurements
_KEYS = {"Omega_m": "OMEGAM", "h": "HUBBLE", "Omega_b": "OMEGAB", "sum_mnu": "MNU", "w0": "W0", "wa": "WA"}

def _variations(labels: Sequence[str], cosmo: Mapping[str, np.ndarray]) -> dict[str, list[tuple[int, float]]]:
    """{param: [(column, theta - fid), ...]} of the one-at-a-time variations (column 0 = fiducial)."""
    keys = _KEYS
    out: dict[str, list[tuple[int, float]]] = {}
    for j in range(1, len(labels)):
        diff = [p for p in PARAMS if not np.isclose(cosmo[keys[p]][j], cosmo[keys[p]][0], rtol=0, atol=1e-12)]
        if len(diff) == 1:
            p = diff[0]
            out.setdefault(p, []).append((j, float(cosmo[keys[p]][j] - cosmo[keys[p]][0])))
    return out


def _read_many(paths) -> tuple[dict, list[str], dict]:
    """Concatenated CLUSTERS of ``rema remeasure`` files, their labels and COSMOLOGIES columns
    (the files must share the cosmology grid)."""
    from ..io.tables import read_table

    cats, labels, cosmo = [], None, None
    for path in [paths] if isinstance(paths, (str, bytes)) or hasattr(paths, "__fspath__") else paths:
        cos = read_table(path, hdu="COSMOLOGIES")
        lab = [x.decode() if isinstance(x, bytes) else str(x) for x in cos["LABEL"]]
        if labels is None:
            labels, cosmo = lab, cos
        elif lab != labels:
            raise ValueError(f"{path}: cosmologies {lab} differ from {labels}")
        cats.append(read_table(path, ["LAMBDA", "Z_LAMBDA"], hdu="CLUSTERS"))
    if not cats:
        raise ValueError("no re-measurement file")
    return {k: np.concatenate([c[k] for c in cats]) for k in cats[0]}, labels, cosmo


def per_cluster_coefficients(lnlam: np.ndarray, z: np.ndarray, var: Sequence[tuple[int, float]]):
    """(d1, d2, dz) per cluster from ln lambda [N, C] and z [N, C] at the variations ``var``.

    One variation: d1 = Delta ln lambda / Delta theta, d2 = 0. Two or more: least squares of
    Delta ln lambda = d1 Delta theta + d2 Delta theta^2 / 2 (exact for a symmetric pair).
    """
    cols = [j for j, _ in var]
    h = np.array([d for _, d in var])
    dl = lnlam[:, cols] - lnlam[:, [0]]
    dzz = z[:, cols] - z[:, [0]]
    if h.size == 1:
        return dl[:, 0] / h[0], np.zeros(dl.shape[0]), dzz[:, 0] / h[0]
    A = np.stack([h, 0.5 * h**2], axis=1)
    pinv = np.linalg.pinv(A)
    d1, d2 = pinv @ dl.T
    dz = np.linalg.pinv(h[:, None]) @ dzz.T
    return d1, d2, dz[0]


def from_remeasure(paths: Iterable, z_edges: Sequence[float], lam_edges: Sequence[float], *,
                   min_count: int = 10, degree: tuple[int, int] = (2, 1)) -> tuple[ResponseTable, dict]:
    """Response table from ``rema remeasure`` files of the same cosmology grid.

    Clusters are binned by their fiducial (z_lambda, lambda); per bin the median of the per-cluster
    coefficients (and their NMAD) is taken, and a polynomial of degree ``degree`` in
    (z - 0.4, ln lambda - ln 30), weighted by the counts of the bins with at least ``min_count``
    clusters, gives the tabulated values at the bin centres. Returns (table, binned statistics).
    """
    cat, labels, cosmo = _read_many(paths)
    var = _variations(labels, cosmo)
    if not var:
        raise ValueError(f"no one-at-a-time variation among the cosmologies {labels}")
    lam0 = cat["LAMBDA"][:, 0]
    z0 = cat["Z_LAMBDA"][:, 0]
    good = (np.all(cat["LAMBDA"] > 0, axis=1) & np.all(cat["Z_LAMBDA"] > 0, axis=1)
            & np.isfinite(lam0) & np.isfinite(z0))
    lnlam = np.log(np.where(cat["LAMBDA"] > 0, cat["LAMBDA"], 1.0))
    ze, le = np.asarray(z_edges, np.float64), np.asarray(lam_edges, np.float64)
    zc, lc = 0.5 * (ze[1:] + ze[:-1]), np.sqrt(le[1:] * le[:-1])
    iz = np.digitize(z0, ze) - 1
    il = np.digitize(lam0, le) - 1
    inside = good & (iz >= 0) & (iz < zc.size) & (il >= 0) & (il < lc.size)
    params = tuple(p for p in PARAMS if p in var)
    shape = (len(params), zc.size, lc.size)
    stats = {k: np.full(shape, np.nan) for k in ("D1", "D2", "DZ", "D1_NMAD")}
    stats["N"] = np.zeros((zc.size, lc.size), np.int64)
    out = {k: np.zeros(shape) for k in ("D1", "D2", "DZ")}
    for a in range(zc.size):
        for b in range(lc.size):
            stats["N"][a, b] = np.sum(inside & (iz == a) & (il == b))
    zz, ll = np.meshgrid(zc - 0.4, np.log(lc) - np.log(30.0), indexing="ij")
    terms = [zz**i * ll**j for i in range(degree[0] + 1) for j in range(degree[1] + 1)]
    for k, p in enumerate(params):
        d1, d2, dz = per_cluster_coefficients(lnlam, np.where(cat["Z_LAMBDA"] > 0, cat["Z_LAMBDA"], 0.0), var[p])
        for a in range(zc.size):
            for b in range(lc.size):
                sel = inside & (iz == a) & (il == b)
                if sel.sum() >= min_count:
                    stats["D1"][k, a, b] = np.median(d1[sel])
                    stats["D2"][k, a, b] = np.median(d2[sel])
                    stats["DZ"][k, a, b] = np.median(dz[sel])
                    stats["D1_NMAD"][k, a, b] = 1.4826 * np.median(np.abs(d1[sel] - np.median(d1[sel])))
        ok = np.isfinite(stats["D1"][k])
        if not ok.any():
            raise ValueError(f"{p}: no (z, lambda) bin with {min_count} clusters")
        w = np.sqrt(stats["N"][ok])
        A = np.stack([t[ok] for t in terms], axis=1) * w[:, None]
        nterm = min(A.shape[1], int(ok.sum()))
        for key in ("D1", "D2", "DZ"):
            coef = np.linalg.lstsq(A[:, :nterm], stats[key][k][ok] * w, rcond=None)[0]
            out[key][k] = sum(c * t for c, t in zip(coef, terms[:nterm]))
    fid = tuple(float(cosmo[_KEYS[p]][0]) for p in params)
    tab = ResponseTable(jnp.asarray(zc), jnp.asarray(np.log(lc)), jnp.asarray(out["D1"]),
                        jnp.asarray(out["D2"]), jnp.asarray(out["DZ"]), params, fid)
    stats.update(z=zc, lam=lc, params=params)
    return tab, stats
