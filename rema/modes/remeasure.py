"""Re-measurement of catalogued clusters at fixed centres, in other cosmologies.

The cluster finder depends on the cosmology through the angular diameter distance D_A(z) in
h^-1 Mpc (every aperture, the background per (h^-1 Mpc)^2, the mask quadrature) and through E(z)
in zred (redMaPPer's volume factor); see ``docs/cosmology_sensitivity.rst``. This module measures
that response on a catalogue, cluster by cluster, keeping what the catalogue decided:

- the centre (ID_CENT[:, 0]) and the starting redshift (Z_LAMBDA_RAW);
- optionally the percolation, through the free fractions of the neighbours rebuilt from the
  members table (:class:`MemberPfree`): exact for the members, from the claims of the
  higher-ranked clusters' members otherwise. Without it every neighbour is free (pfree = 1).

For each cosmology the region is rebuilt with its distances and zred
(:meth:`rema.modes.common.Region.with_cosmology`), z_lambda is iterated from Z_LAMBDA_RAW in the
percolation aperture and lambda measured there, as in the blind mode's percolation step; the
calibration (red sequence, backgrounds, corrections) stays that of the catalogue. ``fixed_z``
measures lambda at the starting redshift instead, which isolates the aperture effect.

:func:`response_jvp` gives the derivatives d ln(lambda)/d(theta) at fixed redshift by forward-mode
autodiff through the distance table (:meth:`rema.model.cosmo.CosmoTable.jvp`), with
d ln(lambda)/dz for the redshift term. The aperture completeness (mask, depth) is computed outside
the differentiated function, so its change with D_A is left out of the derivatives (the
re-measurements include it).

Seeds that are not in the galaxy table (a catalogue from another ingest) are skipped with a
warning.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..config import CosmologyConfig
from ..core.richness import RadialQuad, Stage, richness
from ..model.cosmo import PARAMS, CosmoTable
from .blind import _iter_batches, _run_batched
from .common import Region

log = logging.getLogger(__name__)

#: Columns of :func:`remeasure`, per cosmology.
COLUMNS = ("LAMBDA", "LAMBDA_E", "Z_LAMBDA", "Z_LAMBDA_E", "Z_LAMBDA_RAW", "R_LAMBDA", "SCALEVAL",
           "MASKFRAC", "LNLAMLIKE")


@dataclass
class Centres:
    """Clusters to re-measure: catalogue rows, galaxy-table rows of their centres, start redshifts."""

    rows: np.ndarray
    gi: np.ndarray
    z0: np.ndarray

    @property
    def size(self) -> int:
        return int(self.rows.size)


def centres_from_catalog(region: Region, cat: Mapping[str, np.ndarray],
                         rows: np.ndarray | None = None) -> Centres:
    """Centres (ID_CENT[:, 0]) and start redshifts (Z_LAMBDA_RAW, else Z_LAMBDA) of catalogue rows."""
    ids = np.asarray(cat["ID_CENT"])
    ids = ids[:, 0] if ids.ndim == 2 else ids
    z0 = np.asarray(cat["Z_LAMBDA_RAW"] if "Z_LAMBDA_RAW" in cat else cat["Z_LAMBDA"], np.float64)
    rows = np.arange(ids.size) if rows is None else np.asarray(rows, np.int64)
    gid = np.asarray(region.gal["ID"])
    order = np.argsort(gid, kind="stable")
    pos = np.clip(np.searchsorted(gid, ids[rows], sorter=order), 0, max(gid.size - 1, 0))
    found = gid[order[pos]] == ids[rows] if gid.size else np.zeros(rows.size, bool)
    if not np.all(found):
        log.warning("%d of %d centres are not in the galaxy table; skipped", int((~found).sum()),
                    rows.size)
    return Centres(rows=rows[found], gi=order[pos[found]], z0=z0[rows[found]])


def stage_and_quad(cfg) -> tuple[Stage, RadialQuad]:
    """The percolation aperture and its radial quadrature (as the blind mode's final step)."""
    rc = cfg.richness
    st = Stage.make(rc.percolation.r0, rc.percolation.beta, rc.maxrad_factor)
    q = RadialQuad.make(rmax=rc.percolation.r0 * 20.0 ** rc.percolation.beta + 5 * cfg.model.rsig)
    return st, q


class MemberPfree:
    """Free fractions of the neighbours of catalogued clusters, from the members table.

    The percolation gives galaxy i, when cluster k is measured, pfree = 1 - (sum of p of the
    higher-ranked clusters that claimed it), ranked by decreasing LNLIKE (ties by SEED_ID). The
    members table keeps that pfree for the members of k (PFREE); for its other neighbours the
    claims are summed from the members (P) of the clusters ranked above k. Claims of galaxies that
    are not members of the claiming cluster (beyond its r_lambda, within its R_MASK) are not in
    the table and are missed.

    ``cat``: the catalogue (MEM_MATCH_ID, LNLIKE, SEED_ID), whose rows ``centres.rows`` index;
    ``mem``: its members (MEM_MATCH_ID, ID, P, PFREE).
    """

    def __init__(self, region: Region, cat: Mapping[str, np.ndarray], mem: Mapping[str, np.ndarray],
                 centres: Centres):
        lnlike = np.asarray(cat["LNLIKE"], np.float64)
        ncl = lnlike.size
        seed = np.asarray(cat["SEED_ID"]) if "SEED_ID" in cat else np.arange(ncl)
        rank = np.empty(ncl, np.int64)
        rank[np.lexsort((seed, -lnlike))] = np.arange(ncl)
        mm = np.asarray(cat["MEM_MATCH_ID"], np.int64)
        om = np.argsort(mm, kind="stable")
        k = np.clip(np.searchsorted(mm, np.asarray(mem["MEM_MATCH_ID"], np.int64), sorter=om), 0, ncl - 1)
        ci = om[k]
        ok = mm[ci] == np.asarray(mem["MEM_MATCH_ID"], np.int64)
        gid = np.asarray(region.gal["ID"])
        og = np.argsort(gid, kind="stable")
        kg = np.clip(np.searchsorted(gid, np.asarray(mem["ID"]), sorter=og), 0, gid.size - 1)
        gr = og[kg]
        ok &= gid[gr] == np.asarray(mem["ID"])
        ci, gr = ci[ok], gr[ok].astype(np.int64)
        self.ngal, self.ncl = np.int64(gid.size), np.int64(ncl)
        self.rank = rank
        self.cat_rows = np.asarray(centres.rows, np.int64)
        ek = ci * self.ngal + gr
        oe = np.argsort(ek, kind="stable")
        self.exact_key, self.exact_val = ek[oe], np.asarray(mem["PFREE"], np.float64)[ok][oe]
        ck = gr * (self.ncl + 1) + rank[ci]
        oc = np.argsort(ck, kind="stable")
        self.claim_key = ck[oc]
        self.claim_cs = np.concatenate([[0.0], np.cumsum(np.asarray(mem["P"], np.float64)[ok][oc])])

    def __call__(self, rows, idx, valid) -> np.ndarray:
        ck = self.cat_rows[np.asarray(rows)]
        idx = np.asarray(idx, np.int64)
        key = ck[:, None] * self.ngal + idx
        if self.exact_key.size:
            pos = np.clip(np.searchsorted(self.exact_key, key), 0, self.exact_key.size - 1)
            hit = self.exact_key[pos] == key
            exact = self.exact_val[pos]
        else:
            hit, exact = np.zeros(key.shape, bool), np.ones(key.shape)
        lo = np.searchsorted(self.claim_key, idx * (self.ncl + 1))
        hi = np.searchsorted(self.claim_key, idx * (self.ncl + 1) + self.rank[ck][:, None])
        claimed = self.claim_cs[hi] - self.claim_cs[lo]
        pf = np.where(hit, exact, np.clip(1.0 - claimed, 0.0, 1.0))
        return np.where(valid, pf, 1.0).astype(np.float32)


def remeasure(region: Region, centres: Centres, *, pfree=None, fixed_z: bool = False,
              batch: int = 1024) -> dict[str, np.ndarray]:
    """lambda, z_lambda and the aperture columns of the clusters in the region's cosmology.

    z_lambda is iterated from ``centres.z0`` (Z_LAMBDA_RAW, Z_LAMBDA_E_RAW before the calibrated
    correction; Z_LAMBDA, Z_LAMBDA_E after it), unless ``fixed_z``. ``pfree``: a
    :class:`MemberPfree` (default: every neighbour free).
    """
    st, q = stage_and_quad(region.cfg)
    kind = "richness" if fixed_z else "zlambda"
    out = _run_batched(region, centres.gi, centres.z0, st, q, kind, batch, log_every=0,
                       pfree_nb=pfree)
    out["Z_LAMBDA_RAW"] = out["Z_LAMBDA"].copy()
    if fixed_z:
        out["Z_LAMBDA_E"] = np.full(centres.size, np.nan, np.float32)
    else:
        zc, ec = region.correct_zlambda(out["Z_LAMBDA"], out["Z_LAMBDA_E"], out["LAMBDA"])
        out["Z_LAMBDA"], out["Z_LAMBDA_E"] = zc.astype(np.float32), ec.astype(np.float32)
    return {k: out[k] for k in COLUMNS}


def cosmology_grid(base: CosmologyConfig, vary: Mapping[str, Sequence[float]]) -> dict[str, CosmologyConfig]:
    """The base cosmology and one-at-a-time variations, keyed by label (base first).

    >>> list(cosmology_grid(CosmologyConfig(), {"Omega_m": [0.25, 0.35], "w0": [-0.8]}))
    ['fiducial', 'Omega_m=0.25', 'Omega_m=0.35', 'w0=-0.8']
    """
    out = {base.label(): base}
    for key, values in vary.items():
        if key not in PARAMS:
            raise ValueError(f"unknown cosmological parameter {key!r}; known: {PARAMS}")
        for v in values:
            c = dataclasses.replace(base, **{key: float(v)})
            out[c.label()] = c
    return out


def remeasure_cosmologies(region: Region, centres: Centres,
                          cosmologies: Mapping[str, CosmologyConfig], *, pfree=None,
                          fixed_z: bool = False, mstar_follows: bool = False,
                          batch: int = 1024) -> dict[str, dict[str, np.ndarray]]:
    """:func:`remeasure` in each cosmology (zred recomputed; see :meth:`Region.with_cosmology`)."""
    res = {}
    for label, c in cosmologies.items():
        reg = region.with_cosmology(c, mstar_follows=mstar_follows)
        res[label] = remeasure(reg, centres, pfree=pfree, fixed_z=fixed_z, batch=batch)
        lam0, lam = res[next(iter(res))]["LAMBDA"], res[label]["LAMBDA"]
        good = (lam0 > 0) & (lam > 0)
        log.info("%s: %d clusters, median ln(lambda / lambda_%s) = %+.4f", label, centres.size,
                 next(iter(res)), float(np.median(np.log(lam[good] / lam0[good]))) if good.any() else np.nan)
    return res


@jax.jit
def _richness_jvp(nb, z, frad, fgeo, quad, model, stage, dcosmo):
    """(ln lambda, SCALEVAL, MASKFRAC) and their derivatives along ``dcosmo`` and along z."""
    def f(cosmo, zz):
        r = richness(nb, zz, frad, fgeo, quad, dataclasses.replace(model, cosmo=cosmo), stage)
        return jnp.log(jnp.maximum(r.lam, 1e-30)), r.scaleval, r.maskfrac

    prim, d_cosmo = jax.jvp(f, (model.cosmo, z), (dcosmo, jnp.zeros_like(z)))
    zero = jax.tree_util.tree_map(jnp.zeros_like, dcosmo)
    _, d_z = jax.jvp(f, (model.cosmo, z), (zero, jnp.ones_like(z)))
    return prim, d_cosmo, d_z


def response_jvp(region: Region, centres: Centres, params: Sequence[str] = ("Omega_m", "w0"), *,
                 z: np.ndarray | None = None, pfree=None, batch: int = 1024) -> dict[str, np.ndarray]:
    """Derivatives at fixed redshift ``z`` (default ``centres.z0``) and fixed centre.

    Columns: LNLAMBDA, DLNLAMBDA_DZ, and per parameter P: DLNLAMBDA_D<P>, DSCALEVAL_D<P>,
    DMASKFRAC_D<P> (the last two without the change of the mask completeness with D_A). The
    region's distance table must be that of its configuration (no ``cosmo=`` override).
    """
    st, q = stage_and_quad(region.cfg)
    z0 = centres.z0 if z is None else np.asarray(z, np.float64)
    _, der = CosmoTable.jvp(region.cfg.cosmology, params)
    n = centres.size
    out = {"LNLAMBDA": np.full(n, np.nan, np.float32), "DLNLAMBDA_DZ": np.full(n, np.nan, np.float32)}
    for p in params:
        for c in ("DLNLAMBDA", "DSCALEVAL", "DMASKFRAC"):
            out[f"{c}_D{p.upper()}"] = np.full(n, np.nan, np.float32)
    for bt in _iter_batches(region, centres.gi, z0, st, q, "richness", batch, pfree_nb=pfree):
        tgt, m = bt.rows[:bt.n], bt.n
        zj = jnp.asarray(bt.z, jnp.float32)
        for k, p in enumerate(params):
            prim, dc, dz = _richness_jvp(bt.nb, zj, bt.frad, bt.fgeo, q, region.model, st, der[p])
            ok = np.asarray(prim[0])[:m] > -1.0          # lambda < 1 is reported as -1
            if k == 0:
                out["LNLAMBDA"][tgt] = np.where(ok, np.asarray(prim[0])[:m], np.nan)
                out["DLNLAMBDA_DZ"][tgt] = np.where(ok, np.asarray(dz[0])[:m], np.nan)
            for c, d in zip(("DLNLAMBDA", "DSCALEVAL", "DMASKFRAC"), dc):
                out[f"{c}_D{p.upper()}"][tgt] = np.where(ok, np.asarray(d)[:m], np.nan)
    return out


def response_fd(region: Region, centres: Centres, steps: Mapping[str, float], *, pfree=None,
                fixed_z: bool = True, batch: int = 1024) -> dict[str, np.ndarray]:
    """Central differences d ln(lambda)/d(theta) (fixed redshift by default), for checking
    :func:`response_jvp`: DLNLAMBDA_D<P>_FD."""
    out = {}
    base = region.cfg.cosmology
    for p, h in steps.items():
        lam = []
        for s in (+1, -1):
            c = dataclasses.replace(base, **{p: getattr(base, p) + s * h})
            reg = region.with_cosmology(c, recompute_zred=not fixed_z)
            lam.append(remeasure(reg, centres, pfree=pfree, fixed_z=fixed_z, batch=batch)["LAMBDA"])
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (np.log(lam[0]) - np.log(lam[1])) / (2 * h)
        out[f"DLNLAMBDA_D{p.upper()}_FD"] = np.where((lam[0] > 0) & (lam[1] > 0), d, np.nan).astype(np.float32)
    return out


def galaxies_near(gal: Mapping[str, np.ndarray], ra, dec, radius_deg, nside: int = 512) -> dict:
    """The galaxies in the HEALPix pixels (``nside``) touching a disc of ``radius_deg`` (per
    centre) around any centre: a superset of those within the discs, to re-measure a few clusters
    of a large table without computing zred for all of it."""
    import healpy as hp

    ra, dec = np.atleast_1d(np.asarray(ra, np.float64)), np.atleast_1d(np.asarray(dec, np.float64))
    rad = np.broadcast_to(np.radians(np.asarray(radius_deg, np.float64)), ra.shape)
    vec = hp.ang2vec(ra, dec, lonlat=True) if ra.size else np.zeros((0, 3))
    pix = [hp.query_disc(nside, v, r, inclusive=True, nest=True) for v, r in zip(vec, rad)]
    pix = np.unique(np.concatenate(pix)) if pix else np.zeros(0, np.int64)
    keep = np.isin(hp.ang2pix(nside, gal["RA"], gal["DEC"], nest=True, lonlat=True), pix)
    return {k: np.asarray(v)[keep] for k, v in gal.items()}


#: Steps of :func:`response_fd` for each parameter.
FD_STEPS = {"Omega_m": 0.01, "h": 0.01, "Omega_b": 0.005, "sum_mnu": 0.02, "w0": 0.03, "wa": 0.1}

#: Catalogue columns copied to the output (with a ``_CAT`` suffix for the measured ones).
CAT_ID = ("MEM_MATCH_ID", "RA", "DEC")
CAT_VALUES = ("LAMBDA", "Z_LAMBDA", "Z_LAMBDA_RAW", "R_LAMBDA", "SCALEVAL", "MASKFRAC")


def parse_vary(items) -> dict[str, list[float]]:
    """``["Omega_m=0.25,0.35", "w0=-0.8"]`` -> ``{"Omega_m": [0.25, 0.35], "w0": [-0.8]}``.

    >>> parse_vary(["Omega_m=0.25,0.35", "w0=-0.8"])
    {'Omega_m': [0.25, 0.35], 'w0': [-0.8]}
    """
    out: dict[str, list[float]] = {}
    for item in items or ():
        key, sep, values = str(item).partition("=")
        key = key.strip()
        if not sep or key not in PARAMS:
            raise ValueError(f"--vary {item!r}: expected KEY=V1,V2,... with KEY in {PARAMS}")
        out.setdefault(key, []).extend(float(v) for v in values.split(",") if v.strip())
    return out


def write_remeasure(path, cat: Mapping[str, np.ndarray], centres: Centres,
                    results: Mapping[str, Mapping[str, np.ndarray]],
                    cosmologies: Mapping[str, CosmologyConfig],
                    extra: Mapping[str, np.ndarray] | None = None, header: Mapping | None = None):
    """CLUSTERS (catalogue identity and values, then each column of :data:`COLUMNS` as
    [n_cluster, n_cosmology] in the order of COSMOLOGIES, then ``extra``) and COSMOLOGIES HDUs."""
    from ..io.tables import table_hdu, write_fits

    rows = centres.rows
    cols: dict[str, np.ndarray] = {}
    for k in CAT_ID:
        if k in cat:
            cols[k] = np.asarray(cat[k])[rows]
    ids = np.asarray(cat["ID_CENT"])
    cols["ID_CENT"] = (ids[:, 0] if ids.ndim == 2 else ids)[rows]
    for k in CAT_VALUES:
        if k in cat:
            cols[f"{k}_CAT"] = np.asarray(cat[k], np.float32)[rows]
    labels = list(cosmologies)
    for k in COLUMNS:
        cols[k] = np.stack([np.asarray(results[lab][k], np.float32) for lab in labels], axis=1)
    cols.update(extra or {})
    keys = list(CosmologyConfig().header())
    cos = {"LABEL": np.array([lab.encode() for lab in labels])}
    for k in keys:
        cos[k] = np.array([cosmologies[lab].header()[k] for lab in labels])
    return write_fits(path, [table_hdu(cols, extname="CLUSTERS"), table_hdu(cos, extname="COSMOLOGIES")],
                      {"MODE": "remeasure", **(header or {})})


def read_remeasure(path) -> tuple[dict, list[str], dict]:
    """(clusters, cosmology labels, primary header) of a :func:`write_remeasure` file."""
    from astropy.io import fits

    from ..io.tables import read_table

    cat = read_table(path, hdu="CLUSTERS")
    labels = [x.decode() if isinstance(x, bytes) else str(x)
              for x in read_table(path, hdu="COSMOLOGIES")["LABEL"]]
    return cat, labels, dict(fits.getheader(path))
