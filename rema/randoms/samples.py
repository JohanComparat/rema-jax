"""Volume-limited cluster samples and their randoms.

A :class:`SampleSpec` (YAML) defines a sample: redshift and observable bins, the detection cuts
(LAMBDA / SCALEVAL and MASKFRAC, as the catalogue), extra column cuts and an external selection.
The steps, each a function here and a ``rema cluster-randoms`` subcommand:

1. :func:`select_sources`: the clusters of the sample, in the domain, below the z_vlim of their
   own position, inside the bins, passing the cuts and the external survey's own cut;
2. :func:`make_copies`: ``n_samp`` copies of each at random points of D(z) (:mod:`.generate`);
3. :func:`place`: SCALEVAL, MASKFRAC, R_LAMBDA of every copy at its position (:mod:`.aperture`);
4. :func:`build_sample`: the weights (:mod:`.weigh`) of the copies of the sample's clusters, for
   that sample or any other sample whose clusters are among the sources (a sample with fewer
   bins, a higher threshold).

>>> s = SampleSpec(z_edges=(0.1, 0.3), obs_edges=(20.0, 40.0))
>>> s.bin_of(np.array([0.2, 0.35, 0.29]), np.array([25.0, 30.0, 40.0]))
(array([ 0, -1,  0]), array([ 0,  0, -1]))
"""

from __future__ import annotations

import dataclasses
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..config import _build, _to_plain
from .weigh import copy_weights, detection_q

_OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt,
        "==": operator.eq, "!=": operator.ne}


@dataclass(frozen=True)
class SampleSpec:
    name: str = "lambda_vlim"
    ra_column: str = "RA"
    dec_column: str = "DEC"
    z_column: str = "Z_LAMBDA"                # redshift of the bins and of the volume limit
    z_raw_column: str = "Z_LAMBDA_RAW"        # redshift of the aperture completeness
    lambda_column: str = "LAMBDA"
    observable: str = "LAMBDA"                # binned column
    z_edges: tuple[float, ...] = (0.1, 0.2, 0.3, 0.45, 0.6, 0.75)
    obs_edges: tuple[float, ...] = (20.0, 30.0, 45.0, 60.0, 1000.0)
    # Detection: LAMBDA / SCALEVAL >= lambda_det (None: the catalogue's richness.minlambda), or
    # >= the lower edge of the cluster's observable bin with detection_from_bin (redMaPPer's
    # select_scaleval, for richness bins); MASKFRAC < max_maskfrac (None: the catalogue's).
    lambda_det: float | None = None
    detection_from_bin: bool = False
    max_maskfrac: float | None = None
    cuts: tuple[str, ...] = ()                # e.g. "EXT_LIKE >= 6" (column, operator, number)
    selection: dict | None = None             # external selection, see rema.randoms.selection
    n_samp: int = 100

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SampleSpec":
        return _build(cls, d or {})

    @classmethod
    def from_yaml(cls, path) -> "SampleSpec":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    # ------------------------------------------------------------------ bins and cuts
    def bin_of(self, z, obs) -> tuple[np.ndarray, np.ndarray]:
        """(redshift bin, observable bin) of each row; -1 outside the edges."""
        def b(x, e):
            e = np.asarray(e, np.float64)
            k = np.searchsorted(e, np.asarray(x, np.float64), side="right") - 1
            return np.where((k >= 0) & (k < e.size - 1), k, -1)
        return b(z, self.z_edges), b(obs, self.obs_edges)

    def detection_threshold(self, obs_bin, cfg) -> np.ndarray:
        """LAMBDA / SCALEVAL threshold of each row."""
        base = cfg.richness.minlambda if self.lambda_det is None else self.lambda_det
        t = np.full(np.shape(obs_bin), float(base))
        if self.detection_from_bin:
            lo = np.asarray(self.obs_edges, np.float64)[np.maximum(obs_bin, 0)]
            t = np.maximum(t, lo)
        return t

    def maskfrac_max(self, cfg) -> float:
        return float(cfg.mask.max_maskfrac if self.max_maskfrac is None else self.max_maskfrac)

    def apply_cuts(self, cat: dict) -> np.ndarray:
        n = len(cat[self.ra_column])
        ok = np.ones(n, bool)
        for c in self.cuts:
            col, op, val = c.split()
            if op not in _OPS:
                raise ValueError(f"cut {c!r}: operator must be one of {sorted(_OPS)}")
            ok &= _OPS[op](np.asarray(cat[col.upper()]), float(val))
        return ok

    def columns(self, selection=None) -> list[str]:
        """Catalogue columns the sample needs."""
        cols = {self.ra_column, self.dec_column, self.z_column, self.z_raw_column, self.lambda_column,
                self.observable, "SCALEVAL", "MASKFRAC"}
        cols |= {c.split()[0].upper() for c in self.cuts}
        if selection is not None:
            cols |= set(selection.columns)
        return sorted(cols)


def select_sources(cat: dict, spec: SampleSpec, zmap, domain, selection, cfg) -> tuple[np.ndarray, dict]:
    """Rows of ``cat`` in the sample, and per-row annotations (ZVLIM, IN_DOMAIN, BIN_Z,
    BIN_OBS, LAMBDA_DET) of the whole catalogue."""
    from .vlim import annotate

    ra, dec = np.asarray(cat[spec.ra_column], np.float64), np.asarray(cat[spec.dec_column], np.float64)
    z = np.asarray(cat[spec.z_column], np.float64)
    ann = annotate(ra, dec, zmap, domain)
    bz, bo = spec.bin_of(z, cat[spec.observable])
    thr = spec.detection_threshold(bo, cfg)
    lam, sv, mf = (np.asarray(cat[k], np.float64) for k in (spec.lambda_column, "SCALEVAL", "MASKFRAC"))
    det = detection_q(lam, sv, mf, thr, spec.maskfrac_max(cfg))
    with np.errstate(invalid="ignore"):
        keep = (ann["IN_DOMAIN"] & (z < np.where(np.isfinite(ann["ZVLIM"]), ann["ZVLIM"], -np.inf))
                & (bz >= 0) & (bo >= 0) & (det > 0) & spec.apply_cuts(cat))
    keep &= np.asarray(selection.in_footprint(ra, dec), bool) & np.asarray(selection.own_cut(cat), bool)
    ann.update(BIN_Z=bz.astype(np.int16), BIN_OBS=bo.astype(np.int16), LAMBDA_DET=thr)
    return keep, ann


def source_table(cat: dict, rows: np.ndarray, ann: dict, spec: SampleSpec, selection) -> dict:
    """Columns of the sample's clusters carried by their copies."""
    src = {"ROW": rows.astype(np.int64),
           "Z": np.asarray(cat[spec.z_column], np.float64)[rows],
           "Z_RAW": np.asarray(cat[spec.z_raw_column], np.float64)[rows],
           "LAMBDA": np.asarray(cat[spec.lambda_column], np.float64)[rows],
           "OBS": np.asarray(cat[spec.observable], np.float64)[rows],
           "LAMBDA_DET": ann["LAMBDA_DET"][rows], "BIN_Z": ann["BIN_Z"][rows],
           "BIN_OBS": ann["BIN_OBS"][rows]}
    if "MEM_MATCH_ID" in cat:
        src["MEM_MATCH_ID"] = np.asarray(cat["MEM_MATCH_ID"])[rows]
    for c in selection.columns:
        src[c] = np.asarray(cat[c])[rows]
    return src


def make_copies(src: dict, pool, n_samp: int, seed: int) -> tuple[dict, dict]:
    """``n_samp`` copies of every source at random points of D(Z) (see :func:`.generate.generate`)."""
    from .generate import generate

    rng = np.random.default_rng(seed)
    return generate(src, pool, n_samp, rng, z_col="Z")


def place(rand: dict, src: dict, fmap, cfg, batch: int = 16384) -> dict:
    """SCALEVAL, MASKFRAC and R_LAMBDA of each copy at its position, for the richness and
    redshift (Z_RAW) of its source cluster."""
    from .aperture import aperture_values

    s = np.asarray(rand["SRC"])
    out = aperture_values(rand["RA"], rand["DEC"], np.asarray(src["Z_RAW"])[s],
                          np.asarray(src["LAMBDA"])[s], fmap, cfg, batch=batch)
    return {**rand, **{k: v.astype(np.float32) for k, v in out.items()}}


def build_sample(src: dict, rand: dict, keep_rows: np.ndarray, lambda_det: np.ndarray, selection,
                 max_maskfrac: float, chunk: int = 5_000_000, zmap=None,
                 domain=None) -> tuple[np.ndarray, dict, dict]:
    """Weights of the copies of the clusters at catalogue rows ``keep_rows`` (a subset of the
    sources' ROW); ``lambda_det`` the detection threshold of every catalogue row (that of the
    sample's spec, see :func:`select_sources`). Returns (index of those copies in ``rand``, their
    columns Q, P_EXT, WEIGHT, per-source diagnostics SUMQ, NEFF, NPASS for the kept sources).

    ``zmap``, ``domain``: the sample's z_vlim map and domain, when they are stricter than those
    the copies were drawn with (a higher depth significance, a seam buffer): copies outside them
    get q = 0. Copies drawn uniformly in D(z) and kept in a subset of it are uniform in the
    subset, so the copies of one generation serve every stricter sample."""
    pos = {r: i for i, r in enumerate(src["ROW"])}
    missing = [r for r in keep_rows if r not in pos]
    if missing:
        raise ValueError(f"{len(missing)} clusters of the sample have no copies (not among the "
                         f"sources of the randoms; e.g. catalogue rows {missing[:5]})")
    srcsel = np.zeros(src["ROW"].size, bool)
    srcsel[[pos[r] for r in keep_rows]] = True
    idx = np.flatnonzero(srcsel[rand["SRC"]])
    s = rand["SRC"][idx]
    q = detection_q(src["LAMBDA"][s], rand["SCALEVAL"][idx], rand["MASKFRAC"][idx],
                    np.asarray(lambda_det)[src["ROW"][s]], max_maskfrac)
    if zmap is not None or domain is not None:
        ra, dec = np.asarray(rand["RA"])[idx], np.asarray(rand["DEC"])[idx]
        inside = np.ones(idx.size, bool)
        if zmap is not None:
            zv = zmap.value_at(ra, dec)
            inside &= np.where(np.isfinite(zv), zv, -np.inf) > np.asarray(src["Z"])[s]
        if domain is not None:
            inside &= domain.contains(ra, dec)
        q = q * inside
    pext = np.ones(idx.size)
    if selection.name != "none":
        cols = {c: np.asarray(v)[s] for c, v in src.items()}
        for lo in range(0, idx.size, chunk):
            sl = slice(lo, lo + chunk)
            pext[sl] = selection.p_detect({c: v[sl] for c, v in cols.items()},
                                          rand["RA"][idx][sl], rand["DEC"][idx][sl])
    w, diag = copy_weights(s, q * pext, src["N_SAMP"])
    return idx, {"Q": q.astype(np.float32), "P_EXT": pext.astype(np.float32), "WEIGHT": w}, \
        {k: v[srcsel] for k, v in diag.items()}
