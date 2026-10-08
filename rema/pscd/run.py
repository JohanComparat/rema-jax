"""A PSCD run over a region: galaxies, noise, grid and cube, detections, catalogue and members.

The galaxies are those of the galaxy tables (``rema ingest``) with a usable photo-z and brighter
than the magnitude limit (``photoz.mag_max``, and the catalogue's); the null tests of
:mod:`rema.validate.null` apply first. The noise is the stacked photo-z background of these
galaxies over the effective area of the footprint. A region keeps the detections centred in its
own box, within the redshift range and with MASKFRAC < ``pscd.max_maskfrac``.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import numpy as np

from ..config import RemaConfig
from ..model.cosmo import CosmoTable
from ..model.photoz import photoz_sigma
from ..model.profiles import MStar
from ..validate.null import null_shuffle
from .detect import Galaxies, build_cube, extract, magnitude_limit
from .grid import Grid
from .model import Template, WidthTable

log = logging.getLogger(__name__)


def run_pscd(gal: dict, cfg: RemaConfig, *, footprint=None, sky=None, own=None, region_id: int = 0,
             rs=None, area_deg2: float | None = None, info: dict | None = None):
    """PSCD detections of a galaxy table: (catalogue, members) dicts.

    ``footprint``: the region's footprint (unmasked fraction and depth; without one the whole
    grid is observed down to the magnitude limit). ``sky``: the data box (pixels outside are
    masked). ``own``: keep detections centred there. ``rs``: a red-sequence model, for the
    members' CHISQ_RS. ``area_deg2``: the area without a footprint (default: from the positions).
    """
    from ..modes.common import _area_function, photoz_background, red_sequence_chisq

    t0 = time.time()
    g = dict(gal)
    bands = tuple(cfg.survey.bands)
    iref = bands.index(cfg.survey.ref_band)
    if cfg.null.shuffle != "none":
        g = null_shuffle(g, cfg.null, iref)
        log.info("null test: %s shuffled within %.2f mag (seed %d)", cfg.null.shuffle, cfg.null.magbin,
                 cfg.null.seed)
    if "ZPHOT" not in g or "ZPHOT_STD" not in g:
        raise KeyError("rema pscd needs the ZPHOT and ZPHOT_STD columns (rema ingest with the photo-z sweeps)")
    g["ZPHOT_E"] = photoz_sigma(g["ZPHOT"], g["ZPHOT_STD"], g["REFMAG"], cfg.photoz)
    lim = magnitude_limit(cfg)
    rows = np.flatnonzero((g["ZPHOT_E"] > 0) & (np.asarray(g["REFMAG"]) < lim))
    stats = {"n_galaxies": int(np.size(g["REFMAG"])), "n_used": int(rows.size)}
    if rows.size == 0:
        log.warning("no galaxy with a usable photo-z: empty catalogue")
        if info is not None:
            info.update(stats, n_detections=0, n_clusters=0, t_total=time.time() - t0)
        return {}, {}
    sub = {k: np.asarray(g[k])[rows] for k in ("RA", "DEC", "REFMAG", "ZPHOT", "ZPHOT_E")}
    lim = min(lim, float(np.max(sub["REFMAG"])) + 1e-3)
    area_fn = _area_function(sub, SimpleNamespace(ref_band=cfg.survey.ref_band), cfg, footprint, area_deg2, sky)
    bkg = photoz_background(sub, cfg, area_fn)
    widths = WidthTable.build(sub["ZPHOT"], sub["ZPHOT_E"], sub["REFMAG"])
    cosmo = CosmoTable.from_config(cfg.cosmology)
    tpl = Template.create(cfg.pscd, MStar(cfg.model.mstar), cosmo)
    grid = Grid.covering(sub["RA"], sub["DEC"], cfg.pscd.pixel)
    rac, decc = grid.centres()
    if footprint is not None:
        fmap = np.asarray(footprint.fine.lookup_np(rac, decc, "FRACGOOD"), np.float64)
        sigmap = np.asarray(footprint.fine.lookup_np(rac, decc, f"SIGF_{cfg.survey.ref_band.upper()}"),
                            np.float64)
    else:
        fmap, sigmap = np.ones(grid.shape), None
    if sky is not None:
        fmap = fmap * sky.contains(rac, decc)
    px, py = grid.to_pix(sub["RA"], sub["DEC"])
    gx = Galaxies(rows=rows, ra=sub["RA"], dec=sub["DEC"], refmag=sub["REFMAG"].astype(np.float64),
                  zphot=sub["ZPHOT"].astype(np.float64), s=sub["ZPHOT_E"].astype(np.float64), px=px, py=py,
                  pfield=np.ones(rows.size))
    log.info("pscd: %d galaxies, grid %d x %d pixels of %.3f deg, magnitude limit %.2f", rows.size,
             grid.nx, grid.ny, grid.pixel, lim)
    cube = build_cube(gx, grid, tpl, bkg, widths, cfg, fmap, sigmap, mag_limit=lim)
    t_cube = time.time() - t0
    dets, mems = extract(cube, gx)
    stats.update(n_detections=len(dets), t_cube=t_cube, grid=(grid.ny, grid.nx, cube.zs.size))

    # ---------------------------------------------------------------- catalogue
    pc = cfg.pscd
    zr = pc.zrange if pc.zrange is not None else cfg.model.zrange
    cat_rows, mem_rows = [], []
    for rank, (d, (idx, P, th)) in enumerate(zip(dets, mems)):
        keep = (zr[0] <= d.z <= zr[1]) and d.maskfrac < pc.max_maskfrac
        if own is not None:
            keep = keep and bool(own.contains(np.array([d.ra]), np.array([d.dec]))[0])
        if not keep:
            continue
        m = gx.refmag[idx]
        r200 = float(tpl.r200_deg(d.z))
        star = (m < tpl.mstar(d.z) + pc.lambda_star_dmag) & (th < r200)
        bright = np.flatnonzero(P >= 0.5)
        ib = bright[np.argmin(m[bright])] if bright.size else -1
        src = rows[idx[ib]] if ib >= 0 else -1
        cat_rows.append({
            "RANK": rank, "RA": d.ra, "DEC": d.dec, "Z": d.z, "Z_E": d.z_e, "A": d.A, "A_E": d.A_e,
            "SNR": d.snr, "SNR_NOCL": d.snr_nocl, "LNLIKE": d.lnlike, "LAMBDA": float(P.sum()),
            "LAMBDA_STAR": float(P[star].sum()), "N_EXP": d.n_exp, "NMEM": int(np.sum(P >= pc.pmin)),
            "MASKFRAC": d.maskfrac, "MLIM": d.mlim, "R200": r200 * float(tpl.mpc_per_deg(d.z)),
            "ID_BCG": int(g["ID"][src]) if src >= 0 else -1,
            "RA_BCG": float(g["RA"][src]) if src >= 0 else -400.0,
            "DEC_BCG": float(g["DEC"][src]) if src >= 0 else -400.0,
            "REFMAG_BCG": float(g["REFMAG"][src]) if src >= 0 else np.nan,
            "ZPHOT_BCG": float(g["ZPHOT"][src]) if src >= 0 else -1.0,
        })
        sel = P >= pc.pmin
        mem_rows.append((len(cat_rows) - 1, rows[idx[sel]], P[sel], th[sel] * float(tpl.mpc_per_deg(d.z)), d.z))
    if not cat_rows:
        if info is not None:
            info.update(stats, n_clusters=0, t_total=time.time() - t0)
        return {}, {}
    cat = {k: np.array([c[k] for c in cat_rows]) for k in cat_rows[0]}
    order = np.argsort(-cat["SNR"], kind="stable")
    cat = {k: v[order] for k, v in cat.items()}
    new = np.empty(order.size, np.int64)
    new[order] = np.arange(order.size)
    mmid = (np.int64(region_id) << 32) | np.arange(order.size, dtype=np.int64)
    cat["MEM_MATCH_ID"] = mmid

    # ---------------------------------------------------------------- members
    ci = np.concatenate([np.full(r.size, new[c]) for c, r, _, _, _ in mem_rows])
    gr = np.concatenate([r for _, r, _, _, _ in mem_rows])
    mem = {"MEM_MATCH_ID": mmid[ci], "ID": np.asarray(g["ID"])[gr], "RA": np.asarray(g["RA"])[gr],
           "DEC": np.asarray(g["DEC"])[gr],
           "Z": np.concatenate([np.full(r.size, z, np.float32) for _, r, _, _, z in mem_rows]),
           "P": np.concatenate([p for _, _, p, _, _ in mem_rows]).astype(np.float32),
           "PFIELD": gx.pfield[np.searchsorted(rows, gr)].astype(np.float32),
           "R": np.concatenate([r for _, _, _, r, _ in mem_rows]).astype(np.float32),
           "REFMAG": np.asarray(g["REFMAG"])[gr], "REFMAG_ERR": np.asarray(g["REFMAG_ERR"])[gr],
           "ZPHOT": np.asarray(g["ZPHOT"])[gr], "ZPHOT_STD": np.asarray(g["ZPHOT_STD"])[gr],
           "ZPHOT_E": g["ZPHOT_E"][gr]}
    if "ZSPEC" in g:
        mem["ZSPEC"] = np.asarray(g["ZSPEC"])[gr]
    if rs is not None and gr.size:
        bidx = [bands.index(b) for b in rs.bands]
        mem["CHISQ_RS"] = red_sequence_chisq(rs, np.asarray(g["FLUX"])[gr][:, bidx],
                                             np.asarray(g["FLUX_IVAR"])[gr][:, bidx], mem["Z"],
                                             cfg.model.chisq_mode, cfg.survey.flux_floor)
    o = np.lexsort((-mem["P"], mem["MEM_MATCH_ID"]))
    mem = {k: v[o] for k, v in mem.items()}
    if info is not None:
        import resource

        stats.update(n_clusters=len(cat["RA"]), t_total=time.time() - t0,
                     peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        info.update(stats)
    return cat, mem
