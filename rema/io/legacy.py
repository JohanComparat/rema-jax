"""Legacy Surveys DR11 sweeps -> compact galaxy tables.

A sweep file and its row-matched photo-z file (``11.0-photo-z/<name>-pz.fits``) are reduced to
the columns the cluster finder needs:

ID (LS_ID_DR11 = RELEASE<<42 | BRICKID<<22 | OBJID), RA, DEC, dereddened FLUX[nb] and
FLUX_IVAR[nb] (nanomaggies), REFMAG, REFMAG_ERR, EBV, TYPE (code), MASKBITS, ZSPEC (-1 if none),
ZSPEC_SRC (index into SURVEYS, 0 if none), ZPHOT, ZPHOT_STD.

Selection (all explicit, see :class:`rema.config.SurveyConfig`): MASKBITS clean (objects fit as
SGA large galaxies are kept), TYPE != PSF, NOBS >= 1 in every band, reference-band S/N >= 5,
REFMAG < mag_max and, when ``survey.ebv_max`` is set, E(B-V) < ebv_max (the randoms get the same
cut, :func:`rema.sky.maps.good_randoms`). Fluxes are not otherwise cut, so faint or negative
fluxes in the bluer bands are kept with their errors.

For large runs, :func:`ingest_sweeps` writes one table per sweep (named as the sweep) and
:func:`read_galaxies` assembles the galaxies of any box or set of boxes from them.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import yaml

from ..config import RemaConfig
from ..sky.regions import Box, BoxUnion, sky_header, sweep_box, sweeps_overlapping
from .tables import Table, read_header, read_table, write_table

log = logging.getLogger(__name__)

# SURVEY_BITMASK bit numbers of the DR11 photo-z sweeps; ZSPEC_SRC = bit + 1.
SURVEYS = ("DESI", "DESI-COSMOS", "SDSS", "BOSS", "eBOSS-LRG", "eBOSS-ELG", "DEEP2+3", "DEEP2",
           "AGES", "VIPERS", "GAMA", "WiggleZ", "OzDES", "2dFLenS", "C3R2", "VVDS", "COSMOS2015",
           "SDSS-QSO")
TYPES = ("PSF", "REX", "EXP", "DEV", "SER", "DUP")
FITBITS_LARGEGALAXY = 9


def ls_id(release, brickid, objid) -> np.ndarray:
    """LS_ID_DR11 = RELEASE << 42 | BRICKID << 22 | OBJID."""
    return ((np.asarray(release, np.int64) << 42) | (np.asarray(brickid, np.int64) << 22)
            | np.asarray(objid, np.int64))


def pz_path(sweep: str | Path) -> Path:
    """Row-matched photo-z sweep of a sweep file."""
    sweep = Path(sweep)
    return sweep.parent.parent / f"{sweep.parent.name}-photo-z" / sweep.name.replace(".fits", "-pz.fits")


def default_mag_max(cfg: RemaConfig) -> float:
    """Reference-magnitude limit: m*(zmax) + 2.5 (the faint end of the background histogram)."""
    from ..model.profiles import MStar

    if cfg.survey.mag_max is not None:
        return float(cfg.survey.mag_max)
    zmax = max(cfg.model.zrange[1], cfg.scan.zrange[1])
    return float(MStar(cfg.model.mstar)(zmax)) - 2.5 * np.log10(cfg.background.lmax_faint)


def survey_hash(cfg: RemaConfig) -> str:
    """Short hash of everything that decides which galaxies a table holds (the ``survey``
    configuration and the magnitude limit), recorded as SURVHASH in galaxy tables."""
    text = yaml.safe_dump({"survey": cfg.to_dict()["survey"],
                           "mag_max": round(float(default_mag_max(cfg)), 6)}, sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def ingest_sweep(sweep: str | Path, cfg: RemaConfig | None = None, box: Box | None = None,
                 pz: str | Path | None = "auto", mag_max: float | None = None,
                 require_pz: bool = False) -> Table:
    """Select and convert the galaxies of one sweep (optionally restricted to ``box``).

    Always returns every column, with zero rows when nothing is selected. ``require_pz``: a
    missing photo-z file (``pz="auto"``) is an error instead of a warning.
    """
    cfg = cfg or RemaConfig()
    sv = cfg.survey
    bands = [b.upper() for b in sv.bands]
    ref = sv.ref_band.upper()
    mag_max = default_mag_max(cfg) if mag_max is None else mag_max

    pos = read_table(sweep, ["RA", "DEC"])
    keep = np.isfinite(pos["RA"]) & np.isfinite(pos["DEC"])
    if box is not None:
        keep &= box.contains(pos["RA"], pos["DEC"])
    rows = np.flatnonzero(keep)

    cols = ["RELEASE", "BRICKID", "OBJID", "TYPE", "EBV", "MASKBITS", "FITBITS"]
    for b in bands:
        cols += [f"FLUX_{b}", f"FLUX_IVAR_{b}", f"MW_TRANSMISSION_{b}", f"NOBS_{b}"]
    t = read_table(sweep, cols, rows=rows)

    reject = np.int64(sum(1 << int(bit) for bit in sv.maskbits_reject))
    largegal = (t["FITBITS"].astype(np.int64) & (1 << FITBITS_LARGEGALAXY)) != 0
    ok = (t["MASKBITS"].astype(np.int64) & reject) == 0
    if sv.keep_largegalaxy:
        ok |= largegal
    typ = t["TYPE"].astype("U3")
    if sv.reject_psf:
        ok &= typ != "PSF"
    ok &= typ != "DUP"
    for b in bands:
        ok &= t[f"NOBS_{b}"] >= sv.nobs_min
        ok &= t[f"MW_TRANSMISSION_{b}"] > 0
    if sv.ebv_max is not None:
        ok &= t["EBV"] < sv.ebv_max
    f_ref, iv_ref = t[f"FLUX_{ref}"], t[f"FLUX_IVAR_{ref}"]
    ok &= (iv_ref > 0) & (f_ref * np.sqrt(np.maximum(iv_ref, 0)) >= sv.ref_snr_min)
    mw_ref = t[f"MW_TRANSMISSION_{ref}"]
    with np.errstate(divide="ignore", invalid="ignore"):
        refmag = sv.zeropoint - 2.5 * np.log10(np.where(ok, f_ref / mw_ref, 1.0))
    ok &= refmag < mag_max

    sel = np.flatnonzero(ok)
    out: Table = {}
    out["ID"] = ls_id(t["RELEASE"][sel], t["BRICKID"][sel], t["OBJID"][sel])
    out["RA"] = pos["RA"][rows[sel]].astype(np.float64)
    out["DEC"] = pos["DEC"][rows[sel]].astype(np.float64)
    flux = np.stack([t[f"FLUX_{b}"][sel] / t[f"MW_TRANSMISSION_{b}"][sel] for b in bands], axis=1)
    ivar = np.stack([t[f"FLUX_IVAR_{b}"][sel] * t[f"MW_TRANSMISSION_{b}"][sel] ** 2
                     for b in bands], axis=1)
    out["FLUX"] = flux.astype(np.float32)
    out["FLUX_IVAR"] = ivar.astype(np.float32)
    iref = bands.index(ref)
    out["REFMAG"] = refmag[sel].astype(np.float32)
    snr = flux[:, iref] * np.sqrt(ivar[:, iref])
    out["REFMAG_ERR"] = (2.5 / np.log(10) / snr).astype(np.float32)
    out["EBV"] = t["EBV"][sel].astype(np.float32)
    out["TYPE"] = np.array([TYPES.index(s) if s in TYPES else 255 for s in typ[sel]], dtype=np.uint8)
    out["MASKBITS"] = t["MASKBITS"][sel].astype(np.int32)

    zspec = np.full(sel.size, -1.0, np.float32)
    zsrc = np.zeros(sel.size, np.uint8)
    zphot = np.full(sel.size, -1.0, np.float32)
    zphot_std = np.full(sel.size, -1.0, np.float32)
    pzfile = pz_path(sweep) if pz == "auto" else pz
    if pzfile is not None and Path(pzfile).exists():
        prow = rows[sel]
        ztab = read_table(pzfile, ["RELEASE", "BRICKID", "OBJID", "Z_SPEC", "SURVEY",
                                   "Z_PHOT_MEDIAN_I", "Z_PHOT_STD_I"], rows=prow)
        same = ls_id(ztab["RELEASE"], ztab["BRICKID"], ztab["OBJID"]) == out["ID"]
        if not np.all(same):
            raise ValueError(f"{pzfile} is not row-matched to {sweep}")
        z = ztab["Z_SPEC"].astype(np.float64)
        surv = ztab["SURVEY"].astype("U11")
        good = (z > cfg.survey.zspec_range[0]) & (z < cfg.survey.zspec_range[1])
        good &= ~np.isin(surv, list(cfg.survey.zspec_exclude_surveys))
        zspec[good] = z[good]
        lookup = {s: i + 1 for i, s in enumerate(SURVEYS)}
        zsrc[good] = [lookup.get(s, 255) for s in surv[good]]
        zp, zs = ztab["Z_PHOT_MEDIAN_I"], ztab["Z_PHOT_STD_I"]
        okp = zp > -1
        zphot[okp], zphot_std[okp] = zp[okp], zs[okp]
    elif pz == "auto":
        if require_pz:
            raise FileNotFoundError(f"no photo-z file for {sweep} (expected {pzfile})")
        log.warning("no photo-z file for %s; ZSPEC/ZPHOT left at -1", sweep)
    out["ZSPEC"] = zspec
    out["ZSPEC_SRC"] = zsrc
    out["ZPHOT"] = zphot
    out["ZPHOT_STD"] = zphot_std
    return out


def concat(tables: list[Table]) -> Table:
    tables = [t for t in tables if t]
    if not tables:
        return {}
    return {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}


def ingest(sweeps: list[str | Path], out: str | Path | None = None, cfg: RemaConfig | None = None,
           box: Box | None = None) -> Table:
    """Ingest several sweeps (restricted to ``box`` if given) into one galaxy table."""
    cfg = cfg or RemaConfig()
    mag_max = default_mag_max(cfg)
    parts = []
    for s in sweeps:
        if box is not None and not box.overlaps(sweep_box(s)):
            continue
        t = ingest_sweep(s, cfg, box=box, mag_max=mag_max)
        log.info("%s: %d galaxies", Path(s).name, len(t.get("ID", [])))
        parts.append(t)
    tab = concat(parts)
    if out is not None:
        header = {**_table_header(cfg, mag_max), "NSWEEPS": len(sweeps), **sky_header(box)}
        write_table(out, tab, header=header, extname="GALAXIES")
    return tab


def _table_header(cfg: RemaConfig, mag_max: float) -> dict:
    return {"BANDS": ",".join(cfg.survey.bands), "REFBAND": cfg.survey.ref_band,
            "MAGMAX": mag_max, "SNRMIN": cfg.survey.ref_snr_min,
            "FLXUNIT": "nanomaggies, dereddened", "SURVHASH": survey_hash(cfg)}


def ingest_sweeps(sweeps: list[str | Path], outdir: str | Path, cfg: RemaConfig | None = None,
                  *, require_pz: bool = True) -> list[Path]:
    """One galaxy table per sweep, ``outdir/<sweep name>``; returns the table paths.

    A table that exists with the same SURVHASH is kept; one made with another survey
    configuration is rewritten. Empty selections give zero-row tables. The header records SWEEP,
    NGAL, NZSPEC, EBVMEAN, SURVHASH and the sweep's box, for :func:`galaxy_counts`.
    """
    cfg = cfg or RemaConfig()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mag_max = default_mag_max(cfg)
    h = survey_hash(cfg)
    out = []
    for s in map(Path, sweeps):
        path = outdir / s.name
        if path.exists():
            old = read_header(path).get("SURVHASH")
            if old == h:
                out.append(path)
                continue
            log.warning("%s was made with SURVHASH %s, not %s: ingesting again", path, old, h)
        t = ingest_sweep(s, cfg, mag_max=mag_max, require_pz=require_pz)
        zs = t["ZSPEC"]
        header = {**_table_header(cfg, mag_max), "SWEEP": s.name, "NGAL": int(zs.size),
                  "NZSPEC": int(np.sum(zs > 0)),
                  "EBVMEAN": float(np.mean(t["EBV"])) if zs.size else 0.0,
                  **sky_header(sweep_box(s))}
        write_table(path, t, header=header, extname="GALAXIES")
        log.info("%s: %d galaxies (%d with ZSPEC)", s.name, zs.size, header["NZSPEC"])
        out.append(path)
    return out


def read_galaxies(source: str | Path, box: Box | BoxUnion | None = None,
                  cfg: RemaConfig | None = None) -> Table:
    """Galaxies of one table, or of a directory of per-sweep tables cut to ``box``.

    A directory needs a box (or a union of boxes): the tables of the sweeps overlapping it are
    read in name order and cut to it. Their SURVHASH must agree with each other and, when
    ``cfg`` is given, with that configuration (tables without one are not checked).
    """
    src = Path(source)
    if src.is_dir():
        if box is None:
            raise ValueError(f"{src} is a directory of per-sweep tables: give a box")
        files = sweeps_overlapping(box, src)
        if not files:
            raise FileNotFoundError(f"no per-sweep galaxy table in {src} overlaps {box}")
    else:
        files = [src]
    parts, hashes = [], set()
    for f in files:
        hashes.add(read_header(f).get("SURVHASH"))
        t = read_table(f)
        if box is not None and t:
            keep = box.contains(t["RA"], t["DEC"])
            t = {k: v[keep] for k, v in t.items()}
        parts.append(t)
    hashes.discard(None)
    if len(hashes) > 1:
        raise ValueError(f"galaxy tables made with different survey configurations: {sorted(hashes)}")
    if cfg is not None and hashes and survey_hash(cfg) not in hashes:
        raise ValueError(f"galaxy tables have SURVHASH {hashes.pop()} but the configuration gives "
                         f"{survey_hash(cfg)}: ingest them with this configuration")
    return concat(parts)


def galaxy_counts(galaxies_dir: str | Path) -> Table:
    """Per-sweep summary of a directory of per-sweep tables (headers only): NAME, the sweep
    box (RA0, RA1, DEC0, DEC1), NGAL, NZSPEC and EBVMEAN."""
    rows = []
    for p in sorted(Path(galaxies_dir).glob("sweep-*.fits")):
        hdr = read_header(p)
        b = sweep_box(p)
        rows.append((p.name, b.ra_min, b.ra_max, b.dec_min, b.dec_max, int(hdr.get("NAXIS2", 0)),
                     int(hdr.get("NZSPEC", 0)), float(hdr.get("EBVMEAN", 0.0))))
    names = ("NAME", "RA0", "RA1", "DEC0", "DEC1", "NGAL", "NZSPEC", "EBVMEAN")
    cols = list(zip(*rows)) if rows else [[] for _ in names]
    out = {n: np.asarray(c) for n, c in zip(names, cols)}
    out["NAME"] = np.asarray(out["NAME"], dtype=str)
    return out
