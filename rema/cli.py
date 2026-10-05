"""Command-line interface: ``rema <command> ...``.

Commands (each reads and writes FITS files, so stages can be resumed and run per region):

  ingest     DR11 sweeps (+ row-matched photo-z) -> compact galaxy table
  maps       DR11 randoms -> footprint/depth map
  calib-import  redMaPPer *_pars.fit -> rema calibration file (red sequence + zred correction)
  background galaxies + calibration -> calibration with the chi^2 background added
  zred       add ZRED columns to a galaxy table
  scan       richness and redshift at given positions (redMaPPer zscan)
  blind      blind cluster finding over a region
  specpost   spectroscopic redshift and velocity dispersion of a cluster catalogue
  calibrate  red-sequence calibration on DR11 (spectroscopic seeds)
  randoms-index  DR11 randoms -> pixel-sorted copy, for fast footprints of any box
  regions    per-sweep galaxy tables -> region plan of a large run
  status     done / missing / stale regions of a run
  merge      region catalogues -> one catalogue, with boundary QA

Boxes: ``--box RA0 RA1 DEC0 DEC1`` (repeatable: a union of boxes) gives the data box, and
``--regions PLAN --region-id I`` takes the own and data boxes of a planned region. ``--galaxies``
is a galaxy table or a directory of per-sweep tables (``rema ingest --outdir``).

Run ``rema <command> -h`` for options. GPU: set ``JAX_PLATFORMS=cuda``.

Configuration: ``--config`` if given, else the configuration stored in the calibration (scan,
blind, zred, background) or in the input catalogue (specpost), else the defaults.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("rema")


def _cfg(path):
    from .config import RemaConfig

    return RemaConfig.from_yaml(path) if path else RemaConfig()


def _config_diff(a, b) -> list[str]:
    """Dotted keys (``section.key``) whose values differ between two configurations."""
    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
        return out

    fa, fb = flat(a.to_dict()), flat(b.to_dict())
    return sorted(k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k))


def _run_cfg(path, stored, source: str):
    """Configuration of a run: ``--config`` if given, else the one stored with the input (the
    calibration, or a catalogue's CONFIG HDU), else the defaults. A ``--config`` that differs from
    the stored one is used, with a warning listing the differences."""
    from .config import RemaConfig

    if path:
        cfg = RemaConfig.from_yaml(path)
        if stored is not None and stored != cfg:
            log.warning("--config differs from the %s's configuration in %s", source,
                        ", ".join(_config_diff(stored, cfg)))
        return cfg
    if stored is not None:
        log.info("configuration: the %s's", source)
        return stored
    log.info("configuration: defaults (the %s stores none)", source)
    return RemaConfig()


def _catalog_cfg(path):
    """Configuration in the CONFIG HDU of a catalogue written by rema (None without one)."""
    import yaml
    from astropy.io import fits

    from .config import RemaConfig

    with fits.open(path) as h:
        try:
            text = h["CONFIG"].data["YAML"][0]
        except KeyError:
            return None
    text = text.decode() if isinstance(text, bytes) else str(text)
    return RemaConfig.from_dict(yaml.safe_load(text))


def _box(vals):
    from .sky.regions import Box

    return Box(*map(float, vals)) if vals else None


def _sky(boxes):
    """A Box, a BoxUnion (several ``--box``) or None."""
    from .sky.regions import Box, sky_union

    return sky_union([Box(*map(float, b)) for b in boxes]) if boxes else None


def _plan_region(args):
    """(own Box, data Box/BoxUnion, plan meta) of ``--regions/--region-id``, or None."""
    if not getattr(args, "regions", None):
        return None
    from .pipeline import plan_boxes, read_plan

    if args.region_id is None:
        raise SystemExit("--regions needs --region-id")
    plan, meta = read_plan(args.regions)
    own, data = plan_boxes(plan, args.region_id, meta)
    return own, data, meta


def _data_sky(args):
    """The data box of a command: from the plan, else from ``--box``."""
    pr = _plan_region(args)
    if pr is not None:
        return pr[1]
    return _sky(getattr(args, "box", None))


def _sha1_text(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _sweeps(items):
    out = []
    for it in items:
        p = Path(it)
        out += sorted(x for x in p.glob("sweep-*.fits") if not x.name.endswith("-pz.fits")) if p.is_dir() else [p]
    return out


def _region(args, need_bkg=True, rebuild_bkg=False, sky=None):
    from .calibration import Calibration
    from .io.legacy import read_galaxies
    from .modes.common import Region
    from .sky.maps import Footprint

    cal = Calibration.read(args.calib)
    cfg = _run_cfg(args.config, cal.config, "calibration")
    gal = read_galaxies(args.galaxies, sky, cfg)
    log.info("%d galaxies from %s%s", gal["ID"].size if gal else 0, args.galaxies,
             f" in {sky}" if sky is not None else "")
    fp = Footprint.read(args.footprint) if getattr(args, "footprint", None) else None
    if fp is not None and sky is not None and fp.box is not None and fp.box != sky:
        log.warning("the footprint was built for %s, not for the data box %s", fp.box, sky)
    if cal.bkg is None and need_bkg:
        log.warning("calibration has no background; building it from the galaxies")
    if getattr(args, "centering", None):
        cfg = cfg.replace(centering={"method": args.centering})
    reg = Region.build(gal, cal.rs, cfg, footprint=fp, zredcorr=cal.zredcorr,
                       bkg=None if rebuild_bkg else cal.bkg, zlcorr=cal.zlcorr,
                       zbkg=None if rebuild_bkg else cal.zbkg, wcen_params=cal.wcen)
    return reg, cal


def _centring(reg) -> str:
    """Resolved centring method of a run (logged, with a warning when a wcen model goes unused)."""
    method = reg.centering_method()
    if method == "bcg" and reg.wcen is not None:
        log.warning("the calibration has a wcen model but the configuration selects BCG centring "
                    "(set centering.method: auto, or wcen, to use it)")
    log.info("centring: %s", method)
    return method


def _write_catalog(path, cat, mem, cfg, extra=None, members_path=None):
    from .io.tables import write_catalog

    write_catalog(path, cat, mem, cfg, extra, members_path=members_path)
    log.info("wrote %s (%d clusters, %d members)", path, len(next(iter(cat.values()), [])) if cat else 0,
             len(next(iter(mem.values()), [])) if mem else 0)


# --------------------------------------------------------------------------- commands
def cmd_ingest(args):
    from .io.legacy import ingest, ingest_sweeps
    from .sky.regions import sweep_box

    cfg = _cfg(args.config)
    sweeps = _sweeps(args.sweeps)
    sky = _sky(args.box)
    t0 = time.time()
    if args.outdir:
        if sky is not None:
            sweeps = [s for s in sweeps if sky.overlaps(sweep_box(s))]
        paths = ingest_sweeps(sweeps, args.outdir, cfg, require_pz=args.require_pz)
        log.info("%d per-sweep tables in %s in %.1fs", len(paths), args.outdir, time.time() - t0)
        return
    if not args.out:
        raise SystemExit("give --out (one table) or --outdir (one table per sweep)")
    tab = ingest(sweeps, args.out, cfg, box=sky)
    log.info("ingested %d galaxies from %d sweeps in %.1fs", len(tab.get("ID", [])), len(sweeps), time.time() - t0)


def cmd_randoms_index(args):
    from .sky.maps import index_randoms

    cfg = _cfg(args.config)
    for path in args.randoms:
        t0 = time.time()
        out = index_randoms(path, args.outdir, cfg, box=_sky(args.box))
        log.info("indexed %s -> %s in %.1fs", path, out, time.time() - t0)


def cmd_maps(args):
    from .sky.maps import build_footprint, randoms_index_files, read_randoms, read_randoms_index

    cfg = _cfg(args.config)
    box = _data_sky(args)
    if args.index:
        stems = randoms_index_files(args.index)[: args.nrand or None]
        rnd = read_randoms_index(args.index, cfg, box, files=stems)
        nfiles = len(stems)
    elif args.randoms:
        parts = [read_randoms(p, cfg, box) for p in args.randoms]
        rnd = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        nfiles = len(args.randoms)
    else:
        raise SystemExit("give randoms files or --index")
    fp = build_footprint(rnd, cfg, box=box, density=cfg.mask.randoms_density * nfiles)
    fp.write(args.out)
    log.info("footprint: %d randoms files, %d pixels, unmasked area %.2f deg2", nfiles,
             fp.fine.pixels.size, fp.area_deg2())


def cmd_calib_import(args):
    from .calibration import Calibration

    cfg = _cfg(args.config)
    cal = Calibration.from_redmapper_pars(args.pars, cfg)
    cal.write(args.out)
    log.info("wrote %s (bands %s, reference %s)", args.out, ",".join(cal.rs.bands), cal.rs.ref_band)


def cmd_background(args):
    from .modes.common import _area_function, zred_background

    # Both backgrounds are rebuilt from the given galaxies.
    reg, cal = _region(args, need_bkg=False, rebuild_bkg=True, sky=_data_sky(args))
    cal.bkg = reg.model.bkg
    cal.zbkg = zred_background(reg.gal, reg.cfg,
                               _area_function(reg.gal, reg.rs, reg.cfg, reg.footprint, None))
    cal.config = reg.cfg
    cal.write(args.out)


def cmd_zred(args):
    from .io.tables import read_table, write_table

    reg, _ = _region(args, need_bkg=False, sky=_data_sky(args))
    cols = {k: reg.gal[k] for k in ("ID", "ZRED", "ZRED_E", "ZRED_UNCORR", "ZRED_UNCORR_E", "ZRED_CHISQ")}
    write_table(args.out, cols, extname="ZRED")


def cmd_scan(args):
    from .io.tables import read_table
    from .modes.scan import run_scan
    from .modes import specpost

    pr = _plan_region(args)
    reg, cal = _region(args, sky=pr[1] if pr else None)
    method = _centring(reg)
    cfg = reg.cfg
    pos = read_table(args.positions)
    ra, dec = pos[args.ra_col.upper()], pos[args.dec_col.upper()]
    ids = pos[args.id_col.upper()] if args.id_col else np.arange(ra.size)
    keep_box = _box(args.box) if args.box else (pr[0] if pr else None)
    if keep_box is not None:
        sel = keep_box.contains(ra, dec)
        ra, dec, ids = ra[sel], dec[sel], ids[sel]
    cat, mem = run_scan(reg, ra, dec, ids, batch=args.batch)
    if args.specpost:
        sc = cfg.spec
        cat, mem = specpost.process(cat, mem, min_members=sc.min_members, vmax_init=sc.vmax_init,
                                    nsigma_clip=sc.nsigma_clip, niter=sc.niter, gapper_nmax=sc.gapper_nmax,
                                    c_location=sc.c_location, c_scale=sc.c_scale, nboot=sc.nboot,
                                    seed=sc.seed)
    _write_catalog(args.out, cat, mem, cfg, {"MODE": "scan", "CENTRING": method})


def cmd_blind(args):
    from .modes.blind import run_blind
    from .modes import specpost

    import jax

    from .pipeline import file_sha1
    from .sky.regions import sky_header

    pr = _plan_region(args)
    sky = pr[1] if pr else _sky(args.box)
    own = pr[0] if pr else _box(args.own)
    region_id = args.region_id if args.region_id is not None else 0
    reg, cal = _region(args, sky=sky)
    method = _centring(reg)
    cfg = reg.cfg
    info = {}
    cat, mem = run_blind(reg, own=own, region_id=region_id, checkpoint=args.checkpoint, info=info)
    hdr = {"MODE": "blind", "CENTRING": method, "REGION": region_id, "CALSHA1": file_sha1(args.calib),
           "CFGSHA1": _sha1_text(cfg.to_yaml()), "DEVICE": jax.default_backend(),
           "NGAL": info.get("n_galaxies", 0), "NSEED": info.get("n_seeds", 0),
           "NFIRST": info.get("n_firstpass", 0), "NCAND": info.get("n_candidates", 0),
           "NDEP": info.get("n_dependencies", 0), "TFIRST": round(info.get("t_firstpass", 0.0), 1),
           "TLIKE": round(info.get("t_likelihood", 0.0), 1),
           "TTOTAL": round(info.get("t_total", 0.0), 1),
           "PEAKRSS": round(info.get("peak_rss_gb", 0.0), 2), **sky_header(sky)}
    if own is not None:
        hdr.update(OWNRA0=own.ra_min, OWNRA1=own.ra_max, OWNDEC0=own.dec_min, OWNDEC1=own.dec_max)
    if pr is not None:
        hdr["PLANHASH"] = pr[2]["PLANHASH"]
    if args.specpost and cat:
        sc = cfg.spec
        cat, mem = specpost.process(cat, mem, min_members=sc.min_members, vmax_init=sc.vmax_init,
                                    nsigma_clip=sc.nsigma_clip, niter=sc.niter, gapper_nmax=sc.gapper_nmax,
                                    c_location=sc.c_location, c_scale=sc.c_scale, nboot=sc.nboot,
                                    seed=sc.seed)
    _write_catalog(args.out, cat, mem, cfg, hdr)


def cmd_specpost(args):
    from .io.tables import read_table
    from .modes import specpost

    from .io.tables import read_catalog

    cfg = _run_cfg(args.config, _catalog_cfg(args.catalog), "catalogue")
    cat, mem, _ = read_catalog(args.catalog)
    if not cat:
        _write_catalog(args.out, cat, mem, cfg, {"MODE": "specpost"})
        return
    sc = cfg.spec
    cat, mem = specpost.process(cat, mem, min_members=sc.min_members, vmax_init=sc.vmax_init,
                                nsigma_clip=sc.nsigma_clip, niter=sc.niter, gapper_nmax=sc.gapper_nmax,
                                c_location=sc.c_location, c_scale=sc.c_scale, nboot=sc.nboot, seed=sc.seed)
    _write_catalog(args.out, cat, mem, cfg, {"MODE": "specpost"})


def cmd_calibrate(args):
    from .calib.driver import calibrate

    calibrate(args)


def cmd_regions(args):
    from .io.legacy import galaxy_counts
    from .pipeline import calib_suggest, plan_regions, required_buffer, uncovered_tiles, write_plan
    from .sky.regions import sweep_box

    cfg = _cfg(args.config)
    counts = galaxy_counts(args.galaxies)
    sky = _sky(args.box)
    if sky is not None:
        keep = np.array([sky.overlaps(sweep_box(n)) for n in counts["NAME"]], bool)
        counts = {k: v[keep] for k, v in counts.items()}
    log.info("%d per-sweep tables, %.0f M galaxies, %d with ZSPEC", counts["NAME"].size,
             counts["NGAL"].sum() / 1e6, counts["NZSPEC"].sum())
    if args.calib_suggest:
        for c in calib_suggest(counts, args.calib_suggest):
            b = c["box"]
            print(f"calib_box={b.ra_min:g} {b.ra_max:g} {b.dec_min:g} {b.dec_max:g} "
                  f"nzspec={c['NZSPEC']} ngal={c['NGAL']} area={c['AREA']:.0f} ebvmax={c['EBVMAX']:.3f}")
        return
    if args.index:
        missing = uncovered_tiles(counts, args.index, sky)
        if missing:
            msg = f"{len(missing)} tiles have randoms but no galaxy table, e.g. {missing[:5]}"
            if not args.allow_uncovered:
                raise SystemExit(msg + " (ingest them, or pass --allow-uncovered)")
            log.warning(msg)
    need = required_buffer(cfg, 100.0)
    if args.buffer < need["first_order"]:
        log.warning("buffer %.2f deg < %.2f deg, the first-order reach of a lambda = 100 cluster at "
                    "z = %.2f", args.buffer, need["first_order"], cfg.model.zrange[0])
    log.info("buffer %.2f deg; exact reach at z = %.2f: %.2f deg (lambda 100)", args.buffer,
             cfg.model.zrange[0], need["exact"])
    plan, meta = plan_regions(counts, target_area=args.target_area, buffer=args.buffer,
                              band_height=args.band_height, polar_cap=args.polar_cap,
                              max_gal=args.max_gal, max_pairs=args.max_pairs, sky=sky)
    write_plan(args.out, plan, meta)
    a, g, pp = plan["AREA_OWN"], plan["NGAL_DATA"], plan["PAIRS_EST"]
    log.info("wrote %s: %d regions; own area %.0f-%.0f deg2 (median %.0f); galaxies in the data box "
             "median %.1f M, max %.1f M; dependency pairs median %.0f M, max %.0f M", args.out,
             meta["NREGION"], a.min(), a.max(), np.median(a), np.median(g) / 1e6, g.max() / 1e6,
             np.median(pp) / 1e6, pp.max() / 1e6)


def cmd_status(args):
    from . import __version__
    from .pipeline import region_status

    if not Path(args.plan).exists():
        raise SystemExit(f"no region plan at {args.plan}: it is made by the first 'run' of the driver "
                         "(rema regions), after 'prepare'")
    st = region_status(args.plan, args.runs, args.calib, __version__ if args.check_version else None)
    prime = "" if st["prime"] is None else st["prime"]
    print(f"done={len(st['done'])} missing={len(st['missing'])} stale={len(st['stale'])} prime={prime}")
    print(f"prime={prime}")
    todo = sorted(st["missing"] + st["stale"])
    print("todo_ids=" + ",".join(map(str, todo)))
    print("todo_array=" + array_spec(todo))
    print("todo_array_noprime=" + array_spec([i for i in todo if i != st["prime"]]))
    print("stale_ids=" + ",".join(map(str, st["stale"])))


def array_spec(ids) -> str:
    """SLURM array specification of integer IDs, with ranges: [0, 1, 2, 5] -> "0-2,5"."""
    ids = sorted(set(int(i) for i in ids))
    out, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(out)


def cmd_todo(args):
    """Array specs of the ingest chunks and randoms files whose outputs are missing."""
    from .sky.regions import sweep_box

    sweeps = _sweeps([args.sweeps])
    sky = _sky(args.box)
    # Chunks index the full sorted list, as in the ingest task, which skips sweeps off the area.
    area = [(i, s) for i, s in enumerate(sweeps) if sky is None or sky.overlaps(sweep_box(s))]
    todo = [(i, s) for i, s in area if not (Path(args.galaxies) / s.name).exists()]
    chunks = [i // args.chunk for i, _ in todo]
    print(f"nsweeps={len(sweeps)} nchunks={(len(sweeps) + args.chunk - 1) // args.chunk}")
    print("ingest_array=" + array_spec(chunks))
    missing = [k for k in range(args.nrand)
               if not (Path(args.index) / f"randoms-south-1-{k}.json").exists()]
    print("randoms_array=" + array_spec(missing))
    print(f"ingest_done={len(area) - len(todo)}/{len(area)}")
    print(f"randoms_done={args.nrand - len(missing)}/{args.nrand}")


def cmd_merge(args):
    import json

    from .io.tables import write_table
    from .pipeline import edge_profile, merge_regions, read_plan

    cat, mem, regions, qa = merge_regions(args.plan, args.runs, allow_missing=args.allow_missing,
                                          calib=args.calib)
    out = Path(args.out)
    members_out = args.members_out or str(out.with_name(out.stem + "_members.fits"))
    _write_catalog(out, cat, mem, None, {"MODE": "merged", "NREGION": qa["n_done"]},
                   members_path=members_out)
    write_table(out.with_name(out.stem + "_regions.fits"), regions, extname="REGIONS")
    if cat:
        plan, meta = read_plan(args.plan)
        prof = edge_profile(cat, plan, meta)
        qa["edge_profile"] = {"d_lo": prof["D_LO"].tolist(), "ratio": np.round(prof["RATIO"], 3).tolist(),
                              "n": prof["N"].tolist()}
    out.with_name(out.stem + "_qa.json").write_text(json.dumps(qa, indent=1, default=str))
    for k, v in qa.items():
        if k not in ("edge_profile", "close_pairs_examples"):
            log.info("qa %s: %s", k, v)


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rema", description=__doc__.split("\n\n")[0])
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, galaxies=True, calib=True, footprint=True):
        sp.add_argument("--config", help="YAML configuration (default: the calibration's)" if calib
                        else "YAML configuration (defaults otherwise)")
        if galaxies:
            sp.add_argument("--galaxies", required=True, help="galaxy table from `rema ingest`")
        if calib:
            sp.add_argument("--calib", required=True, help="calibration file")
        if footprint:
            sp.add_argument("--footprint", help="footprint map from `rema maps`")

    def centering(sp):
        sp.add_argument("--centering", choices=("auto", "bcg", "wcen"),
                        help="centring method (default: the configuration's, 'auto' = wcen when "
                             "the calibration has a wcen model)")

    def boxes(sp, help="data box (repeat for a union of boxes)"):
        sp.add_argument("--box", nargs=4, action="append", metavar=("RA0", "RA1", "DEC0", "DEC1"),
                        help=help)

    def plan(sp):
        sp.add_argument("--regions", metavar="PLAN", help="region plan from `rema regions`")
        sp.add_argument("--region-id", type=int, help="region of the plan (own and data boxes)")

    s = sub.add_parser("ingest", help="DR11 sweeps -> galaxy table(s)")
    s.add_argument("sweeps", nargs="+", help="sweep files or directories")
    boxes(s, "keep galaxies in this box (with --out), or sweeps overlapping it (with --outdir)")
    s.add_argument("--out", help="one galaxy table")
    s.add_argument("--outdir", help="one table per sweep, for large runs")
    s.add_argument("--require-pz", action="store_true",
                   help="a missing photo-z file is an error (production)")
    s.add_argument("--config")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("randoms-index", help="DR11 randoms -> pixel-sorted copies (read once)")
    s.add_argument("randoms", nargs="+")
    s.add_argument("--outdir", required=True)
    boxes(s, "index only the randoms in this box")
    s.add_argument("--config")
    s.set_defaults(func=cmd_randoms_index)

    s = sub.add_parser("maps", help="DR11 randoms -> footprint/depth map")
    s.add_argument("randoms", nargs="*", help="randoms files (or use --index)")
    s.add_argument("--index", help="directory of `rema randoms-index` outputs")
    s.add_argument("--nrand", type=int, help="use the first NRAND indexed files (default all)")
    boxes(s)
    plan(s)
    s.add_argument("--out", required=True)
    s.add_argument("--config")
    s.set_defaults(func=cmd_maps)

    s = sub.add_parser("calib-import", help="redMaPPer pars file -> rema calibration")
    s.add_argument("pars")
    s.add_argument("--out", required=True)
    s.add_argument("--config")
    s.set_defaults(func=cmd_calib_import)

    s = sub.add_parser("background", help="add the chi^2 and zred backgrounds to a calibration")
    common(s)
    boxes(s)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_background)

    s = sub.add_parser("zred", help="zred of a galaxy table")
    common(s, footprint=False)
    boxes(s)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_zred)

    s = sub.add_parser("scan", help="richness/redshift at given positions")
    common(s)
    centering(s)
    plan(s)
    s.add_argument("--positions", required=True)
    s.add_argument("--ra-col", default="RA")
    s.add_argument("--dec-col", default="DEC")
    s.add_argument("--id-col")
    s.add_argument("--box", nargs=4, metavar=("RA0", "RA1", "DEC0", "DEC1"),
                   help="scan only the positions in this box (default with --regions: its own box)")
    s.add_argument("--batch", type=int, default=128)
    s.add_argument("--specpost", action="store_true", help="also run the spectroscopic post-processing")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("blind", help="blind cluster finding")
    common(s)
    centering(s)
    boxes(s)
    s.add_argument("--regions", metavar="PLAN", help="region plan from `rema regions`")
    s.add_argument("--own", nargs=4, metavar=("RA0", "RA1", "DEC0", "DEC1"),
                   help="keep clusters centred in this box (the region without its buffer)")
    s.add_argument("--region-id", type=int,
                   help="region number (MEM_MATCH_ID = region << 32 | rank); with --regions, the "
                        "planned region")
    s.add_argument("--checkpoint", metavar="DIR",
                   help="keep the first-pass and likelihood results here; a rerun resumes from them")
    s.add_argument("--specpost", action="store_true")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_blind)

    s = sub.add_parser("specpost", help="spectroscopic post-processing of a catalogue")
    s.add_argument("catalog")
    s.add_argument("--out", required=True)
    s.add_argument("--config", help="YAML configuration (default: the catalogue's CONFIG HDU)")
    s.set_defaults(func=cmd_specpost)

    s = sub.add_parser("calibrate", help="red-sequence calibration on DR11")
    common(s, calib=False)
    boxes(s, "calibration area (repeat for several patches)")
    s.add_argument("--init-pars", help="optional redMaPPer pars file for the initial colours")
    s.add_argument("--plots", help="directory for diagnostic plots")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("regions", help="plan the regions of a large run")
    s.add_argument("--galaxies", required=True, help="directory of per-sweep tables")
    s.add_argument("--index", help="randoms index: tiles with randoms but no galaxies are an error")
    s.add_argument("--allow-uncovered", action="store_true")
    boxes(s, "plan only this area (data boxes are cut to it)")
    s.add_argument("--target-area", type=float, default=100.0, help="own area per region (deg2)")
    s.add_argument("--buffer", type=float, default=2.0, help="data box = own box + buffer (deg)")
    s.add_argument("--band-height", type=float, default=10.0, help="Dec band height (deg)")
    s.add_argument("--polar-cap", type=float, default=85.0, help="|Dec| above which a cap is one region")
    s.add_argument("--max-gal", type=float, help="split regions with more data-box galaxies")
    s.add_argument("--max-pairs", type=float, help="split regions with more estimated pairs")
    s.add_argument("--calib-suggest", type=float, metavar="AREA",
                   help="instead of planning, print calibration boxes of about AREA deg2")
    s.add_argument("--out", help="plan file (FITS)")
    s.add_argument("--config")
    s.set_defaults(func=cmd_regions)

    s = sub.add_parser("status", help="done / missing / stale regions")
    s.add_argument("--plan", required=True)
    s.add_argument("--runs", required=True, help="directory with one NNNN/clusters.fits per region")
    s.add_argument("--calib", help="stale when made with another calibration")
    s.add_argument("--check-version", action="store_true", help="stale when made by another rema version")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("todo", help="missing ingest chunks and randoms indexes (for the driver)")
    s.add_argument("--sweeps", required=True, help="directory of sweep files")
    s.add_argument("--galaxies", required=True, help="directory of per-sweep tables")
    s.add_argument("--index", required=True, help="randoms index directory")
    s.add_argument("--chunk", type=int, default=20, help="sweeps per ingest task")
    s.add_argument("--nrand", type=int, default=20, help="randoms files to index")
    boxes(s, "only the sweeps overlapping this area (the driver's AREA_BOX)")
    s.set_defaults(func=cmd_todo)

    s = sub.add_parser("merge", help="merge region catalogues, with QA")
    s.add_argument("--plan", required=True)
    s.add_argument("--runs", required=True)
    s.add_argument("--out", required=True, help="merged clusters (also <out>_regions.fits, <out>_qa.json)")
    s.add_argument("--members-out", help="merged members (default <out>_members.fits)")
    s.add_argument("--calib", help="treat regions made with another calibration as stale")
    s.add_argument("--allow-missing", action="store_true")
    s.set_defaults(func=cmd_merge)
    return p


def _setup_jax():
    """Persistent compilation cache and no GPU memory preallocation (before JAX starts)."""
    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    cache = os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.path.expanduser("~/.cache/rema/jax")
    import jax

    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    _setup_jax()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
