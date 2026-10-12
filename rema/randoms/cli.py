"""``rema cluster-randoms <step>``: the steps of :mod:`rema.randoms.samples` on files.

  domain    merged runs -> the area they cover (own boxes of the finished regions, |b| cut)
  vlim-map  footprint (+ domain) -> z_vlim map
  copies    catalogue + sample spec + z_vlim map + domain + DR11 randoms -> copies of the clusters
  place     copies + footprint -> SCALEVAL, MASKFRAC, R_LAMBDA of every copy
  sample    catalogue + spec + placed copies -> the sample and its weighted randoms
  qa        sample -> data / randoms ratio against depth, |b|, seam distance (JSON)
  external-join  scan at an X-ray/SZ catalogue -> joined with that catalogue's columns (EXT_*)
  erass1-model  eRASS1 joined catalogue + exposure and HI maps -> approximate detection model
  dump-config  catalogue -> its configuration (YAML), for ``rema randoms-index`` and ``rema maps``

The catalogue configuration (m*(z), reference band, apertures, catalogue cuts) is the CONFIG HDU
of the catalogue (``--catalog``/``--clusters``), else ``--config``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("rema")


def _cfg(args, path=None):
    from ..cli import _catalog_cfg
    from ..config import RemaConfig

    if getattr(args, "config", None):
        return RemaConfig.from_yaml(args.config)
    cfg = _catalog_cfg(path) if path else None
    if cfg is None:
        log.info("configuration: defaults")
        return RemaConfig()
    return cfg


def _read_catalog(path, columns):
    from ..io.tables import read_table

    hdu = "CLUSTERS"
    from astropy.io import fits

    with fits.open(path) as h:
        if hdu not in h:
            hdu = 1
        names = {n.upper() for n in h[hdu].columns.names}
    missing = [c for c in columns if c not in names]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    return read_table(path, columns=columns, hdu=hdu)


def _spec(args):
    from .samples import SampleSpec

    spec = SampleSpec.from_yaml(args.spec) if args.spec else SampleSpec()
    if getattr(args, "n_samp", None):
        spec = SampleSpec.from_dict({**spec.to_dict(), "n_samp": args.n_samp})
    return spec


def _randoms_sources(args) -> list:
    from ..sky.maps import randoms_index_files

    if args.index:
        stems = randoms_index_files(args.index)[: args.nrand or None]
        return [Path(args.index) / s for s in stems]
    if not args.randoms:
        raise SystemExit("give DR11 randoms files or --index")
    return [Path(p) for p in args.randoms[: args.nrand or None]]


# --------------------------------------------------------------------------- commands
def cmd_domain(args):
    from .domain import RunDomain

    d = RunDomain.from_runs(args.runs, glat_min=args.glat_min, seam_buffer_deg=args.seam_buffer)
    d.write(args.out)
    log.info("domain: %d own boxes in %d parts, |b| >= %.1f, %d seam segments, buffer %.2f deg -> %s",
             len(d.boxes), len(set(d.parts.tolist())), d.glat_min, len(d.seams), d.seam_buffer_deg,
             args.out)


def cmd_vlim_map(args):
    from ..sky.maps import Footprint
    from .config import VlimConfig
    from .domain import RunDomain
    from .vlim import zvlim_map_from_footprint

    cfg = _cfg(args, args.catalog)
    fp = Footprint.read(args.footprint)
    dom = RunDomain.read(args.domain) if args.domain else None
    vc = VlimConfig(lstar_frac=args.lstar, nsigma=args.nsigma, band=args.band, nside=fp.nside)
    t0 = time.time()
    chunks, density = None, None
    if args.index or args.randoms:
        from .generate import iter_survey_randoms

        sources = _randoms_sources(args)
        chunks = iter_survey_randoms(sources, cfg)
        density = (args.density or cfg.mask.randoms_density) * len(sources)
    else:
        log.warning("no randoms given: the area of pixels cut by the survey edge is overestimated")
    zm = zvlim_map_from_footprint(fp, cfg, dom, vc, randoms=chunks, density=density)
    zm.write(args.out)
    a = zm.area.sum()
    q = {z: zm.area_above(z) / a for z in (0.3, 0.5, 0.6, 0.7, 0.8)}
    log.info("z_vlim map (%.2f L*, %.0f sigma, %s band): %d pixels, %.0f deg2; area fraction above z: %s "
             "(%.0fs) -> %s", vc.lstar_frac, vc.nsigma, zm.meta["BAND"], zm.pix.size, a,
             ", ".join(f"{z}: {f:.2f}" for z, f in q.items()), time.time() - t0, args.out)


def cmd_copies(args):
    from ..abundance.area import ZvlimMap
    from .domain import RunDomain
    from .generate import PositionPool, iter_survey_randoms
    from .products import write_copies
    from .samples import make_copies, select_sources, source_table
    from .selection import get_selection

    cfg = _cfg(args, args.clusters)
    spec = _spec(args)
    sel = get_selection(spec.selection)
    zmap, dom = ZvlimMap.read(args.zvlim), RunDomain.read(args.domain)
    cat = _read_catalog(args.clusters, spec.columns(sel) + (["MEM_MATCH_ID"] if args.with_id else []))
    keep, ann = select_sources(cat, spec, zmap, dom, sel, cfg)
    rows = np.flatnonzero(keep)
    log.info("%s: %d of %d clusters in the sample", spec.name, rows.size, keep.size)
    src = source_table(cat, rows, ann, spec, sel)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    pool = PositionPool.from_chunks(iter_survey_randoms(_randoms_sources(args), cfg), cfg, zmap, dom,
                                    extra_mask=None if sel.name == "none" else sel.in_footprint,
                                    max_points=args.max_pool, rng=rng)
    log.info("position pool: %d of %d random points (%.0fs)", pool.size, pool.meta["NREAD"], time.time() - t0)
    rand, src = make_copies(src, pool, spec.n_samp, args.seed)
    nflag = int((src["FLAG"] != 0).sum())
    if nflag:
        log.warning("%d clusters have no pool point with z_vlim above their redshift", nflag)
    hdr = {"SAMPLE": spec.name, "NSAMP": spec.n_samp, "SEED": args.seed, "NPOOL": pool.size,
           "CATALOG": str(args.clusters), "ZVLIMF": str(args.zvlim), "DOMDIG": dom.digest(),
           **{k: zmap.meta[k] for k in ("LSTAR", "NSIGMA", "BAND") if k in zmap.meta},
           **sel.provenance()}
    write_copies(args.out, rand, src, hdr, spec.to_yaml(), cfg.to_yaml())
    log.info("wrote %s: %d copies of %d clusters", args.out, rand["RA"].size, rows.size)


def cmd_place(args):
    import yaml

    from ..config import RemaConfig
    from ..sky.maps import Footprint
    from .aperture import footprint_map
    from .products import read_copies, write_copies
    from .samples import place

    rand, src, hdr, spec_yaml, cfg_yaml = read_copies(args.copies)
    cfg = RemaConfig.from_yaml(args.config) if args.config else RemaConfig.from_dict(yaml.safe_load(cfg_yaml))
    fp = Footprint.read(args.footprint)
    t0 = time.time()
    rand = place(rand, src, footprint_map(fp, cfg.survey.ref_band), cfg, batch=args.batch)
    hdr.update(FOOTDIG=fp.digest()[:12], PLACED=True)
    write_copies(args.out or args.copies, rand, src, hdr, spec_yaml, cfg_yaml)
    ok = rand["MASKFRAC"] < cfg.mask.max_maskfrac
    log.info("placed %d copies (%.0fs): %.3f with MASKFRAC < %.2f, median SCALEVAL %.3f",
             rand["RA"].size, time.time() - t0, ok.mean() if ok.size else 0, cfg.mask.max_maskfrac,
             float(np.median(rand["SCALEVAL"][ok])) if ok.any() else np.nan)


def cmd_sample(args):
    from ..abundance.area import ZvlimMap
    from .domain import RunDomain
    import yaml

    from .products import read_copies, write_sample
    from .samples import SampleSpec, build_sample, select_sources
    from .selection import get_selection

    rand, src, hdr, spec_yaml, cfg_yaml = read_copies(args.copies)
    if "SCALEVAL" not in rand:
        raise SystemExit(f"{args.copies}: run `rema cluster-randoms place` first")
    cfg = _cfg(args, args.clusters)
    spec = _spec(args) if args.spec else SampleSpec.from_dict(yaml.safe_load(spec_yaml))
    sel = get_selection(spec.selection)
    zmap, dom = ZvlimMap.read(args.zvlim), RunDomain.read(args.domain)
    extra = [c.upper() for c in (args.keep or [])]
    cat = _read_catalog(args.clusters, sorted(set(spec.columns(sel) + extra)))
    keep, ann = select_sources(cat, spec, zmap, dom, sel, cfg)
    rows = np.flatnonzero(keep)
    same = {"ZVLIMF": str(args.zvlim), "DOMDIG": dom.digest()}
    stricter = {k: v for k, v in same.items() if hdr.get(k) != v}
    if stricter:
        log.info("z_vlim map or domain differ from the copies' (%s): copies outside them are dropped",
                 ", ".join(stricter))
    idx, wcols, diag = build_sample(src, rand, rows, ann["LAMBDA_DET"], sel, spec.maskfrac_max(cfg),
                                    zmap=zmap if "ZVLIMF" in stricter else None,
                                    domain=dom if "DOMDIG" in stricter else None)
    data = {**{k: np.asarray(v)[rows] for k, v in cat.items()}, "ROW": rows,
            "ZVLIM": ann["ZVLIM"][rows].astype(np.float32), "BIN_Z": ann["BIN_Z"][rows],
            "BIN_OBS": ann["BIN_OBS"][rows], "WEIGHT": np.ones(rows.size)}
    s = rand["SRC"][idx]
    row = src["ROW"][s]
    rout = {"RA": rand["RA"][idx], "DEC": rand["DEC"][idx], "Z": src["Z"][s], "WEIGHT": wcols["WEIGHT"],
            "SRC": s, "ROW": row, "LAMBDA": src["LAMBDA"][s],
            "OBS": np.asarray(cat[spec.observable], np.float64)[row],
            "BIN_Z": ann["BIN_Z"][row], "BIN_OBS": ann["BIN_OBS"][row], "Q": wcols["Q"],
            "P_EXT": wcols["P_EXT"], "SCALEVAL": rand["SCALEVAL"][idx], "MASKFRAC": rand["MASKFRAC"][idx],
            "ZVLIM": rand["ZVLIM"][idx]}
    keep_w = rout["WEIGHT"] > 0
    rout = {k: v[keep_w] for k, v in rout.items()}
    srcsel = np.isin(src["ROW"], rows)
    sdiag = {"ROW": src["ROW"][srcsel], "N_SAMP": src["N_SAMP"][srcsel], **diag}
    nzero = int((diag["SUMQ"] == 0).sum())
    if nzero:
        log.warning("%d clusters have no copy that passes the cuts (no randoms): raise n_samp", nzero)
    hdr2 = {**{k: v for k, v in hdr.items() if k in ("NSAMP", "SEED", "NPOOL", "ZVLIMF", "DOMDIG",
                                                     "LSTAR", "NSIGMA", "BAND", "FOOTDIG")},
            "SAMPLE": spec.name, "CATALOG": str(args.clusters), "COPIES": str(args.copies),
            "NDATA": int(rows.size), "NRAND": int(keep_w.sum()), "SUMW": float(rout["WEIGHT"].sum()),
            **sel.provenance()}
    write_sample(args.out, data, rout, sdiag, hdr2, spec.to_yaml(), cfg.to_yaml())
    log.info("%s: %d clusters, %d randoms (sum of weights %.0f = %.1f x the clusters) -> %s", spec.name,
             rows.size, keep_w.sum(), hdr2["SUMW"], hdr2["SUMW"] / max(rows.size, 1), args.out)


def cmd_external_join(args):
    from ..io.tables import read_table, table_hdu, write_fits
    from .external import join_scan, read_original

    scan = read_table(args.scan, hdu="CLUSTERS")
    scan = {k: v for k, v in scan.items() if np.asarray(v).ndim == 1}
    out = join_scan(scan, read_original(args.original, args.catalogue), args.catalogue)
    from ..cli import _catalog_cfg

    cfg = _catalog_cfg(args.scan)
    hdus = [table_hdu(out, {"EXTCAT": args.catalogue, "SCAN": str(args.scan),
                            "ORIGINAL": str(args.original)}, "CLUSTERS")]
    if cfg is not None:
        hdus.append(table_hdu({"YAML": np.array([cfg.to_yaml().encode()])}, extname="CONFIG"))
    write_fits(args.out, hdus)
    log.info("%s: %d rows with %d EXT_ columns -> %s", args.catalogue, len(out["INPUT_ROW"]),
             sum(k.startswith("EXT_") for k in out), args.out)


def cmd_erass1_model(args):
    import yaml
    from astropy.cosmology import FlatLambdaCDM

    from ..io.tables import read_table, table_hdu, write_fits
    from .products import _read_text
    from .selection.erass1 import HealpixColumn, fit_detection_model, theta500_arcmin

    j = read_table(args.joined, hdu="CLUSTERS")
    ra, dec = j[args.ra_col.upper()], j[args.dec_col.upper()]
    texp = HealpixColumn(args.expmap, args.exp_column)(ra, dec)
    nh = HealpixColumn(args.nhmap, args.nh_column)(ra, dec)
    # R500 of the catalogue at its own redshift, in its cosmology (flat, H0 = 70, Omega_m = 0.3)
    theta = theta500_arcmin(j["EXT_R500"], j["EXT_Z"], FlatLambdaCDM(H0=70.0, Om0=0.3))
    m = fit_detection_model(j["EXT_CR500"], j["EXT_F500"], j["EXT_EXT_LIKE"], j["EXT_EXP"], texp, nh, theta)
    m.update(expmap=str(args.expmap), nhmap=str(args.nhmap), joined=str(args.joined))
    Path(args.out_model).write_text(yaml.safe_dump(m, sort_keys=False))
    j["EXT_NH"], j["EXT_THETA500"] = nh, theta
    hdus = [table_hdu(j, {"EXTCAT": "erass1"}, "CLUSTERS")]
    cfg = _read_text(args.joined, "CONFIG")
    if cfg:
        hdus.append(table_hdu({"YAML": np.array([cfg.encode()])}, extname="CONFIG"))
    write_fits(args.out, hdus)
    log.info("eRASS1 approximate model (%d clusters): exposure = %.3f x reference, n_abs = %.2g cm^-2 (%s), "
             "ln EXT_LIKE = %.2f + %.2f ln C + %.2f ln theta500, scatter %.2f -> %s, %s", m["n_fit"],
             m["exp_scale"], m["n_abs"], "fitted" if m["n_abs_fitted"] else "default", m["c0"], m["c1"], m["c2"],
             m["sigma"], args.out_model, args.out)


def cmd_dump_config(args):
    cfg = _cfg(args, args.catalog)
    cfg.to_yaml(args.out)
    log.info("wrote %s", args.out)


def cmd_qa(args):
    import json

    from ..abundance.area import ZvlimMap
    from ..io.tables import read_table
    from ..pipeline import galactic_latitude
    from .domain import RunDomain
    from .qa import sample_qa

    data, rand = read_table(args.sample, hdu="CLUSTERS"), read_table(args.sample, hdu="RANDOMS")
    zmap = ZvlimMap.read(args.zvlim)
    q = {"ZVLIM": lambda ra, dec: zmap.value_at(ra, dec)}
    for c in zmap.extra:
        if c.startswith("DEPTH_") or c == "FRACGOOD":
            q[c] = (lambda col: lambda ra, dec: zmap.value_at(ra, dec, col))(c)
    q["ABS_GLAT"] = lambda ra, dec: np.abs(galactic_latitude(ra, dec))
    if args.domain:
        dom = RunDomain.read(args.domain)
        if dom.seams:
            q["SEAM_DIST"] = dom.seam_distance
    out = sample_qa(data, rand, q, nbins=args.nbins)
    for name, res in out.items():
        a = res["all"]
        log.info("%-10s chi2 %.1f / %d; ratio %s", name, a["CHI2"], a["NDOF"],
                 " ".join(f"{r:.3f}" for r in a["RATIO"]))
    plain = {n: {k: {c: (v.tolist() if isinstance(v, np.ndarray) else v) for c, v in t.items()}
                 for k, t in res.items()} for n, res in out.items()}
    Path(args.out).write_text(json.dumps(plain, indent=1))
    log.info("wrote %s", args.out)


# --------------------------------------------------------------------------- parsers
def randoms_args(s, what: str):
    s.add_argument("randoms", nargs="*", help=f"DR11 randoms files (or --index), for {what}")
    s.add_argument("--index", help="randoms index directory (`rema randoms-index`)")
    s.add_argument("--nrand", type=int, help="use the first NRAND randoms files")
    s.add_argument("--density", type=float, help="random points per deg2 in one file (default: the "
                   "configuration's mask.randoms_density, 2500 for DR11)")


def add_parser(sub):
    p = sub.add_parser("cluster-randoms", help="random catalogues of volume-limited cluster samples")
    steps = p.add_subparsers(dest="step", required=True)

    s = steps.add_parser("domain", help="merged runs -> their area (own boxes of finished regions)")
    s.add_argument("--runs", nargs="+", required=True, help="merged run directories (part k = k-th)")
    s.add_argument("--glat-min", type=float, help="|b| cut [deg] (default: the runs')")
    s.add_argument("--seam-buffer", type=float, default=0.0,
                   help="leave out the area within this distance [deg] of the seams between parts")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_domain)

    s = steps.add_parser("vlim-map", help="footprint -> z_vlim map")
    s.add_argument("--footprint", required=True, help="footprint from `rema maps` (whole area)")
    s.add_argument("--domain", help="area of the run (`rema cluster-randoms domain`)")
    s.add_argument("--catalog", help="catalogue whose CONFIG HDU gives the configuration")
    s.add_argument("--config")
    s.add_argument("--lstar", type=float, default=0.2, help="luminosity limit in L* (default 0.2)")
    s.add_argument("--nsigma", type=float, default=5.0, help="depth significance (default 5)")
    s.add_argument("--band", help="depth band (default: the reference band)")
    randoms_args(s, "the area: good random points in the domain (recommended)")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_vlim_map)

    def sample_inputs(s):
        s.add_argument("--clusters", required=True, help="cluster catalogue (CLUSTERS HDU)")
        s.add_argument("--spec", help="sample YAML (default: a richness sample, see SampleSpec)")
        s.add_argument("--zvlim", required=True, help="z_vlim map")
        s.add_argument("--domain", required=True, help="domain file")
        s.add_argument("--config", help="configuration (default: the catalogue's CONFIG HDU)")

    s = steps.add_parser("copies", help="copies of the sample's clusters at random points")
    sample_inputs(s)
    randoms_args(s, "the positions of the copies")
    s.add_argument("--n-samp", type=int, help="copies per cluster (default: the spec's)")
    s.add_argument("--max-pool", type=int, help="keep at most this many pool points")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--with-id", action="store_true", help="carry MEM_MATCH_ID")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_copies)

    s = steps.add_parser("place", help="aperture completeness of the copies")
    s.add_argument("--copies", required=True)
    s.add_argument("--footprint", required=True, help="footprint (`rema maps`) of the whole area")
    s.add_argument("--config", help="configuration (default: the one stored with the copies)")
    s.add_argument("--batch", type=int, default=16384)
    s.add_argument("--out", help="output (default: update the copies file)")
    s.set_defaults(func=cmd_place)

    s = steps.add_parser("sample", help="the sample and its weighted randoms")
    sample_inputs(s)
    s.add_argument("--copies", required=True, help="placed copies of (a superset of) the sample")
    s.add_argument("--keep", action="append", metavar="COLUMN", help="also keep these catalogue columns")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_sample)

    s = steps.add_parser("external-join", help="scan at an X-ray/SZ catalogue + its own columns (EXT_*)")
    s.add_argument("--catalogue", required=True, help="act_dr6, spt_2500d, psz2, erass1 or mcxc")
    s.add_argument("--scan", required=True, help="output of `rema scan-positions` on that catalogue")
    s.add_argument("--original", required=True, help="the original catalogue file")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_external_join)

    s = steps.add_parser("erass1-model", help="fit the approximate eRASS1 detection model")
    s.add_argument("--joined", required=True, help="external-join output for erass1")
    s.add_argument("--expmap", required=True, help="reference exposure map (HEALPix)")
    s.add_argument("--exp-column", default=None)
    s.add_argument("--nhmap", required=True, help="HI4PI NHI_HPX.fits")
    s.add_argument("--nh-column", default="NHI")
    s.add_argument("--ra-col", default="INPUT_RA")
    s.add_argument("--dec-col", default="INPUT_DEC")
    s.add_argument("--out-model", required=True, help="YAML of the model")
    s.add_argument("--out", required=True, help="the joined catalogue with EXT_NH and EXT_THETA500")
    s.set_defaults(func=cmd_erass1_model)

    s = steps.add_parser("dump-config", help="the configuration stored with a catalogue, as YAML")
    s.add_argument("--catalog", required=True)
    s.add_argument("--config", help=argparse.SUPPRESS)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_dump_config)

    s = steps.add_parser("qa", help="data / randoms ratio against depth, |b|, seams, ...")
    s.add_argument("--sample", required=True, help="sample file (`cluster-randoms sample`)")
    s.add_argument("--zvlim", required=True)
    s.add_argument("--domain")
    s.add_argument("--nbins", type=int, default=8)
    s.add_argument("--out", required=True, help="JSON table of the ratios")
    s.set_defaults(func=cmd_qa)
    return p
