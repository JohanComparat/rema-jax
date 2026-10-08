"""Red-sequence calibration on DR11 with spectroscopic seeds (``rema calibrate``).

1. Initial model: the BC03 template colours (``rema/data``) refined on the spectroscopic
   galaxies by iterative clipping (:mod:`rema.calib.init`).
2. EM iterations (``calib.niter``):

   a. zred of all galaxies and the chi^2 background with the current model;
   b. spectroscopic seeds: galaxies with ZSPEC in range, brighter than m*(z) + 1 and with
      chi^2(ZSPEC) < chisq_max;
   c. cluster runs at fixed z = ZSPEC (likelihood pass, then percolation with fixed z), so that
      each cluster counts once;
   d. M-step: joint fit of the red-sequence nodes to the members of clusters with
      lambda >= ``calib.minlambda``, weighted by pmem (:mod:`rema.calib.fit`); pivot magnitudes
      from the members' weighted medians;
   e. zred correction from spectroscopic galaxies among the members (pmem > 0.5);
3. z_lambda correction: z_lambda of the calibration clusters, started at the seed's ZSPEC,
   compared with ZSPEC; and the z -> zred_uncorr mapping of LNCGLIKE: the median zred of their
   central galaxies against that raw z_lambda (redMaPPer's zred_uncorr).
4. wcen centring model (``calib.wcen``), as redMaPPer's WcenCalibrator on BCG-centred clusters
   (calib_niter = 1, as in the DR10 run): the ln W distributions of random points and random satellites
   within the clusters (re-run at fixed z), the central-magnitude model and the central ln W;
   with ``calib.wcen_niter = 2`` the clusters are re-run with wcen centring and the central
   terms refitted on the mixture (P_CEN, P_SAT).
5. The calibration file holds the model, the corrections, the backgrounds (chi^2 and zred), the
   wcen parameters and the configuration.
"""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np

from ..calibration import Calibration
from ..config import RemaConfig, parse_cosmology_overrides
from ..core.richness import RadialQuad, Stage
from ..io.tables import read_table
from ..model.profiles import MStar
from ..model.redsequence import RSModel
from ..modes.blind import _run_batched, percolate
from ..modes.common import Region, _area_function, zred_background
from .corrections import fit_zlambda_correction, fit_zred_correction, fit_zred_uncorr
from .fit import fit_redsequence, update_pivot
from .init import initial_model

log = logging.getLogger(__name__)


def spec_seeds(region: Region, cfg: RemaConfig, dmag: float = 1.0) -> np.ndarray:
    """Indices of spectroscopic galaxies usable as cluster seeds."""
    import jax.numpy as jnp

    from ..model.likelihood import chisq

    g = region.gal
    zs = g["ZSPEC"]
    zr = cfg.model.zrange
    ok = (zs >= zr[0]) & (zs <= zr[1])
    ok &= g["REFMAG"] < region.mstar_np(np.clip(zs, 0.01, 2.0)) + dmag
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return idx
    bidx = region.band_idx
    c2, _ = chisq(jnp.asarray(g["FLUX"][idx][:, bidx]), jnp.asarray(g["FLUX_IVAR"][idx][:, bidx]),
                  region.rs.at(jnp.asarray(zs[idx], jnp.float32)), region.rs.iref,
                  cfg.model.chisq_mode, cfg.survey.flux_floor)
    return idx[np.asarray(c2) < cfg.model.chisq_max]


def _stage(cfg: RemaConfig):
    rc = cfg.richness
    st = Stage.make(rc.percolation.r0, rc.percolation.beta, rc.maxrad_factor)
    q = RadialQuad.make(rmax=rc.percolation.r0 * 20.0 ** rc.percolation.beta + 5 * cfg.model.rsig)
    return st, q


def calibration_candidates(region: Region, seeds: np.ndarray, cfg: RemaConfig) -> dict:
    """Likelihood pass of the spectroscopic seeds at their ZSPEC (keepz candidates)."""
    st, q = _stage(cfg)
    zs = region.gal["ZSPEC"][seeds]
    lk = _run_batched(region, seeds, zs, st, q, "richness")
    lnlike = lk["LNLAMLIKE"].astype(np.float64) + lk["LNCGLIKE"].astype(np.float64)
    keep = (lk["LAMBDA"] >= cfg.richness.minlambda) & np.isfinite(lnlike) & (lk["NINCUT"] >= 3)
    return {"GI": seeds[keep], "Z_LAMBDA": zs[keep].astype(np.float32), "LAMBDA": lk["LAMBDA"][keep],
            "LNLIKE": lnlike[keep], "LNCGLIKE": lk["LNCGLIKE"][keep]}


def calibration_clusters(region: Region, seeds: np.ndarray, cfg: RemaConfig, centering: str = "bcg",
                         cand: dict | None = None):
    """Spectroscopically seeded clusters at fixed redshift (percolated)."""
    st, q = _stage(cfg)
    if cand is None:
        cand = calibration_candidates(region, seeds, cfg)
    return percolate(region, cand, st, q, keepz=True, centering=centering, seed=cfg.calib.seed)


def _central_terms(region: Region, cat: dict):
    """Per-cluster inputs of the central-magnitude fit: m*(z), chi^2 pdf of the central's colours
    (cwt) and the chi^2-background counts within r_lambda (bcounts), as redMaPPer's WcenCalibrator."""
    import jax.numpy as jnp

    from . import wcen as W

    z = np.asarray(cat["Z_LAMBDA"], np.float64)
    chisq = np.asarray(cat["CHISQ"], np.float64)
    refmag = np.asarray(cat["REFMAG"], np.float64)
    sg = np.asarray(region.model.bkg.lookup(jnp.asarray(z, jnp.float32), jnp.asarray(chisq, jnp.float32),
                                            jnp.asarray(refmag, jnp.float32)), np.float64)
    return (region.mstar_np(z), W.chisq_pdf(chisq, region.rs.ncol),
            W.background_counts(sg, region.mpc_per_deg_np(z), cat["R_LAMBDA"]))


MIN_WCEN_SAMPLE = 30      # below this many clusters a wcen fit is flagged as poorly constrained


def calibrate_wcen(region: Region, seeds: np.ndarray, cfg: RemaConfig, cand: dict | None = None,
                   cat: dict | None = None) -> tuple[dict, dict]:
    """Fit the wcen centring model on spectroscopically seeded, BCG-centred clusters.

    Returns (WCEN parameters, info). Random points and random satellites are drawn within the
    consolidated clusters (MASKFRAC < max_maskfrac, lambda/S > minlambda) whose BCG is their own
    spectroscopic seed (redMaPPer's sub_like sample) and re-run at fixed z; their ln W give the
    foreground and satellite distributions. With ``calib.wcen_niter`` >= 2 the clusters seeded
    on those centres are re-run with wcen centring and the central terms refitted.

    Empty or too small samples give undefined (NaN) parameters with a warning; runs then use BCG
    centring. Deviation from redMaPPer: clusters of these fixed-z runs are not rejected when
    z_lambda would fail at their new centre (rema's fixed-z runs do not compute it).
    """
    from . import wcen as W

    pivot = cfg.centering.wcen_pivot
    minlam = cfg.richness.minlambda
    st, q = _stage(cfg)
    if cand is None:
        cand = calibration_candidates(region, seeds, cfg)
    if cat is None:
        cat, _ = calibration_clusters(region, seeds, cfg, centering="bcg", cand=cand)
    g = region.gal

    def table(c):
        """(lambda/S, training selection) of a cluster catalogue; empty for no clusters."""
        if not c:
            return np.zeros(0), np.zeros(0, bool)
        lam_s = np.asarray(c["LAMBDA"], np.float64) / np.asarray(c["SCALEVAL"], np.float64)
        sel = W.select_training(c["LAMBDA"], c["SCALEVAL"], c["W"], c["Z_LAMBDA"], c["MASKFRAC"], cfg)
        return lam_s, sel

    def lnw_fit(c, name):
        lam_s, sel = table(c)
        if sel.sum() < 2:
            log.warning("wcen: %d usable %s points; LNW_%s undefined", sel.sum(), name, name.upper())
            return (np.nan, np.nan), int(sel.sum())
        return W.fit_lnw(np.log(c["W"][sel]), lam_s[sel], pivot), int(sel.sum())

    # Random points and satellites within the consolidated clusters centred on their own seed.
    if cat:
        cons = (cat["MASKFRAC"] < cfg.mask.max_maskfrac) & (cat["LAMBDA"] > minlam * cat["SCALEVAL"])
        centres = set(np.asarray(cat["ID_CENT"][cons, 0]).tolist())
    else:
        centres = set()
    sub = np.isin(g["ID"][cand["GI"]], list(centres))
    sub_cand = {k: v[sub] for k, v in cand.items()}
    rand = rsat = {}
    if sub.any():
        rand, _ = percolate(region, sub_cand, st, q, keepz=True, centering="random", seed=cfg.calib.seed)
        rsat, _ = percolate(region, sub_cand, st, q, keepz=True, centering="randsat",
                            seed=cfg.calib.seed + 1)
    (fg_mean, fg_sigma), n_rand = lnw_fit(rand, "fg")
    (sat_mean, sat_sigma), n_randsat = lnw_fit(rsat, "sat")

    phi1 = W.phi1_model(alpha=cfg.model.alpha, pivot=pivot, rng=np.random.default_rng(cfg.calib.seed))
    params = {"PIVOT": pivot, "LNW_FG_MEAN": fg_mean, "LNW_FG_SIGMA": fg_sigma,
              "LNW_SAT_MEAN": sat_mean, "LNW_SAT_SIGMA": sat_sigma, **phi1}
    info = {"n_cand": int(cand["GI"].size), "n_sub": int(sub.sum()), "n_rand": n_rand,
            "n_randsat": n_randsat}
    central_keys = ("DELTA0", "DELTA1", "SIGMA_M", "LNW_CEN_MEAN", "LNW_CEN_SIGMA")

    def fit_centrals(c, params):
        lam_s, sel = table(c)
        if sel.sum() < 3:
            log.warning("wcen: %d usable training clusters; central terms undefined", sel.sum())
            return {**params, **dict.fromkeys(central_keys, np.nan)}, int(sel.sum())
        pcen, psat = c["P_CEN"][:, 0].astype(np.float64), c["P_SAT"][:, 0].astype(np.float64)
        mstar, cwt, bcounts = _central_terms(region, c)
        d0, d1, sm = W.fit_central_mag(c["REFMAG"][sel], mstar[sel], c["LAMBDA"][sel], lam_s[sel],
                                       pcen[sel], psat[sel], cwt[sel], bcounts[sel], params, pivot)
        cm, cs = W.fit_lnw_cen(np.log(c["W"][sel]), lam_s[sel], pcen[sel], psat[sel],
                               (params["LNW_SAT_MEAN"], params["LNW_SAT_SIGMA"]),
                               (params["LNW_FG_MEAN"], params["LNW_FG_SIGMA"]), pivot)
        return {**params, "DELTA0": d0, "DELTA1": d1, "SIGMA_M": sm, "LNW_CEN_MEAN": cm,
                "LNW_CEN_SIGMA": cs}, int(sel.sum())

    def defined(p):
        return all(np.isfinite(v) for v in p.values())

    params, info["n_train"] = fit_centrals(cat, params)
    log.info("wcen (BCG clusters): %s", ", ".join(f"{k}={v:.3f}" for k, v in params.items()))
    for name, n in (("training", info["n_train"]), ("random-centre", info["n_rand"]),
                    ("random-satellite", info["n_randsat"])):
        if n < MIN_WCEN_SAMPLE:
            log.warning("wcen: only %d %s clusters; the fit is poorly constrained", n, name)
    if not defined(params):
        log.warning("wcen: some parameters are undefined; runs will use BCG centring")
        return params, info
    # As redMaPPer's later iterations: re-seed on the centres of the previous clusters.
    seeds_w = seeds[np.isin(g["ID"][seeds], list(centres))]
    for it in range(1, cfg.calib.wcen_niter):
        # Re-run the clusters with wcen centring and refit the central terms on the mixture.
        reg_w = Region.build(region.gal, region.rs, cfg, table_bands=region.table_bands,
                             footprint=region.footprint, zredcorr=region.zredcorr,
                             bkg=region.model.bkg, zbkg=region.zbkg, wcen_params=params)
        cand_w = calibration_candidates(reg_w, seeds_w, cfg)
        cat_w, _ = calibration_clusters(reg_w, seeds_w, cfg, centering="wcen", cand=cand_w)
        new, n = fit_centrals(cat_w, params)
        if not defined(new):
            log.warning("wcen: iteration %d gave undefined parameters; keeping iteration %d",
                        it + 1, it)
            break
        params, info["n_train"] = new, n
        log.info("wcen (iteration %d, wcen clusters): %s", it + 1,
                 ", ".join(f"{k}={v:.3f}" for k, v in params.items()))
    return params, info


def calibrate_region(gal: dict, cfg: RemaConfig, footprint=None, rs0: RSModel | None = None):
    """Run the calibration on a galaxy table; returns a :class:`Calibration`."""
    t0 = time.time()
    mstar = MStar(cfg.model.mstar)
    zr = cfg.model.zrange
    if rs0 is None:
        rs0 = RSModel.from_template(bands=cfg.survey.bands, ref_band=cfg.survey.ref_band,
                                    template=cfg.model.template, zrange=(zr[0], zr[1] + 0.05),
                                    mstar=cfg.model.mstar)
    tb = tuple(cfg.survey.bands)
    bidx = [tb.index(b) for b in rs0.bands]
    spec = gal["ZSPEC"] > 0
    rs, keep = initial_model(gal["FLUX"][spec][:, bidx], gal["FLUX_IVAR"][spec][:, bidx],
                             gal["REFMAG"][spec], gal["ZSPEC"][spec], rs0, mstar, zrange=zr)
    log.info("initial model from %d spectroscopic red galaxies (%.1fs)", int(keep.sum()), time.time() - t0)
    zredcorr = None
    base = {k: v for k, v in gal.items() if not k.startswith("ZRED")}
    history = []
    for it in range(cfg.calib.niter):
        region = Region.build(base, rs, cfg, table_bands=tb, footprint=footprint, zredcorr=zredcorr)
        seeds = spec_seeds(region, cfg)
        cat, mem = calibration_clusters(region, seeds, cfg)
        if not cat:
            raise RuntimeError("no calibration clusters found")
        good = cat["LAMBDA"] >= cfg.calib.minlambda
        rank_ok = np.flatnonzero(good)
        msel = np.isin(mem["RANK"], rank_ok) & (mem["PMEM"] > 0.01)
        log.info("iteration %d: %d seeds, %d clusters (%d with lambda >= %.0f), %d member weights",
                 it, seeds.size, len(cat["LAMBDA"]), good.sum(), cfg.calib.minlambda, msel.sum())
        mflux = mem["FLUX"][msel][:, bidx]
        mivar = mem["FLUX_IVAR"][msel][:, bidx]
        mz = mem["Z"][msel]
        mw = mem["PMEM"][msel]
        rs, info = fit_redsequence(rs, mflux, mivar, mz, mw, mode=cfg.model.chisq_mode,
                                   eps=cfg.survey.flux_floor, smooth=cfg.calib.smooth_prior)
        rs = update_pivot(rs, mem["REFMAG"][msel], mz, mw)
        # zred correction from spectroscopic members with the new model.
        region2 = Region.build(base, rs, cfg, table_bands=tb, footprint=footprint, zredcorr=None,
                               bkg=region.model.bkg)
        g2 = region2.gal
        sp_mem = np.isin(g2["ID"], mem["ID"][msel & (mem["PMEM"] > 0.5)]) & (g2["ZSPEC"] > 0)
        zredcorr = fit_zred_correction(g2["ZRED_UNCORR"][sp_mem], g2["ZRED_UNCORR_E"][sp_mem],
                                       g2["ZSPEC"][sp_mem], zrange=zr)
        history.append({"iteration": it, "nclusters": int(good.sum()), "loss": float(info["loss"][-1])})
    # Final background and z_lambda correction with the final model.
    region = Region.build(base, rs, cfg, table_bands=tb, footprint=footprint, zredcorr=zredcorr)
    region.zbkg = zred_background(region.gal, cfg, _area_function(region.gal, rs, cfg, footprint, None))
    seeds = spec_seeds(region, cfg)
    cand = calibration_candidates(region, seeds, cfg)
    cat, mem = calibration_clusters(region, seeds, cfg, cand=cand)
    good = cat["LAMBDA"] >= cfg.calib.minlambda
    rc = cfg.richness
    st = Stage.make(rc.percolation.r0, rc.percolation.beta, rc.maxrad_factor)
    q = RadialQuad.make(rmax=rc.percolation.r0 * 20.0 ** rc.percolation.beta + 5 * cfg.model.rsig)
    order = np.argsort(region.gal["ID"])
    gi = order[np.searchsorted(region.gal["ID"], cat["ID_CENT"][good, 0], sorter=order)]
    zl = _run_batched(region, gi, cat["Z_LAMBDA"][good].astype(np.float64), st, q, "zlambda")
    zlcorr = fit_zlambda_correction(zl["Z_LAMBDA"], zl["Z_LAMBDA_E"], cat["Z_LAMBDA"][good],
                                    zl["LAMBDA"], zrange=zr, pivot=cfg.zlambda.pivot)
    # z -> zred_uncorr mapping of LNCGLIKE: the central's zred against the raw z_lambda of these
    # clusters, with redMaPPer's cuts lambda/S > minlambda (calib_zlambda_minlambda
    # is 3 in the DR10 run, as richness.minlambda) and MASKFRAC < max_maskfrac (zlambdacal.py:219-221, 338-343).
    cen = {k: np.asarray(cat[k])[good]
           for k in ("LAMBDA", "SCALEVAL", "MASKFRAC", "ZRED", "ZRED_E")}
    zsel = ((cen["SCALEVAL"] > 0) & (cen["LAMBDA"] > cfg.richness.minlambda * cen["SCALEVAL"])
            & (cen["MASKFRAC"] < cfg.mask.max_maskfrac) & (cen["ZRED_E"] > 0)
            & (zl["Z_LAMBDA"] > 0))
    zru = fit_zred_uncorr(zl["Z_LAMBDA"][zsel], cen["ZRED"][zsel], np.asarray(zlcorr.z))
    if zru is None:
        log.warning("z -> zred_uncorr: %d clusters; LNCGLIKE will use z itself", zsel.sum())
    else:
        import jax.numpy as jnp

        zlcorr = dataclasses.replace(zlcorr, zred_uncorr=jnp.asarray(zru))
        log.info("z -> zred_uncorr from %d clusters; zred_uncorr - z at z = %s: %s", zsel.sum(),
                 " ".join(f"{z:.2f}" for z in np.asarray(zlcorr.z)),
                 " ".join(f"{d:+.4f}" for d in zru - np.asarray(zlcorr.z, np.float64)))
    dz = (cat["Z_LAMBDA"][good] - zl["Z_LAMBDA"]) / (1 + cat["Z_LAMBDA"][good])
    ok = np.isfinite(dz) & (zl["Z_LAMBDA"] > 0)
    nmad = 1.4826 * np.median(np.abs(dz[ok] - np.median(dz[ok]))) if ok.any() else np.nan
    log.info("calibration done in %.1fs: %d clusters, z_lambda vs z_spec bias %.4f NMAD %.4f",
             time.time() - t0, good.sum(), float(np.median(dz[ok])) if ok.any() else np.nan, nmad)
    meta = {"NCLUSTER": int(good.sum()), "NSPECGAL": int(spec.sum()), "ZLNMAD": float(nmad),
            "NZRMOD": int(zsel.sum()) if zru is not None else 0}
    info = {"history": history, "clusters": cat, "members": mem, "zl": zl, "good": good,
            "zrmod_sel": zsel, "band_idx": bidx}
    wcen = None
    if cfg.calib.wcen:
        tw = time.time()
        try:
            wcen, info["wcen"] = calibrate_wcen(region, seeds, cfg, cand=cand, cat=cat)
        except Exception:      # keep the rest of the calibration
            log.exception("wcen calibration failed; the calibration has no wcen model")
            wcen, info["wcen"] = None, {"n_train": 0}
        meta["NWCEN"] = info["wcen"]["n_train"]
        log.info("wcen calibration in %.1fs (%s)", time.time() - tw, info["wcen"])
    return Calibration(rs=rs, zredcorr=zredcorr, bkg=region.model.bkg, zlcorr=zlcorr, wcen=wcen,
                       config=cfg, meta=meta, zbkg=region.zbkg), info


def calibrate(args):
    """CLI entry point."""
    from ..sky.maps import Footprint

    from ..io.legacy import read_galaxies
    from ..sky.regions import Box, sky_header, sky_union

    from ..config import apply_overrides

    cfg = RemaConfig.from_yaml(args.config) if args.config else RemaConfig()
    cfg = cfg.replace(cosmology=parse_cosmology_overrides(getattr(args, "cosmology", None)))
    cfg = apply_overrides(cfg, getattr(args, "set", None))
    if cfg.model.filter != "redsequence":
        raise SystemExit(f"rema calibrate fits the red sequence: model.filter must be redsequence, "
                         f"not {cfg.model.filter}")
    boxes = getattr(args, "box", None)
    sky = sky_union([Box(*map(float, b)) for b in boxes]) if boxes else None
    gal = read_galaxies(args.galaxies, sky, cfg)
    fp = Footprint.read(args.footprint) if args.footprint else None
    rs0 = None
    if args.init_pars:
        log.info("initial colours from %s are not used for band sets other than its own", args.init_pars)
    cal, info = calibrate_region(gal, cfg, fp, rs0)
    cal.meta = {**(cal.meta or {}), **sky_header(sky)}
    cal.write(args.out)
    log.info("wrote %s", args.out)
    if getattr(args, "plots", None):
        write_plots(cal, info, args.plots)


def write_plots(cal: Calibration, info: dict, outdir) -> None:
    """Diagnostic plots of the red sequence and of z_lambda vs z_spec."""
    from pathlib import Path

    from .plots import plot_redsequence, plot_zlambda

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    mem, cat, good, bidx = info["members"], info["clusters"], info["good"], info["band_idx"]
    sel = np.isin(mem["RANK"], np.flatnonzero(good)) & (mem["PMEM"] > 0.3)
    plot_redsequence(cal.rs, mem["FLUX"][sel][:, bidx], mem["FLUX_IVAR"][sel][:, bidx], mem["Z"][sel],
                     mem["PMEM"][sel], mem["REFMAG"][sel], out / "redsequence.png")
    plot_zlambda(cat["Z_LAMBDA"][good], info["zl"]["Z_LAMBDA"], info["zl"]["Z_LAMBDA_E"], out / "zlambda.png")
    log.info("plots in %s", out)
