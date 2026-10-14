"""The z_vlim map: the redshift up to which the catalogue is volume limited, per pixel.

As redMaPPer's VolumeLimitMask (zmax from the depth map) and Kluge et al. (2024, Fig. B.3): in
each pixel of the footprint, z_vlim is the redshift at which a galaxy of ``lstar_frac`` L*
(m*(z) - 2.5 log10(lstar_frac) in the reference band) reaches the ``nsigma`` depth,

    depth = zeropoint - 2.5 log10(nsigma sigma_f),

with sigma_f the pixel's median dereddened 1-sigma flux error of a galaxy (SIGF_<band> of the
footprint). With the default 0.2 L* at 5 sigma, the members that make lambda (brighter than
0.2 L*, S/N >= 5 in z) are all detectable below z_vlim, so lambda does not depend on depth there.

The area of a pixel is the number of good random points in it that lie in the run's domain,
divided by their density. Without the randoms, it is FRACGOOD times the pixel area times the
fraction of the pixel inside the domain, which overcounts pixels cut by the edge of the survey
(FRACGOOD is the good fraction of the random points present).
"""

from __future__ import annotations

import numpy as np

from ..abundance.area import ZvlimMap, zvlim_of
from ..model.profiles import MStar
from ..sky.healpix import pix_area_deg2
from ..sky.maps import Footprint
from .config import VlimConfig
from .domain import RunDomain


def depth_mag(sigf, nsigma: float, zeropoint: float = 22.5) -> np.ndarray:
    """Magnitude of an ``nsigma`` detection for a 1-sigma flux error ``sigf`` (NaN without depth).

    >>> float(np.round(depth_mag(0.1, 5.0), 3))
    23.253
    """
    sigf = np.asarray(sigf, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sigf > 0, zeropoint - 2.5 * np.log10(nsigma * sigf), np.nan)


def domain_fraction(domain: RunDomain, nside: int, pix: np.ndarray, sub: int = 3,
                    chunk: int = 2_000_000) -> np.ndarray:
    """Fraction of each NESTED pixel inside ``domain``: 0 or 1 when its centre and corners agree,
    else the fraction of its 4**``sub`` sub-pixel centres inside."""
    import healpy as hp

    pix = np.asarray(pix, np.int64)
    out = np.empty(pix.size, np.float64)
    for s0 in range(0, pix.size, chunk):
        p = pix[s0:s0 + chunk]
        ra, dec = hp.pix2ang(nside, p, nest=True, lonlat=True)
        c = domain.contains(ra, dec).astype(np.float64)
        vec = hp.boundaries(nside, p, step=1, nest=True)              # [n, 3, 4]
        cra, cdec = hp.vec2ang(np.moveaxis(vec, 1, 2).reshape(-1, 3), lonlat=True)
        corners = domain.contains(cra, cdec).reshape(p.size, 4)
        edge = np.flatnonzero(np.any(corners != (c[:, None] > 0), axis=1))
        if edge.size:
            k = 4**sub
            child = (p[edge][:, None] * k + np.arange(k)[None, :]).ravel()
            sra, sdec = hp.pix2ang(nside * 2**sub, child, nest=True, lonlat=True)
            c[edge] = domain.contains(sra, sdec).reshape(edge.size, k).mean(axis=1)
        out[s0:s0 + chunk] = c
    return out


def count_randoms(chunks, cfg, nside: int, domain: RunDomain | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """(pixels, counts, number read) of the random points of ``chunks`` that pass the galaxies'
    cuts and lie in ``domain``, per NESTED pixel at ``nside``."""
    from ..sky.healpix import ang2pix_nest_np
    from ..sky.maps import good_randoms

    acc, nread = {}, 0
    for r in chunks:
        nread += r["RA"].size
        ok = good_randoms(r, cfg)
        ra, dec = np.asarray(r["RA"], np.float64)[ok], np.asarray(r["DEC"], np.float64)[ok]
        if domain is not None:
            k = domain.contains(ra, dec)
            ra, dec = ra[k], dec[k]
        u, c = np.unique(ang2pix_nest_np(nside, ra, dec), return_counts=True)
        for a, b in zip(u.tolist(), c.tolist()):
            acc[a] = acc.get(a, 0) + b
    pix = np.fromiter(acc.keys(), np.int64, len(acc))
    cnt = np.fromiter(acc.values(), np.int64, len(acc))
    o = np.argsort(pix)
    return pix[o], cnt[o], nread


def zvlim_map_from_footprint(fp: Footprint, cfg, domain: RunDomain | None = None,
                             vcfg: VlimConfig | None = None, randoms=None, density: float | None = None,
                             sub: int = 3) -> ZvlimMap:
    """z_vlim map of the footprint ``fp`` (pixels with area > 0), with the catalogue
    configuration ``cfg`` (m*(z) model, reference band, zeropoint).

    ``randoms``: chunks of DR11 random points (:func:`rema.randoms.generate.iter_survey_randoms`)
    of total ``density`` per deg2, for the area (see the module docstring).

    Columns: AREA [deg2], ZVLIM, FRACGOOD, DOMFRAC (or NRAND with the randoms) and DEPTH_<band>
    (the ``nsigma`` depth in every band of the footprint). Header: LSTAR, NSIGMA, BAND, MSTAR,
    FOOTDIG, DENSITY, DOMDIG, GLATMIN.
    """
    vcfg = vcfg or VlimConfig()
    band = (vcfg.band or cfg.survey.ref_band).upper()
    if fp.nside != vcfg.nside:
        raise ValueError(f"footprint NSIDE {fp.nside} differs from vlim.nside {vcfg.nside}")
    good_all = np.asarray(fp.fine.values["FRACGOOD"], np.float64)
    if randoms is not None:
        if not density:
            raise ValueError("the area from randoms needs their density")
        pix, cnt, _ = count_randoms(randoms, cfg, fp.nside, domain)
        i, found = fp.fine.index_np(pix)
        pix, cnt, i = pix[found], cnt[found], i[found]
        area = cnt / float(density)
        extra = {"NRAND": cnt.astype(np.int32), "FRACGOOD": good_all[i].astype(np.float32)}
    else:
        keep = good_all > 0
        i = np.flatnonzero(keep)
        pix = fp.fine.pixels[i]
        dom = domain_fraction(domain, fp.nside, pix, sub) if domain is not None else np.ones(pix.size)
        area = good_all[i] * dom * pix_area_deg2(fp.nside)
        sel = area > 0
        pix, area, i, dom = pix[sel], area[sel], i[sel], dom[sel]
        extra = {"FRACGOOD": good_all[i].astype(np.float32), "DOMFRAC": dom.astype(np.float32)}
    depth = {b.upper(): depth_mag(np.asarray(fp.fine.values[f"SIGF_{b.upper()}"])[i], vcfg.nsigma,
                                  cfg.survey.zeropoint) for b in fp.bands}
    zv = zvlim_of(depth[band], MStar(cfg.model.mstar), lval=vcfg.lstar_frac)
    extra.update({f"DEPTH_{b}": d.astype(np.float32) for b, d in depth.items()})
    meta = {"LSTAR": vcfg.lstar_frac, "NSIGMA": vcfg.nsigma, "BAND": band, "MSTAR": cfg.model.mstar,
            "FOOTDIG": fp.digest()[:12], "DENSITY": float(density) if randoms is not None else 0.0}
    if domain is not None:
        meta.update(DOMDIG=domain.digest(), GLATMIN=domain.glat_min)
    return ZvlimMap(fp.nside, pix, area, zv, extra, meta)


def annotate(ra, dec, zmap: ZvlimMap, domain: RunDomain | None = None) -> dict:
    """ZVLIM (NaN outside the map) and IN_DOMAIN of positions (clusters or random points)."""
    out = {"ZVLIM": zmap.value_at(ra, dec)}
    out["IN_DOMAIN"] = (domain.contains(ra, dec) if domain is not None
                        else np.isfinite(out["ZVLIM"]))
    return out
