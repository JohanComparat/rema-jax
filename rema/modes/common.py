"""Shared run context: galaxies, neighbour search, model, footprint.

A :class:`Region` holds the galaxy table of a sky region (from ``rema ingest``), its zred, a
neighbour index, the filter model (red sequence + background + cosmology + m*), the wcen
centring model when calibrated, and, when available, the footprint built from the DR11 randoms.
Modes ask it for padded neighbour batches (:class:`rema.core.richness.Neighbors`) and for
aperture completeness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from ..config import RemaConfig
from ..core.context import FilterModel
from ..core.maskcorr import no_footprint, radial_completeness
from ..core.richness import Neighbors, RadialQuad
from ..core.zred import ZredGrid, compute_zred
from ..model.background import ChisqBkg, ZredBkg, build_chisq_bkg, build_zred_bkg
from ..model.cosmo import CosmoTable
from ..model.profiles import MStar
from ..model.redsequence import RSModel, ZredCorrection
from ..sky.maps import Footprint
from ..sky.neighbors import NeighborIndex, Padded

log = logging.getLogger(__name__)


@dataclass
class Region:
    gal: dict
    table_bands: tuple
    rs: RSModel
    model: FilterModel
    cfg: RemaConfig
    footprint: Footprint | None = None
    zredcorr: ZredCorrection | None = None
    band_idx: list = field(default_factory=list)
    zlcorr: object = None                 # rema.calibration.ZlambdaCorrection, applied to outputs
    zbkg: ZredBkg | None = None           # zred background of the wcen centring model
    wcen: object = None                   # rema.core.centering.WcenModel when calibrated
    wcen_params: dict | None = None
    _index: NeighborIndex | None = None
    _sub: dict = field(default_factory=dict)
    _fmap: object = None
    _ones: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, gal: dict, rs: RSModel, cfg: RemaConfig | None = None, *,
              table_bands=None, footprint: Footprint | None = None,
              zredcorr: ZredCorrection | None = None, bkg: ChisqBkg | None = None,
              area_deg2: float | None = None, zlcorr=None, zbkg: ZredBkg | None = None,
              wcen_params: dict | None = None) -> "Region":
        """Region of a galaxy table with the given model.

        Missing zred columns and backgrounds are computed from the table. The wcen centring
        model is set up when ``wcen_params`` are calibrated (its zred background built from the
        table when ``zbkg`` is not given), with the z -> zred_uncorr mapping of ``zlcorr`` for
        LNCGLIKE when it has one (zrmod(z) = z otherwise).
        """
        cfg = cfg or RemaConfig()
        table_bands = tuple(table_bands or cfg.survey.bands)
        band_idx = [table_bands.index(b) for b in rs.bands]
        cosmo = CosmoTable.create(cfg.cosmology.Omega_m, cfg.cosmology.h)
        mstar = MStar(cfg.model.mstar)
        gal = dict(gal)
        flux = np.asarray(gal["FLUX"])[:, band_idx]
        ivar = np.asarray(gal["FLUX_IVAR"])[:, band_idx]
        if "ZRED" not in gal:
            grid = ZredGrid.create(rs, mstar, cosmo, cfg.model.zrange, cfg.model.zbin_coarse)
            res = compute_zred(flux, ivar, grid, rs.iref, mode=cfg.model.chisq_mode,
                               eps=cfg.survey.flux_floor, alpha=cfg.model.alpha,
                               use_lndet=cfg.zred.use_lndet, correction=zredcorr, rs=rs)
            gal.update(res.as_columns())
        area_fn = _area_function(gal, rs, cfg, footprint, area_deg2)
        if bkg is None:
            bkg = build_chisq_bkg(flux, ivar, gal["REFMAG"], rs, mstar, area_fn,
                                  zrange=cfg.model.zrange, iref=rs.iref,
                                  zbinsize=cfg.background.zbinsize,
                                  chisqbinsize=cfg.background.chisqbinsize,
                                  chisq_max=cfg.model.chisq_max,
                                  magbinsize=cfg.background.refmagbinsize,
                                  mag_max=float(np.max(gal["REFMAG"])) + 0.2,
                                  lmax_faint=cfg.background.lmax_faint,
                                  min_counts=cfg.background.min_counts,
                                  smooth_z=cfg.background.smooth_z,
                                  mode=cfg.model.chisq_mode, eps=cfg.survey.flux_floor)
        model = FilterModel.create(rs, bkg, cfg, cosmo=cosmo, mstar=mstar)
        wcen = None
        if wcen_params:
            from ..core.centering import WcenModel, wcen_calibrated

            if wcen_calibrated(wcen_params):
                if zbkg is None:
                    zbkg = zred_background(gal, cfg, area_fn)
                zrmod = zlcorr.zrmod_table() if zlcorr is not None else None
                wcen = WcenModel.create(wcen_params, zbkg, cfg, zrmod=zrmod)
        return cls(gal=gal, table_bands=table_bands, rs=rs, model=model, cfg=cfg,
                   footprint=footprint, zredcorr=zredcorr, band_idx=band_idx, zlcorr=zlcorr,
                   zbkg=zbkg, wcen=wcen, wcen_params=wcen_params)

    def centering_method(self, method: str | None = None) -> str:
        """"bcg" or "wcen": ``method`` or cfg.centering.method; "auto" is wcen when calibrated."""
        method = method or self.cfg.centering.method
        if method == "auto":
            return "wcen" if self.wcen is not None else "bcg"
        if method == "wcen" and self.wcen is None:
            raise ValueError("wcen centring requested but the calibration has no wcen model")
        if method not in ("bcg", "wcen"):
            raise ValueError(f"unknown centring method {method!r}")
        return method

    # ------------------------------------------------------------------ neighbours
    @property
    def index(self) -> NeighborIndex:
        if self._index is None:
            self._index = NeighborIndex(self.gal["RA"], self.gal["DEC"])
        return self._index

    def _subindex(self, mag_max: float, max_cached: int = 8):
        """Index of galaxies brighter than mag_max (rounded up to 0.5 mag).

        At most ``max_cached`` indices are kept (least recently used dropped first): each costs
        ~40 bytes per galaxy.
        """
        key = float(np.ceil(mag_max * 2) / 2)
        if key in self._sub:
            val = self._sub.pop(key)
        else:
            sel = np.flatnonzero(self.gal["REFMAG"] < key)
            val = (NeighborIndex(self.gal["RA"][sel], self.gal["DEC"][sel]), sel)
            while len(self._sub) >= max_cached:
                self._sub.pop(next(iter(self._sub)))
        self._sub[key] = val
        return val

    def query(self, ra_c, dec_c, radius_deg, mag_max: float | None = None, kmin: int = 128) -> Padded:
        """Neighbours within ``radius_deg`` (and brighter than ``mag_max``); indices into ``gal``."""
        if mag_max is None:
            return self.index.query(ra_c, dec_c, radius_deg, kmin=kmin)
        sub, sel = self._subindex(mag_max)
        pad = sub.query(ra_c, dec_c, radius_deg, kmin=kmin)
        pad.idx = np.where(pad.valid, sel[pad.idx], 0)
        return pad

    def neighbors(self, pad: Padded, pfree: np.ndarray | None = None,
                  center_ids: np.ndarray | None = None, pfree_nb: np.ndarray | None = None) -> Neighbors:
        """Device arrays for padded neighbour lists (model bands only).

        ``pfree`` is per galaxy of the table, ``pfree_nb`` per neighbour ([B, K]); default 1.
        """
        g, i = self.gal, pad.idx
        bidx = self.band_idx
        is_center = np.zeros(i.shape, bool)
        if center_ids is not None:
            is_center = pad.valid & (g["ID"][i] == np.asarray(center_ids)[:, None])
        if pfree_nb is not None:
            pf = np.asarray(pfree_nb, np.float32)
        elif pfree is not None:
            pf = pfree[i].astype(np.float32)
        else:
            pf = np.ones(i.shape, np.float32)
        return Neighbors(theta=jnp.asarray(pad.theta, jnp.float32),
                         refmag=jnp.asarray(g["REFMAG"][i]), refmag_err=jnp.asarray(g["REFMAG_ERR"][i]),
                         flux=jnp.asarray(g["FLUX"][i][..., bidx]),
                         ivar=jnp.asarray(g["FLUX_IVAR"][i][..., bidx]),
                         zred=jnp.asarray(g["ZRED"][i]), zred_e=jnp.asarray(g["ZRED_E"][i]),
                         pfree=jnp.asarray(pf), valid=jnp.asarray(pad.valid),
                         is_center=jnp.asarray(is_center))

    # ------------------------------------------------------------------ completeness
    def completeness(self, ra, dec, z, quad: RadialQuad):
        """(frad, fgeo) [B, n_r]; all ones without a footprint."""
        n = np.size(ra)
        if self.footprint is None:
            key = (n, quad.r.shape[0])
            if key not in self._ones:
                self._ones[key] = no_footprint(n, quad)
            return self._ones[key]
        if self._fmap is None:
            ref = self.rs.ref_band.upper()
            self._fmap = self.footprint.fine.device(("FRACGOOD", f"SIGF_{ref}"))
        mag_max = float(self.cfg.survey.mag_max) if self.cfg.survey.mag_max else 99.0
        return radial_completeness(jnp.asarray(ra, jnp.float32), jnp.asarray(dec, jnp.float32),
                                   jnp.asarray(z, jnp.float32), self._fmap, quad, self.model,
                                   ref_band=self.rs.ref_band, snr_min=self.cfg.survey.ref_snr_min,
                                   mag_max=mag_max, n_phi=self.cfg.mask.n_phi,
                                   n_mag=self.cfg.mask.n_mag)

    def correct_zlambda(self, z, z_e, lam):
        """Calibrated z_lambda correction (identity without one); invalid entries (z <= 0) kept."""
        z, z_e, lam = (np.asarray(a, np.float64) for a in (z, z_e, lam))
        if self.zlcorr is None:
            return z, z_e
        zc, ec = self.zlcorr.apply(jnp.asarray(z), jnp.asarray(z_e), jnp.asarray(lam))
        good = z > 0
        return np.where(good, np.asarray(zc), z), np.where(good, np.asarray(ec), z_e)

    # ------------------------------------------------------------------ host helpers (numpy)
    # Host bookkeeping must not call jnp on Python scalars: every such eager op takes JAX's
    # slow dispatch path (~0.3 ms), which dominated the percolation loop.
    def _host_tables(self):
        if not hasattr(self, "_ht"):
            m = self.model
            self._ht = (np.asarray(m.cosmo.z, np.float64), np.asarray(m.cosmo.da_tab, np.float64),
                        np.asarray(m.mstar_z, np.float64), np.asarray(m.mstar_m, np.float64))
        return self._ht

    def mpc_per_deg_np(self, z):
        zt, da, _, _ = self._host_tables()
        return np.interp(np.asarray(z, np.float64), zt, da) * np.pi / 180.0

    def mstar_np(self, z):
        _, _, zm, mm = self._host_tables()
        return np.interp(np.asarray(z, np.float64), zm, mm)

    def radius_deg(self, rmax_mpc, z):
        """Angular radius of ``rmax_mpc`` h^-1 Mpc at redshift ``z``."""
        return np.asarray(rmax_mpc) / self.mpc_per_deg_np(z)

    def maxmag(self, z):
        return self.mstar_np(z) - 2.5 * np.log10(self.model.lval)


def _area_function(gal, rs, cfg, footprint, area_deg2):
    """Effective area (deg^2) of the catalogue as a function of reference magnitude."""
    if footprint is not None:
        def area_fn(m):
            return float(footprint.effective_area(m, rs.ref_band, cfg.survey.ref_snr_min)[0])
        return area_fn
    area = area_deg2 if area_deg2 is not None else _area_from_positions(gal["RA"], gal["DEC"])

    def area_fn(m):
        return area
    return area_fn


def zred_background(gal: dict, cfg: RemaConfig, area_fn) -> ZredBkg:
    """Zred background of a galaxy table (wcen centring), over the zred grid's range."""
    zr = cfg.model.zrange
    return build_zred_bkg(gal["ZRED"], gal["ZRED_CHISQ"], gal["REFMAG"], area_fn,
                          zred_range=(max(0.01, zr[0] - 0.05), zr[1] + 0.1),
                          zredbinsize=cfg.background.zredbinsize,
                          magbinsize=cfg.background.refmagbinsize,
                          mag_max=float(np.max(gal["REFMAG"])) + 0.2,
                          chisq_max=cfg.centering.wcen_zred_chisq_max)


def _area_from_positions(ra, dec, nside: int = 1024) -> float:
    """Rough area (deg^2) of the occupied HEALPix pixels, for runs without a footprint."""
    import healpy as hp

    from ..sky.healpix import pix_area_deg2

    pix = np.unique(hp.ang2pix(nside, ra, dec, nest=True, lonlat=True))
    return float(pix.size * pix_area_deg2(nside))
