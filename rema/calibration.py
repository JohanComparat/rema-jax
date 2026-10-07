"""Calibration products, read from and written to one FITS file.

A :class:`Calibration` holds everything a run needs besides the galaxies and the footprint:

- the red-sequence model (HDUs RS_MEAN, RS_SLOPE, RS_LOG_SIGMA, RS_CORR, RS_PIVOT);
- the zred correction (ZREDCORR);
- the chi^2 background (BKG_CHISQ image + BKG_Z, BKG_C, BKG_M axes);
- the zred background of the wcen centring model (BKG_ZRED image + BKG_ZRED_Z, BKG_ZRED_M);
- the z_lambda correction (ZLAMBDACORR, with a ZRED_UNCORR column for the z -> zred_uncorr
  mapping of LNCGLIKE when fitted) and the wcen centring parameters (WCEN, upper-case keys
  DELTA0, DELTA1, SIGMA_M, PIVOT, LNW_{CEN,SAT,FG}_{MEAN,SIGMA}, PHI1_*), when fitted;
- provenance: the configuration (CONFIG HDU, YAML text) and the PRIMARY header (rema version,
  chi^2 definition, bands, inputs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits

from . import __version__
from .config import RemaConfig
from .io.tables import table_hdu, write_fits
from .model.background import ChisqBkg, ZredBkg
from .model.redsequence import RSModel, ZredCorrection


# Grid of the z -> zred_uncorr mapping, as redMaPPer's ZlambdaCorrectionPar: every
# zlambda_binsize from 0.02 below the first node to 0.07 above the last (zlambdacal.py:347-350).
ZRMOD_BINSIZE = 0.002
ZRMOD_PAD = (0.02, 0.07)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ZlambdaCorrection:
    """z_lambda += offset(z) + slope(z) ln(lambda / pivot); z_lambda_e^2 += scatter(z)^2.

    ``zred_uncorr`` holds redMaPPer's z -> zred_uncorr mapping at the nodes ``z``: the median
    zred of the central galaxies of clusters at raw z_lambda = z, which centres the zred term of
    LNCGLIKE (:meth:`zrmod_table`). None when not fitted (files written before it): LNCGLIKE then
    uses z itself.
    """

    z: jnp.ndarray
    offset: jnp.ndarray
    slope: jnp.ndarray
    scatter: jnp.ndarray
    zred_uncorr: jnp.ndarray | None = None
    pivot: float = field(default=30.0, metadata=dict(static=True))

    def apply(self, zl, zl_e, lam):
        lam = jnp.maximum(lam, 1.0)
        dz = jnp.interp(zl, self.z, self.offset) + jnp.interp(zl, self.z, self.slope) * jnp.log(lam / self.pivot)
        sc = jnp.interp(zl, self.z, self.scatter)
        return zl + dz, jnp.sqrt(zl_e**2 + sc**2)

    def zrmod_table(self, binsize: float = ZRMOD_BINSIZE):
        """(z, zred) table of the z -> zred_uncorr mapping (float64 numpy), or None without one.

        As redMaPPer's ZlambdaCorrectionPar (zlambda.py:621-622, 662-663): the natural cubic
        spline through (z, zred_uncorr), sampled every ``binsize`` from 0.02 below the first node
        to 0.07 above the last. LNCGLIKE interpolates the table linearly, extrapolating its end
        segments (redMaPPer's ``interpol``, run_likelihoods.py:213-214).
        """
        if self.zred_uncorr is None:
            return None
        from scipy.interpolate import CubicSpline

        nodes = np.asarray(self.z, np.float64)
        lo, hi = nodes[0] - ZRMOD_PAD[0], nodes[-1] + ZRMOD_PAD[1]
        zz = binsize * np.arange(int(np.round((hi - lo) / binsize))) + lo
        spl = CubicSpline(nodes, np.asarray(self.zred_uncorr, np.float64), bc_type="natural")
        return zz, spl(zz)


@dataclass
class Calibration:
    rs: RSModel
    zredcorr: ZredCorrection | None = None
    bkg: ChisqBkg | None = None
    zlcorr: ZlambdaCorrection | None = None
    wcen: dict | None = None
    config: RemaConfig | None = None
    meta: dict = field(default_factory=dict)
    zbkg: ZredBkg | None = None

    def write(self, path: str | Path) -> Path:
        hdus = list(self.rs.to_hdus())
        if self.zredcorr is not None:
            zc = self.zredcorr
            hdus.append(table_hdu({"Z": np.asarray(zc.z_corr), "CORR": np.asarray(zc.corr)}, extname="ZREDCORR"))
            hdus.append(table_hdu({"Z": np.asarray(zc.z_slope), "CORR_SLOPE": np.asarray(zc.corr_slope),
                                   "CORR_R": np.asarray(zc.corr_r)}, extname="ZREDCORR_SLOPE"))
        if self.bkg is not None:
            hdus += self.bkg.to_hdus()
        if self.zbkg is not None:
            hdus += self.zbkg.to_hdus()
        if self.zlcorr is not None:
            c = self.zlcorr
            cols = {"Z": np.asarray(c.z), "OFFSET": np.asarray(c.offset),
                    "SLOPE": np.asarray(c.slope), "SCATTER": np.asarray(c.scatter)}
            if c.zred_uncorr is not None:
                cols["ZRED_UNCORR"] = np.asarray(c.zred_uncorr)
            hdus.append(table_hdu(cols, {"PIVOT": c.pivot}, "ZLAMBDACORR"))
        if self.wcen:
            hdus.append(table_hdu({k.upper(): np.atleast_1d(np.asarray(v, np.float64))
                                   for k, v in self.wcen.items()}, extname="WCEN"))
        if self.config is not None:
            text = self.config.to_yaml()
            hdus.append(table_hdu({"YAML": np.array([text.encode()])}, extname="CONFIG"))
        prim = {"REMAVER": __version__, "BANDS": ",".join(self.rs.bands), "REFBAND": self.rs.ref_band}
        if self.config is not None:
            prim["CHI2MODE"] = self.config.model.chisq_mode
            prim["FLXFLOOR"] = self.config.survey.flux_floor
            prim["MSTAR"] = self.config.model.mstar
        for k, v in self.meta.items():
            prim[k[:8].upper()] = v
        if self.config is not None:
            prim.update(self.config.cosmology.header())
        return write_fits(path, hdus, prim)

    @classmethod
    def read(cls, path: str | Path) -> "Calibration":
        rs = RSModel.from_fits(path)
        with fits.open(path) as h:
            names = [x.name for x in h]
            zc = None
            if "ZREDCORR" in names:
                a, b = h["ZREDCORR"].data, h["ZREDCORR_SLOPE"].data
                zc = ZredCorrection(jnp.asarray(np.asarray(a["CORR"], np.float64)),
                                    jnp.asarray(np.asarray(b["CORR_SLOPE"], np.float64)),
                                    jnp.asarray(np.asarray(b["CORR_R"], np.float64)),
                                    jnp.asarray(np.asarray(a["Z"], np.float64)),
                                    jnp.asarray(np.asarray(b["Z"], np.float64)))
            zl = None
            if "ZLAMBDACORR" in names:
                d = h["ZLAMBDACORR"]
                col = lambda c: jnp.asarray(np.asarray(d.data[c], np.float64))
                # Files without ZRED_UNCORR (no mapping): LNCGLIKE uses z itself.
                zru = col("ZRED_UNCORR") if "ZRED_UNCORR" in d.columns.names else None
                zl = ZlambdaCorrection(*(col(c) for c in ("Z", "OFFSET", "SLOPE", "SCATTER")),
                                       zred_uncorr=zru, pivot=float(d.header.get("PIVOT", 30.0)))
            wcen = None
            if "WCEN" in names:
                d = h["WCEN"].data
                wcen = {c.upper(): float(d[c][0]) for c in d.columns.names}
            cfg = None
            if "CONFIG" in names:
                import yaml

                text = h["CONFIG"].data["YAML"][0]
                text = text.decode() if isinstance(text, bytes) else str(text)
                cfg = RemaConfig.from_dict(yaml.safe_load(text))
            meta = {k: h[0].header[k] for k in h[0].header if k not in ("SIMPLE", "BITPIX", "NAXIS", "EXTEND")}
        bkg = ChisqBkg.from_fits(path) if "BKG_CHISQ" in names else None
        zbkg = ZredBkg.from_fits(path) if "BKG_ZRED" in names else None
        return cls(rs=rs, zredcorr=zc, bkg=bkg, zlcorr=zl, wcen=wcen, config=cfg, meta=meta,
                   zbkg=zbkg)

    @classmethod
    def from_redmapper_pars(cls, path: str | Path, config: RemaConfig | None = None) -> "Calibration":
        """Red-sequence model and zred correction of a redMaPPer ``*_pars.fit`` file."""
        return cls(rs=RSModel.from_redmapper_pars(str(path)),
                   zredcorr=ZredCorrection.from_redmapper_pars(str(path)), config=config,
                   meta={"PARSFILE": Path(path).name[:68]})
