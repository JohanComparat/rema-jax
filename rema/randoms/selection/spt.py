"""SPT-SZ 2500d (Bocquet et al. 2019) and SPT-ECS: selection by xi in fields of different depth.

The detection significance xi of a cluster relates to its unbiased significance zeta by
P(xi | zeta) = N(sqrt(zeta^2 + 3), 1), and zeta scales with the depth of the field, gamma_f
(Bocquet et al. 2019, Eqs. 6-7). A cluster of measured xi_i in field f_i, moved to field f_j,
has zeta_j = sqrt(xi_i^2 - 3) gamma_{f_j} / gamma_{f_i} and is detected with probability

    P = Phi(sqrt(zeta_j^2 + 3) - xi_cut).

The fields are given as a table (FITS or CSV, ``#`` comments allowed) with FIELD and GAMMA and
either boxes (RA0, RA1, DEC0, DEC1; RA0 < RA1 modulo 360) or centres (RA, DEC). With centres,
a position belongs to the nearest field centre inside ``region`` (the SPT-SZ survey, R.A. 20h-7h,
Dec -65 to -40 deg, Bleem et al. 2015), an approximation of the field boundaries, which are not
published; ``scripts/randoms/spt_sz_2500d_fields.csv`` holds the 19 SPT-SZ fields.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .base import ExternalSelection, register


SPT_SZ_REGION = (300.0, 465.0, -65.0, -40.0)        # RA0, RA1 (modulo 360), DEC0, DEC1


def read_fields(path) -> dict:
    from astropy.table import Table

    if str(path).endswith((".csv", ".txt")):
        t = Table.read(path, format="ascii.csv", comment="#")
    else:
        t = Table.read(path)
    return {c.upper(): np.asarray(t[c]) for c in t.colnames}


@register("spt")
class SPTSelection(ExternalSelection):
    columns = ("EXT_XI", "EXT_FIELD")

    def __init__(self, fields: str | dict | None = None, xi_cut: float = 5.0, region=SPT_SZ_REGION):
        self.fields_path = fields if isinstance(fields, str) else None
        f = read_fields(fields) if isinstance(fields, str) else fields
        self.name_f = np.asarray([str(x).strip() for x in f["FIELD"]])
        self.gamma = np.asarray(f["GAMMA"], np.float64)
        self.box = (np.stack([np.asarray(f[k], np.float64) for k in ("RA0", "RA1", "DEC0", "DEC1")], axis=1)
                    if "RA0" in f else None)
        self.centre = None if self.box is not None else np.stack([np.asarray(f["RA"], np.float64),
                                                                   np.asarray(f["DEC"], np.float64)], axis=1)
        self.region = tuple(float(x) for x in region)
        self.xi_cut = float(xi_cut)

    def field_of(self, ra, dec) -> np.ndarray:
        ra, dec = np.asarray(ra, np.float64) % 360.0, np.asarray(dec, np.float64)
        out = np.full(ra.shape, -1)
        if self.box is not None:
            for i, (r0, r1, d0, d1) in enumerate(self.box):
                inside = (((ra - r0) % 360.0) < (r1 - r0)) & (dec >= d0) & (dec < d1)
                out[inside & (out < 0)] = i
            return out
        r0, r1, d0, d1 = self.region
        inside = (((ra - r0) % 360.0) < (r1 - r0)) & (dec >= d0) & (dec < d1)
        if inside.any():
            from scipy.spatial import cKDTree

            def xyz(a, d):
                a, d = np.radians(a), np.radians(d)
                return np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], axis=-1)

            _, k = cKDTree(xyz(self.centre[:, 0], self.centre[:, 1])).query(xyz(ra[inside], dec[inside]))
            out[inside] = k
        return out

    def in_footprint(self, ra, dec) -> np.ndarray:
        return self.field_of(ra, dec) >= 0

    def own_cut(self, cat: dict) -> np.ndarray:
        return np.asarray(cat["EXT_XI"], np.float64) >= self.xi_cut

    def gamma_of_name(self, names) -> np.ndarray:
        idx = {n: i for i, n in enumerate(self.name_f)}
        return np.array([self.gamma[idx[str(n).strip()]] if str(n).strip() in idx else np.nan for n in names])

    def p_detect(self, src: dict, ra, dec) -> np.ndarray:
        fj = self.field_of(ra, dec)
        xi = np.asarray(src["EXT_XI"], np.float64)
        g_i = self.gamma_of_name(src["EXT_FIELD"])
        zeta_i = np.sqrt(np.maximum(xi**2 - 3.0, 0.0))
        g_j = np.where(fj >= 0, self.gamma[np.maximum(fj, 0)], np.nan)
        zeta_j = zeta_i * g_j / g_i
        p = ndtr(np.sqrt(zeta_j**2 + 3.0) - self.xi_cut)
        return np.where(np.isfinite(p), p, 0.0)

    def provenance(self) -> dict:
        return {**super().provenance(), "SELFN": str(self.fields_path), "XICUT": self.xi_cut}
