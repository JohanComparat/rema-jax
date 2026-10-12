"""SPT-SZ 2500d (Bocquet et al. 2019) and SPT-ECS: selection by xi in fields of different depth.

The detection significance xi of a cluster relates to its unbiased significance zeta by
P(xi | zeta) = N(sqrt(zeta^2 + 3), 1), and zeta scales with the depth of the field, gamma_f
(Bocquet et al. 2019, Eqs. 6-7). A cluster of measured xi_i in field f_i, moved to field f_j,
has zeta_j = sqrt(xi_i^2 - 3) gamma_{f_j} / gamma_{f_i} and is detected with probability

    P = Phi(sqrt(zeta_j^2 + 3) - xi_cut).

The fields are boxes in RA, Dec with their gamma (a FITS or CSV table FIELD, RA0, RA1, DEC0,
DEC1, GAMMA; RA0 < RA1 modulo 360); positions outside every field are outside the survey.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .base import ExternalSelection, register


def read_fields(path) -> dict:
    from astropy.table import Table

    t = Table.read(path)
    return {c.upper(): np.asarray(t[c]) for c in t.colnames}


@register("spt")
class SPTSelection(ExternalSelection):
    columns = ("EXT_XI", "EXT_FIELD")

    def __init__(self, fields: str | dict | None = None, xi_cut: float = 5.0):
        self.fields_path = fields if isinstance(fields, str) else None
        f = read_fields(fields) if isinstance(fields, str) else fields
        self.name_f = np.asarray([str(x).strip() for x in f["FIELD"]])
        self.box = np.stack([np.asarray(f[k], np.float64) for k in ("RA0", "RA1", "DEC0", "DEC1")], axis=1)
        self.gamma = np.asarray(f["GAMMA"], np.float64)
        self.xi_cut = float(xi_cut)

    def field_of(self, ra, dec) -> np.ndarray:
        ra, dec = np.asarray(ra, np.float64) % 360.0, np.asarray(dec, np.float64)
        out = np.full(ra.shape, -1)
        for i, (r0, r1, d0, d1) in enumerate(self.box):
            inside = (((ra - r0) % 360.0) < (r1 - r0)) & (dec >= d0) & (dec < d1)
            out[inside & (out < 0)] = i
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
