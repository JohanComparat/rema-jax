"""FITS products of the cluster randoms.

Copies file (``rema cluster-randoms copies``, completed by ``place``): HDUs RANDOMS (one row per
copy: SRC, the row of its cluster in SOURCES, RA, DEC, ZVLIM, and after ``place`` R_LAMBDA,
SCALEVAL, MASKFRAC), SOURCES (one row per cluster copied: ROW in the catalogue, Z, Z_RAW, LAMBDA, OBS,
LAMBDA_DET, BIN_Z, BIN_OBS, N_SAMP, N_POOL, FLAG, ...), SPEC (the sample YAML) and CONFIG (the
catalogue's configuration).

Sample file (``rema cluster-randoms sample``): HDUs CLUSTERS (the sample, WEIGHT = 1), RANDOMS
(RA, DEC, Z, WEIGHT, SRC, LAMBDA, OBS, BIN_Z, BIN_OBS, Q, P_EXT, SCALEVAL, MASKFRAC, ZVLIM),
SOURCES (per cluster: ROW, N_SAMP, SUMQ, NEFF, NPASS), SPEC and CONFIG. Headers record the
inputs (catalogue, z_vlim map, domain), the selection (SELNAME, APPROX) and the version.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..io.tables import read_header, read_table, table_hdu, write_fits


def _text_hdu(name: str, text: str):
    return table_hdu({"YAML": np.array([text.encode()])}, extname=name)


def _read_text(path, name: str) -> str | None:
    from astropy.io import fits

    with fits.open(path) as h:
        if name not in h:
            return None
        t = h[name].data["YAML"][0]
    return t.decode() if isinstance(t, bytes) else str(t)


def provenance() -> dict:
    import rema

    return {"REMAVER": rema.__version__, "RANDFMT": 1}


def write_copies(path, rand: dict, src: dict, header: dict, spec_yaml: str, cfg_yaml: str) -> Path:
    hdr = {**provenance(), **header}
    return write_fits(path, [table_hdu(rand, hdr, "RANDOMS"), table_hdu(src, hdr, "SOURCES"),
                             _text_hdu("SPEC", spec_yaml), _text_hdu("CONFIG", cfg_yaml)])


_STRUCTURAL = re.compile(r"^(XTENSION|BITPIX|NAXIS\d*|PCOUNT|GCOUNT|TFIELDS|EXTNAME|CHECKSUM|DATASUM|"
                         r"COMMENT|HISTORY|T(TYPE|FORM|DIM|UNIT|NULL|ZERO|SCAL|DISP)\d+)$")


def info_header(hdr) -> dict:
    """The descriptive keywords of a header (without the table structure)."""
    return {k: v for k, v in hdr.items() if k and not _STRUCTURAL.match(k)}


def read_copies(path) -> tuple[dict, dict, dict, str, str]:
    """(randoms, sources, header keywords, spec YAML, config YAML)."""
    hdr = info_header(read_header(path, "RANDOMS"))
    return (read_table(path, hdu="RANDOMS"), read_table(path, hdu="SOURCES"), hdr,
            _read_text(path, "SPEC"), _read_text(path, "CONFIG"))


def write_sample(path, data: dict, rand: dict, src: dict, header: dict, spec_yaml: str,
                 cfg_yaml: str) -> Path:
    hdr = {**provenance(), **header}
    return write_fits(path, [table_hdu(data, hdr, "CLUSTERS"), table_hdu(rand, hdr, "RANDOMS"),
                             table_hdu(src, hdr, "SOURCES"), _text_hdu("SPEC", spec_yaml),
                             _text_hdu("CONFIG", cfg_yaml)])
