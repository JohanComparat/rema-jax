"""FITS table input/output with astropy.

Tables are plain ``dict[str, numpy.ndarray]`` in native byte order (FITS is big-endian; JAX and
numpy arithmetic want native order). Writes go to a temporary file renamed into place, so an
interrupted stage never leaves a truncated product behind.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from astropy.io import fits

Table = dict[str, np.ndarray]


def _native(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.byteorder not in ("=", "|"):
        a = a.astype(a.dtype.newbyteorder("="))
    return a


def read_table(path: str | Path, columns: Iterable[str] | None = None, rows=None,
               hdu: int | str = 1, upper: bool = True) -> Table:
    """Read columns of a FITS binary table.

    Parameters
    ----------
    columns : names to read (case-insensitive); all when None.
    rows : optional boolean mask or index array applied after reading each column.
    upper : return upper-case keys.
    """
    with fits.open(path, memmap=True) as h:
        data = h[hdu].data
        names = data.columns.names
        lookup = {n.upper(): n for n in names}
        wanted = names if columns is None else [lookup[c.upper()] for c in columns]
        out = {}
        for n in wanted:
            col = data[n]
            if rows is not None:
                col = col[rows]
            col = _native(np.array(col))
            if col.dtype.kind == "S":
                col = np.char.strip(col)
            out[n.upper() if upper else n] = col
    return out


def read_header(path: str | Path, hdu: int | str = 1) -> fits.Header:
    with fits.open(path, memmap=True) as h:
        return h[hdu].header.copy()


def nrows(path: str | Path, hdu: int | str = 1) -> int:
    return int(read_header(path, hdu)["NAXIS2"])


def _fits_format(a: np.ndarray) -> tuple[str, str | None]:
    """FITS TFORM (and TDIM for arrays with more than one trailing dimension)."""
    kind = a.dtype.kind
    size = a.dtype.itemsize
    if kind == "S" or kind == "U":
        width = max(1, int(a.dtype.itemsize if kind == "S" else a.dtype.itemsize // 4))
        return f"{width}A", None
    codes = {("b", 1): "L", ("i", 1): "I", ("u", 1): "B", ("i", 2): "I", ("u", 2): "J",
             ("i", 4): "J", ("u", 4): "K", ("i", 8): "K", ("f", 4): "E", ("f", 8): "D"}
    code = codes[(kind, size)]
    shape = a.shape[1:]
    n = int(np.prod(shape)) if shape else 1
    tdim = None
    if len(shape) > 1:
        tdim = "(" + ",".join(str(s) for s in reversed(shape)) + ")"
    return (f"{n}{code}" if n != 1 or shape else code), tdim


def table_hdu(columns: Mapping[str, np.ndarray], header: Mapping[str, Any] | None = None,
              extname: str | None = None) -> fits.BinTableHDU:
    cols = []
    for name, a in columns.items():
        a = np.asarray(a)
        if a.dtype.kind == "b" and a.dtype.itemsize == 1:
            pass
        elif a.dtype.kind == "i" and a.dtype.itemsize == 1:
            a = a.astype(np.int16)
        fmt, tdim = _fits_format(a)
        cols.append(fits.Column(name=name, format=fmt, dim=tdim, array=a))
    hdu = fits.BinTableHDU.from_columns(cols)
    if extname:
        hdu.name = extname
    for k, v in (header or {}).items():
        hdu.header[k] = v
    return hdu


def write_fits(path: str | Path, hdus: list[fits.hdu.base.ExtensionHDU],
               primary_header: Mapping[str, Any] | None = None) -> Path:
    """Write HDUs atomically (temporary file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prim = fits.PrimaryHDU()
    for k, v in (primary_header or {}).items():
        prim.header[k] = v
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    fits.HDUList([prim, *hdus]).writeto(tmp, overwrite=True)
    os.replace(tmp, path)
    return path


def write_table(path: str | Path, columns: Mapping[str, np.ndarray],
                header: Mapping[str, Any] | None = None, extname: str | None = None,
                primary_header: Mapping[str, Any] | None = None) -> Path:
    """Write one binary table (plus an empty primary HDU)."""
    return write_fits(path, [table_hdu(columns, header, extname)], primary_header)


# --------------------------------------------------------------------------- cluster catalogues
def _fits_columns(d: Mapping[str, np.ndarray]) -> Table:
    """Unicode columns as bytes (FITS has no unicode type)."""
    out = {}
    for k, v in d.items():
        a = np.asarray(v)
        out[k] = a.astype("S") if a.dtype.kind == "U" else a
    return out


def _nrows(t: Mapping[str, np.ndarray]) -> int:
    return len(next(iter(t.values()))) if t else 0


def write_catalog(path: str | Path, cat: Mapping[str, np.ndarray], mem: Mapping[str, np.ndarray] | None,
                  cfg=None, header: Mapping[str, Any] | None = None,
                  members_path: str | Path | None = None) -> Path:
    """Write a cluster catalogue: CLUSTERS, MEMBERS (or ``members_path``) and CONFIG HDUs.

    The primary header gets REMAVER, NCLUSTER, NMEMBER and ``header``. An empty catalogue keeps
    empty CLUSTERS and MEMBERS HDUs, which :func:`read_catalog` reads back as empty dicts.
    """
    from .. import __version__

    def hdu(t, name):
        return table_hdu(_fits_columns(t), extname=name) if t else fits.BinTableHDU(name=name)

    prim = {"REMAVER": __version__, "NCLUSTER": _nrows(cat), "NMEMBER": _nrows(mem or {}),
            **(header or {})}
    hdus = [hdu(cat, "CLUSTERS")]
    if members_path is None:
        hdus.append(hdu(mem or {}, "MEMBERS"))
    else:
        write_fits(members_path, [hdu(mem or {}, "MEMBERS")], prim)
    if cfg is not None:
        hdus.append(table_hdu({"YAML": np.array([cfg.to_yaml().encode()])}, extname="CONFIG"))
    return write_fits(path, hdus, prim)


def read_catalog(path: str | Path, members: bool = True) -> tuple[Table, Table, fits.Header]:
    """(clusters, members, primary header) of a catalogue written by :func:`write_catalog`; empty
    dicts for empty or absent HDUs."""
    with fits.open(path, memmap=True) as h:
        hdr = h[0].header.copy()
        full = {x.name for x in h if x.name in ("CLUSTERS", "MEMBERS") and x.data is not None
                and len(x.columns.names) > 0}
    cat = read_table(path, hdu="CLUSTERS") if "CLUSTERS" in full else {}
    mem = read_table(path, hdu="MEMBERS") if members and "MEMBERS" in full else {}
    return cat, mem, hdr
