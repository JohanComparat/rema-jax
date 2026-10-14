"""The rema measurements at the positions of an X-ray or SZ catalogue, joined to that catalogue.

``rema scan-positions`` kept a few input columns (INPUT_NAME, INPUT_SNR, ...); the selection
functions need others (the fixed-filter y0 and S/N of ACT, the field of SPT, the count rate of
eRASS1). :func:`join_scan` adds them from the original catalogue, by the input row
(INPUT_ROW, the row in the original file), as ``EXT_<NAME>`` columns.
"""

from __future__ import annotations

import numpy as np

#: catalogue -> (original file name, {``EXT_`` column: original column})
JOIN_COLUMNS = {
    "act_dr6": ("DR6_cluster-catalog_v1.0.fits", {
        "EXT_FIXED_Y_C": "fixed_y_c", "EXT_FIXED_ERR_Y_C": "fixed_err_y_c", "EXT_FIXED_SNR": "fixed_SNR",
        "EXT_SNR": "SNR", "EXT_FLAGS": "flags", "EXT_TILE": "tileName", "EXT_Y500": "inferred_Y500Arcmin2",
        "EXT_THETA500": "theta500Arcmin", "EXT_Z": "redshift", "EXT_M500": "M500c",
        "EXT_FOOTPRINT_LEGACY": "footprint_Legacy"}),
    "spt_2500d": ("2500d_cluster_sample_Bocquet19.fits", {
        "EXT_XI": "XI", "EXT_FIELD": "FIELD", "EXT_YSZ": "YSZ", "EXT_Z": "REDSHIFT", "EXT_M500": "M500"}),
    "psz2": ("HFI_PCCS_SZ-union_R2.08.fits.gz", {
        "EXT_SNR": "SNR", "EXT_Y5R500": "Y5R500", "EXT_COSMO": "COSMO", "EXT_Z": "REDSHIFT",
        "EXT_MSZ": "MSZ"}),
    "erass1": ("erass1cl_primary_v3.2.fits", {
        "EXT_EXT_LIKE": "EXT_LIKE", "EXT_DET_LIKE": "DET_LIKE_0", "EXT_EXP": "EXP", "EXT_CR500": "CR500",
        "EXT_CTS500": "CTS500", "EXT_F500": "F500", "EXT_L500": "L500", "EXT_R500": "R500",
        "EXT_Z": "BEST_Z", "EXT_M500": "M500", "EXT_PCONT": "PCONT"}),
    "mcxc": ("mcxc.fits", {"EXT_LX500": "LX_500", "EXT_Z": "REDSHIFT", "EXT_M500": "MASS_500"}),
}


def join_scan(scan: dict, original: dict, catalogue: str) -> dict:
    """Scan table ``scan`` (dict of columns, with INPUT_ROW) plus the ``EXT_`` columns of
    ``catalogue`` taken from ``original`` (dict of the original catalogue's columns).

    >>> scan = {"INPUT_ROW": np.array([2, 0]), "INPUT_NAME": np.array(["c", "a"]), "LAMBDA": np.array([5., 9.])}
    >>> orig = {"NAME": np.array(["a", "b", "c"]), "SNR": np.array([4., 5., 6.]), "Y5R500": np.zeros(3),
    ...         "COSMO": np.ones(3), "REDSHIFT": np.zeros(3), "MSZ": np.zeros(3)}
    >>> join_scan(scan, orig, "psz2")["EXT_SNR"].tolist()
    [6.0, 4.0]
    """
    if catalogue not in JOIN_COLUMNS:
        raise KeyError(f"unknown catalogue {catalogue!r}; known: {sorted(JOIN_COLUMNS)}")
    _, cols = JOIN_COLUMNS[catalogue]
    up = {k.upper(): k for k in original}
    row = np.asarray(scan["INPUT_ROW"]).astype(np.int64)
    n = len(next(iter(original.values())))
    if row.min(initial=0) < 0 or row.max(initial=0) >= n:
        raise ValueError(f"INPUT_ROW outside the original catalogue ({n} rows)")
    if "INPUT_NAME" in scan and "NAME" in up:
        a = np.char.strip(np.asarray(scan["INPUT_NAME"]).astype(str))
        b = np.char.strip(np.asarray(original[up["NAME"]]).astype(str))[row]
        if not np.array_equal(a, b):
            raise ValueError(f"{int(np.sum(a != b))} names differ between the scan and the original rows")
    out = dict(scan)
    for ext, col in cols.items():
        v = np.asarray(original[up[col.upper()]])[row]
        out[ext] = np.char.strip(v.astype(str)) if v.dtype.kind in "SUO" else v
    return out


def read_original(path, catalogue: str) -> dict:
    from astropy.table import Table

    t = Table.read(path, hdu=1)
    want = {c.upper() for c in JOIN_COLUMNS[catalogue][1].values()} | {"NAME"}
    out = {}
    for c in t.colnames:
        if c.upper() in want:
            col = t[c]
            if getattr(col, "mask", None) is not None and np.any(col.mask):
                col = col.filled("" if col.dtype.kind in "US" else -1)
            out[c] = np.asarray(col)
    return out
