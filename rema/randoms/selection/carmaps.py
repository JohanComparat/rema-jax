"""FITS images (CAR, TAN or any WCS) to NESTED HEALPix sparse maps.

The SZ selection functions are released as images: the noise (RMS) maps and area masks of
nemo (ACT), tile by tile. A random point needs the value at its position; the images are
reduced once to a :class:`rema.sky.maps.SparseMap` at ``nside`` (the median of the image
pixels whose centres fall in each HEALPix pixel), which the selection plugins look up.
"""

from __future__ import annotations

import numpy as np

from ...sky.healpix import ang2pix_nest_np
from ...sky.maps import SparseMap


def image_to_pixels(path, nside: int, hdu=None, valid=lambda v: np.isfinite(v) & (v > 0),
                    chunk_rows: int = 512):
    """(HEALPix pixel, value) of every valid image pixel of the FITS image(s) in ``path``
    (all image HDUs, or ``hdu``), reading ``chunk_rows`` rows at a time."""
    from astropy.io import fits
    from astropy.wcs import WCS

    pix, val = [], []
    with fits.open(path, memmap=True) as h:
        hdus = [h[hdu]] if hdu is not None else [x for x in h if x.is_image and x.data is not None
                                                  and x.data.ndim >= 2]
        for x in hdus:
            data = x.data
            while data.ndim > 2:
                data = data[0]
            w = WCS(x.header).celestial
            ny, nx = data.shape
            xx = np.arange(nx)
            for y0 in range(0, ny, chunk_rows):
                block = np.asarray(data[y0:y0 + chunk_rows], np.float64)
                ok = valid(block)
                if not ok.any():
                    continue
                yy, xi = np.nonzero(ok)
                ra, dec = w.all_pix2world(xx[xi], yy + y0, 0)
                good = np.isfinite(ra) & np.isfinite(dec)
                pix.append(ang2pix_nest_np(nside, np.asarray(ra)[good] % 360.0, np.asarray(dec)[good]))
                val.append(block[ok][good])
    if not pix:
        return np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(pix), np.concatenate(val)


def images_to_sparsemap(paths, nside: int, name: str, reducer: str = "median", **kw) -> SparseMap:
    """SparseMap with column ``name``: the ``reducer`` ("median" or "mean") of the image values
    in each HEALPix pixel, over the images ``paths``.

    >>> import tempfile, os
    >>> from astropy.io import fits
    >>> from astropy.wcs import WCS
    >>> w = WCS(naxis=2); w.wcs.ctype = ["RA---CAR", "DEC--CAR"]; w.wcs.crval = [10, 0]
    >>> w.wcs.crpix = [51, 51]; w.wcs.cdelt = [-0.01, 0.01]
    >>> d = tempfile.mkdtemp(); p = os.path.join(d, "rms.fits")
    >>> fits.PrimaryHDU(np.full((100, 100), 2.0), header=w.to_header()).writeto(p)
    >>> m = images_to_sparsemap([p], 256, "RMS")
    >>> float(m.lookup_np(np.array([10.0]), np.array([0.0]), "RMS")[0])
    2.0
    """
    pix, val = [], []
    for p in np.atleast_1d(paths):
        a, b = image_to_pixels(p, nside, **kw)
        pix.append(a)
        val.append(b)
    pix, val = np.concatenate(pix), np.concatenate(val)
    if reducer == "mean":
        u, inv = np.unique(pix, return_inverse=True)
        out = np.bincount(inv, weights=val) / np.bincount(inv)
    elif reducer == "median":
        o = np.lexsort((val, pix))
        pix, val = pix[o], val[o]
        u, start, cnt = np.unique(pix, return_index=True, return_counts=True)
        lo, hi = start + (cnt - 1) // 2, start + cnt // 2
        out = 0.5 * (val[lo] + val[hi])
    else:
        raise ValueError(f"reducer {reducer!r}: 'median' or 'mean'")
    return SparseMap(nside, u, {name: out.astype(np.float32)})
