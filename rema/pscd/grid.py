"""The angular grid of PSCD: a gnomonic projection of the region, cloud-in-cell painting of
galaxy weights and convolution with the cluster profile.

The tangent point is the centre of the region's bounding box; over the 10-20 deg of a region the
projection distorts distances by less than 1 % (a cluster profile spans at most a degree).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import oaconvolve

from ..sky.neighbors import radec_from_unit, unit_vectors


def gnomonic(ra, dec, ra0: float, dec0: float):
    """(xi, eta) [deg] of (ra, dec) on the plane tangent at (ra0, dec0)."""
    a, d = np.radians(np.asarray(ra, np.float64)), np.radians(np.asarray(dec, np.float64))
    a0, d0 = np.radians(ra0), np.radians(dec0)
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    return np.degrees(xi), np.degrees(eta)


def inverse_gnomonic(xi, eta, ra0: float, dec0: float):
    """(ra, dec) [deg] of tangent-plane coordinates (xi, eta) [deg].

    >>> ra, dec = inverse_gnomonic(*gnomonic([10.0, 359.0], [-50.0, 3.0], 5.0, -20.0), 5.0, -20.0)
    >>> np.round(ra, 8).tolist(), np.round(dec, 8).tolist()
    ([10.0, 359.0], [-50.0, 3.0])
    """
    x, y = np.radians(np.asarray(xi, np.float64)), np.radians(np.asarray(eta, np.float64))
    a0, d0 = np.radians(ra0), np.radians(dec0)
    rho = np.hypot(x, y)
    c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.arcsin(np.cos(c) * np.sin(d0) + np.where(rho > 0, y * np.sin(c) * np.cos(d0) / rho, 0.0))
    ra = a0 + np.arctan2(x * np.sin(c), rho * np.cos(d0) * np.cos(c) - y * np.sin(d0) * np.sin(c))
    return np.degrees(ra) % 360.0, np.degrees(dec)


@dataclass
class Grid:
    """Pixels of side ``pixel`` [deg] on the tangent plane at (ra0, dec0); pixel (iy, ix) is
    centred at xi = x0 + ix pixel, eta = y0 + iy pixel."""

    ra0: float
    dec0: float
    pixel: float
    x0: float
    y0: float
    nx: int
    ny: int

    @classmethod
    def covering(cls, ra, dec, pixel: float, margin: float = 0.0) -> "Grid":
        """The grid covering the points (ra, dec), plus ``margin`` deg on every side."""
        ra, dec = np.asarray(ra, np.float64), np.asarray(dec, np.float64)
        if ra.size == 0:
            raise ValueError("no point to cover")
        c = unit_vectors(ra, dec).mean(axis=0)
        ra0, dec0 = radec_from_unit(c / np.linalg.norm(c))
        xi, eta = gnomonic(ra, dec, ra0, dec0)
        x0, y0 = xi.min() - margin, eta.min() - margin
        nx = int(np.ceil((xi.max() + margin - x0) / pixel)) + 1
        ny = int(np.ceil((eta.max() + margin - y0) / pixel)) + 1
        return cls(float(ra0), float(dec0), float(pixel), float(x0), float(y0), nx, ny)

    @property
    def shape(self):
        return self.ny, self.nx

    @property
    def area(self) -> float:
        """Pixel area [deg^2]."""
        return self.pixel * self.pixel

    def to_pix(self, ra, dec):
        """Fractional pixel coordinates (px, py) of sky positions."""
        xi, eta = gnomonic(ra, dec, self.ra0, self.dec0)
        return (xi - self.x0) / self.pixel, (eta - self.y0) / self.pixel

    def to_sky(self, px, py):
        """Sky positions of fractional pixel coordinates."""
        return inverse_gnomonic(self.x0 + np.asarray(px) * self.pixel, self.y0 + np.asarray(py) * self.pixel,
                                self.ra0, self.dec0)

    def centres(self):
        """(ra, dec) of every pixel centre, [ny, nx] each."""
        px, py = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        return self.to_sky(px, py)


def paint(px, py, w, shape) -> np.ndarray:
    """Cloud-in-cell sum of the weights ``w`` at fractional pixel coordinates on a grid of
    ``shape`` (ny, nx); weight falling off the grid is lost.

    >>> img = paint([1.25], [2.0], [4.0], (4, 4))
    >>> img[2, 1], img[2, 2], img.sum()
    (3.0, 1.0, 4.0)
    """
    ny, nx = shape
    px, py, w = (np.asarray(a, np.float64) for a in (px, py, w))
    ix, iy = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
    fx, fy = px - ix, py - iy
    out = np.zeros(ny * nx)
    for dx, wx in ((0, 1 - fx), (1, fx)):
        for dy, wy in ((0, 1 - fy), (1, fy)):
            jx, jy = ix + dx, iy + dy
            ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
            out += np.bincount(jy[ok] * nx + jx[ok], weights=(w * wx * wy)[ok], minlength=ny * nx)
    return out.reshape(ny, nx)


def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """``img`` convolved with a centred, odd-sized ``kernel`` (same shape as ``img``; zero outside).

    >>> k = np.zeros((3, 3)); k[1, 1] = 2.0; k[1, 2] = 1.0
    >>> np.round(convolve(np.eye(4), k)[1], 6).tolist()
    [0.0, 2.0, 1.0, 0.0]
    """
    if not img.any():
        return np.zeros(img.shape)
    return oaconvolve(img, kernel, mode="same")
