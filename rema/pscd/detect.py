"""PSCD detection: the amplitude cube and the extraction-and-cleaning loop (AMICO, Bellagamba
et al. 2018, Sects. 2-3).

Per redshift slice z_k of the grid, the galaxies' weights w_i = Phi(m_i) p_i(z_k) / N(m_i, z_k)
(times their field probability) are painted on the angular grid and convolved with the profile
Psi_k, which gives S. The footprint (the unmasked fraction f of each pixel) convolved with Psi,
Psi^2 and Psi^3, times the magnitude integrals at the local depth, gives beta, alpha and gamma
(Sect. 2.5: the integrals run over the available area only). The cube holds S, alpha, beta, gamma
and the detection score L = A^2 alpha where A > 0 and S/N >= snr_min.

Detections are then taken one at a time, in order of decreasing likelihood: position and redshift
refined by parabolas through the neighbouring cells, membership probabilities of the galaxies
within the truncation radius, and the members' weighted contributions subtracted from S (the
cleaning). The S/N of the threshold is AMICO's A / sigma_A, which includes the cluster's own
shot noise, or (``pscd.snr_kind: background``) A sqrt(alpha), against the background only. The
redshift error Z_E is that of the members' photo-z, (sum_i P_i / s_i^2)^-1/2. The highest cell is found through the maxima of square tiles of pixels, of which only
those touched by a cleaning are recomputed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import RemaConfig
from ..model.background import ZredBkg
from ..sky.neighbors import NeighborIndex, unit_vectors
from .grid import Grid, convolve, paint
from .model import MagIntegrals, Template, WidthTable, noise_np

log = logging.getLogger(__name__)

SQRT2PI = float(np.sqrt(2.0 * np.pi))


@dataclass
class Galaxies:
    """The galaxies of a PSCD run (usable photo-z, brighter than the magnitude limit)."""

    rows: np.ndarray          # rows in the input table
    ra: np.ndarray
    dec: np.ndarray
    refmag: np.ndarray
    zphot: np.ndarray
    s: np.ndarray             # calibrated photo-z width
    px: np.ndarray            # grid coordinates
    py: np.ndarray
    pfield: np.ndarray        # field probability P_f (1 at the start)
    _index: NeighborIndex | None = None
    _xyz: np.ndarray | None = None

    @property
    def index(self) -> NeighborIndex:
        if self._index is None:
            self._index = NeighborIndex(self.ra, self.dec)
        return self._index

    @property
    def xyz(self) -> np.ndarray:
        if self._xyz is None:
            self._xyz = unit_vectors(self.ra, self.dec)
        return self._xyz


@dataclass
class Slice:
    z: float
    kernel: np.ndarray        # Psi on the grid
    kernel200: np.ndarray     # Psi within r200, normalised to sum 1 (MASKFRAC)
    mags: MagIntegrals
    noise_m: np.ndarray       # N(m, z) on the noise table's magnitudes
    mstar: float

    @property
    def half(self) -> int:
        return self.kernel.shape[0] // 2


@dataclass
class Cube:
    grid: Grid
    zs: np.ndarray
    slices: list
    S: np.ndarray             # [nz, ny, nx] float32
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    score: np.ndarray
    fmap: np.ndarray          # [ny, nx] unmasked fraction
    sigmap: np.ndarray | None  # [ny, nx] 1-sigma flux error of the reference band (None: no depth)
    tpl: Template
    bkg: ZredBkg
    cfg: RemaConfig
    mag_limit: float
    tmax: np.ndarray = field(default=None)


def magnitude_limit(cfg: RemaConfig) -> float:
    """Faint limit of the PSCD galaxies: the catalogue's and ``photoz.mag_max`` (99: none)."""
    lim = float(cfg.survey.mag_max) if cfg.survey.mag_max else 99.0
    if cfg.photoz.mag_max is not None:
        lim = min(lim, float(cfg.photoz.mag_max))
    return lim


def z_grid(cfg: RemaConfig) -> np.ndarray:
    """Redshift slices: the run's range, plus one slice on each side for the peak refinement."""
    pc = cfg.pscd
    zr = pc.zrange if pc.zrange is not None else cfg.model.zrange
    n = int(round((zr[1] - zr[0]) / pc.dz))
    return np.round(zr[0] + pc.dz * np.arange(-1, n + 2), 6)


def _weights(gal: Galaxies, sl: Slice, tpl: Template, bkg_m: np.ndarray, nsig: float, rows=None):
    """(galaxy indices, w_i = Phi p_i / N) of the galaxies contributing at slice ``sl``."""
    zp, s, m = (gal.zphot, gal.s, gal.refmag) if rows is None else (gal.zphot[rows], gal.s[rows], gal.refmag[rows])
    x = (zp - sl.z) / s
    sel = np.flatnonzero(np.abs(x) < nsig)
    if sel.size == 0:
        return (sel if rows is None else rows[sel]), np.zeros(0)
    p = np.exp(-0.5 * x[sel] ** 2) / (SQRT2PI * s[sel])
    N = np.interp(m[sel], bkg_m, sl.noise_m, left=np.inf, right=np.inf)
    ok = np.isfinite(N) & (N > 0)
    w = np.where(ok, tpl.phi(m[sel], sl.z) * p / np.where(ok, N, 1.0), 0.0)
    return (sel if rows is None else rows[sel]), w


def build_cube(gal: Galaxies, grid: Grid, tpl: Template, bkg: ZredBkg, widths: WidthTable,
               cfg: RemaConfig, fmap: np.ndarray, sigmap: np.ndarray | None = None,
               mag_limit: float | None = None) -> Cube:
    """The amplitude cube of a region (see the module docstring).

    ``fmap``: unmasked fraction of every pixel; ``sigmap``: the reference band's 1-sigma flux
    error, which with ``survey.ref_snr_min`` gives the local magnitude limit (None: everywhere
    ``mag_limit``, default :func:`magnitude_limit`).
    """
    pc = cfg.pscd
    if pc.snr_kind not in ("amico", "background"):
        raise ValueError(f"pscd.snr_kind {pc.snr_kind!r}: expected 'amico' or 'background'")
    zs = z_grid(cfg)
    lim = magnitude_limit(cfg) if mag_limit is None else float(mag_limit)
    bkg_m = np.asarray(bkg.mag, np.float64)
    shape = (zs.size,) + grid.shape
    S, alpha, beta, gamma = (np.zeros(shape, np.float32) for _ in range(4))
    slices = []
    t0 = time.time()
    snr_min = cfg.survey.ref_snr_min
    zp = cfg.survey.zeropoint
    for k, z in enumerate(zs):
        ker = tpl.kernel(z, grid.pixel)
        k200 = tpl.kernel(z, grid.pixel, rmax=float(tpl.r200_deg(z)))
        sl = Slice(float(z), ker, k200 / k200.sum(),
                   MagIntegrals.build(tpl, bkg, widths, float(z), min(lim, 30.0)),
                   noise_np(bkg, float(z), bkg_m), float(tpl.mstar(z)))
        slices.append(sl)
        idx, w = _weights(gal, sl, tpl, bkg_m, cfg.photoz.nsig_max)
        img = paint(gal.px[idx], gal.py[idx], w * gal.pfield[idx], grid.shape)
        S[k] = convolve(img, ker)
        f1 = convolve(fmap, ker) * grid.area
        f2 = convolve(fmap, ker**2) * grid.area
        f3 = convolve(fmap, ker**3) * grid.area
        if sigmap is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                sbar = convolve(fmap * sigmap, ker) * grid.area / f1
                mlim = np.where(sbar > 0, zp - 2.5 * np.log10(snr_min * sbar), -np.inf)
            mlim = np.minimum(mlim, lim)
        else:
            mlim = np.full(grid.shape, min(lim, 30.0))
        i1, i2, i3 = sl.mags.at(mlim)
        covered = f1 >= pc.min_coverage * ker.sum() * grid.area
        alpha[k] = np.where(covered, f2 * i2, 0.0)
        beta[k] = np.where(covered, f1 * i1, 0.0)
        gamma[k] = np.where(covered, f3 * i3, 0.0)
        if k % 10 == 0:
            log.info("pscd: slice %d/%d (z = %.2f, %d galaxies), %.1fs", k + 1, zs.size, z, idx.size,
                     time.time() - t0)
    cube = Cube(grid, zs, slices, S, alpha, beta, gamma, np.full(shape, -np.inf, np.float32),
                fmap, sigmap, tpl, bkg, cfg, lim)
    for k in range(zs.size):
        cube.score[k] = _score(cube, k, slice(None), slice(None))
    cube.tmax = _tile_max(cube.score.max(axis=0), pc.tile)
    return cube


def amplitude(S, alpha, beta, gamma):
    """(A, sigma_A, S/N, L = A^2 alpha); A = 0, sigma = inf where alpha = 0."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        good = alpha > 0
        A = np.where(good, (S - beta) / np.where(good, alpha, 1.0), 0.0)
        var = np.where(good, 1.0 / np.where(good, alpha, 1.0)
                       + np.maximum(A, 0.0) * gamma / np.where(good, alpha, 1.0) ** 2, np.inf)
        sig = np.sqrt(var)
        return A, sig, np.where(good, A / sig, 0.0), A * A * alpha


def _score(cube: Cube, k, ys, xs):
    a = cube.alpha[k, ys, xs]
    A, _, snr, L = amplitude(cube.S[k, ys, xs], a, cube.beta[k, ys, xs], cube.gamma[k, ys, xs])
    if cube.cfg.pscd.snr_kind == "background":
        snr = A * np.sqrt(np.maximum(a, 0.0))
    ok = (A > 0) & (snr >= cube.cfg.pscd.snr_min) & np.isfinite(L)
    return np.where(ok, L, -np.inf).astype(np.float32)


def _tile_max(img2d, T):
    ny, nx = img2d.shape
    ty, tx = -(-ny // T), -(-nx // T)
    pad = np.full((ty * T, tx * T), -np.inf, np.float32)
    pad[:ny, :nx] = img2d
    return pad.reshape(ty, T, tx, T).max(axis=(1, 3))


def _vertex(lm, l0, lp):
    """Offset of the parabola vertex through (-1, lm), (0, l0), (1, lp), clipped to +-0.5."""
    d = lm - 2.0 * l0 + lp
    if not (np.isfinite(d) and d < 0):
        return 0.0
    return float(np.clip(0.5 * (lm - lp) / d, -0.5, 0.5))


@dataclass
class Detection:
    k: int
    y: int
    x: int
    z: float
    z_e: float
    ra: float
    dec: float
    A: float
    A_e: float
    snr: float
    snr_nocl: float
    lnlike: float
    n_exp: float
    mlim: float
    maskfrac: float


def _cell_values(cube: Cube, k, y, x):
    sl = (k, y, x)
    return amplitude(cube.S[sl], cube.alpha[sl], cube.beta[sl], cube.gamma[sl])


def _refine(cube: Cube, k, y, x) -> Detection:
    nz, ny, nx = cube.S.shape
    L = lambda kk, yy, xx: float(_cell_values(cube, kk, yy, xx)[3]) if (
        0 <= kk < nz and 0 <= yy < ny and 0 <= xx < nx) else np.nan
    A, sig, snr, l0 = (float(v) for v in _cell_values(cube, k, y, x))
    dz = _vertex(L(k - 1, y, x), l0, L(k + 1, y, x))
    dy = _vertex(L(k, y - 1, x), l0, L(k, y + 1, x))
    dx = _vertex(L(k, y, x - 1), l0, L(k, y, x + 1))
    step = cube.cfg.pscd.dz
    ra, dec = cube.grid.to_sky(x + dx, y + dy)
    sl = cube.slices[k]
    h = sl.kernel200.shape[0] // 2
    fwin = _window(cube.fmap, y, x, h)
    maskfrac = 1.0 - float(np.sum(fwin * sl.kernel200))
    mlim = cube.mag_limit
    if cube.sigmap is not None:
        hk = sl.half
        kw = sl.kernel
        fw, sw = _window(cube.fmap, y, x, hk), _window(cube.sigmap, y, x, hk)
        den = float(np.sum(fw * kw))
        sbar = float(np.sum(fw * sw * kw)) / den if den > 0 else 0.0
        if sbar > 0:
            mlim = min(mlim, cube.cfg.survey.zeropoint - 2.5 * np.log10(cube.cfg.survey.ref_snr_min * sbar))
    return Detection(k=k, y=y, x=x, z=float(cube.zs[k] + dz * step), z_e=np.nan, ra=float(ra), dec=float(dec),
                     A=A, A_e=sig, snr=snr, snr_nocl=float(A * np.sqrt(cube.alpha[k, y, x])), lnlike=l0,
                     n_exp=float(A * cube.beta[k, y, x]), mlim=float(mlim), maskfrac=maskfrac)


def _window(img, y, x, h):
    """img[y-h:y+h+1, x-h:x+h+1], zero-padded at the edges."""
    ny, nx = img.shape
    out = np.zeros((2 * h + 1, 2 * h + 1), img.dtype)
    y0, y1, x0, x1 = max(0, y - h), min(ny, y + h + 1), max(0, x - h), min(nx, x + h + 1)
    out[y0 - (y - h):y1 - (y - h), x0 - (x - h):x1 - (x - h)] = img[y0:y1, x0:x1]
    return out


def memberships(cube: Cube, gal: Galaxies, det: Detection, nsig: float):
    """(galaxy indices, P(i in j), angular distance [deg]) of the galaxies around a detection
    (Bellagamba et al. 2018, Eq. 24, with the current field probabilities)."""
    tpl = cube.tpl
    rmax = float(tpl.rmax_deg(det.z))
    idx = np.asarray(gal.index.query_lists(np.array([det.ra]), np.array([det.dec]), rmax)[0], np.int64)
    if idx.size == 0:
        return idx, np.zeros(0), np.zeros(0)
    c = unit_vectors(det.ra, det.dec)
    th = np.degrees(2.0 * np.arcsin(np.clip(np.linalg.norm(gal.xyz[idx] - c, axis=1) / 2.0, 0.0, 1.0)))
    x = (gal.zphot[idx] - det.z) / gal.s[idx]
    m = gal.refmag[idx]
    p = np.where(np.abs(x) < nsig, np.exp(-0.5 * x * x) / (SQRT2PI * gal.s[idx]), 0.0)
    Mc = tpl.psi(th, det.z) * tpl.phi(m, det.z)
    N = noise_np(cube.bkg, det.z, m)
    sig = max(det.A, 0.0) * Mc * p
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(np.isfinite(N) & (sig > 0), gal.pfield[idx] * sig / (sig + N), 0.0)
    keep = P > 0
    return idx[keep], P[keep], th[keep]


def clean(cube: Cube, gal: Galaxies, idx, P, nsig: float):
    """Remove the members' P-weighted contributions from S (and update the score); returns the
    touched (y0, y1, x0, x1) windows."""
    if idx.size == 0:
        return []
    bkg_m = np.asarray(cube.bkg.mag, np.float64)
    zp, s = gal.zphot[idx], gal.s[idx]
    lo = np.searchsorted(cube.zs, (zp - nsig * s).min())
    hi = np.searchsorted(cube.zs, (zp + nsig * s).max(), side="right")
    ny, nx = cube.grid.shape
    touched = []
    for k in range(max(lo, 0), min(hi, cube.zs.size)):
        sl = cube.slices[k]
        rows, w = _weights(gal, sl, cube.tpl, bkg_m, nsig, rows=idx)
        if rows.size == 0:
            continue
        wP = w * P[np.searchsorted(idx, rows)]
        h = sl.half
        px, py = gal.px[rows], gal.py[rows]
        x0 = max(int(np.floor(px.min())) - h - 1, 0)
        x1 = min(int(np.ceil(px.max())) + h + 2, nx)
        y0 = max(int(np.floor(py.min())) - h - 1, 0)
        y1 = min(int(np.ceil(py.max())) + h + 2, ny)
        if x1 <= x0 or y1 <= y0:
            continue
        # paint on a window that holds the whole kernel around every member, then crop
        X0, Y0 = int(np.floor(px.min())) - h - 1, int(np.floor(py.min())) - h - 1
        W = int(np.ceil(px.max())) + h + 2 - X0
        H = int(np.ceil(py.max())) + h + 2 - Y0
        img = convolve(paint(px - X0, py - Y0, wP, (H, W)), sl.kernel)
        cube.S[k, y0:y1, x0:x1] -= img[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0].astype(np.float32)
        cube.score[k, y0:y1, x0:x1] = _score(cube, k, slice(y0, y1), slice(x0, x1))
        touched.append((y0, y1, x0, x1))
    return touched


def _update_tiles(cube: Cube, windows):
    T = cube.cfg.pscd.tile
    tiles = set()
    for y0, y1, x0, x1 in windows:
        for ty in range(y0 // T, (y1 - 1) // T + 1):
            for tx in range(x0 // T, (x1 - 1) // T + 1):
                tiles.add((ty, tx))
    for ty, tx in tiles:
        cube.tmax[ty, tx] = cube.score[:, ty * T:(ty + 1) * T, tx * T:(tx + 1) * T].max()


def extract(cube: Cube, gal: Galaxies, log_every: int = 500):
    """The detection loop. Returns (list of Detection, list of (indices, P, theta) members)."""
    pc = cube.cfg.pscd
    nsig = cube.cfg.photoz.nsig_max
    T = pc.tile
    dets, mems = [], []
    t0 = time.time()
    while len(dets) < pc.max_detections:
        t = int(np.argmax(cube.tmax))
        ty, tx = np.unravel_index(t, cube.tmax.shape)
        if not np.isfinite(cube.tmax[ty, tx]):
            break
        block = cube.score[:, ty * T:(ty + 1) * T, tx * T:(tx + 1) * T]
        k, yy, xx = np.unravel_index(int(np.argmax(block)), block.shape)
        y, x = ty * T + yy, tx * T + xx
        det = _refine(cube, k, y, x)
        idx, P, th = memberships(cube, gal, det, nsig)
        order = np.argsort(idx)
        idx, P, th = idx[order], P[order], th[order]
        if P.sum() > 0:
            # redshift error: the members' photo-z, inverse-variance weighted
            det.z_e = float(1.0 / np.sqrt(np.sum(P / gal.s[idx] ** 2)))
        gal.pfield[idx] = np.maximum(gal.pfield[idx] - P, 0.0)
        windows = clean(cube, gal, idx, P, nsig)
        cube.score[k, y, x] = -np.inf          # progress, whatever the cleaning left
        _update_tiles(cube, windows + [(y, y + 1, x, x + 1)])
        dets.append(det)
        mems.append((idx, P, th))
        if log_every and len(dets) % log_every == 0:
            log.info("pscd: %d detections (S/N %.2f), %.1fs", len(dets), det.snr, time.time() - t0)
    log.info("pscd: %d detections in %.1fs", len(dets), time.time() - t0)
    return dets, mems
