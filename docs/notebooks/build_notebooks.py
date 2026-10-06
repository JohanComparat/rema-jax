"""Build, execute and check the DR11 notebooks of the documentation.

    python docs/notebooks/build_notebooks.py [blind] [pipeline] [scan]   # write (no outputs)
    python docs/notebooks/build_notebooks.py --execute [names]           # write and execute
    python docs/notebooks/build_notebooks.py --check                     # check executed ones

Execution needs the DR11 data and the ``rema`` Jupyter kernel
(``python -m ipykernel install --user --name rema``); run it on a GPU with
``JAX_PLATFORMS=cuda``. The order matters: ``pipeline`` compares its merged catalogue with the
one-region catalogue of ``blind`` (same sweeps, same calibration). Paths come from the
environment variables REMA_DR11_DIR (or LEGACYSURVEY_DIR), REMA_PRODUCTS, REMA_CALIB and
REMA_WORK; without them the DR11 data system at CC-IN2P3 is used when /sps is mounted: the sweeps,
the randoms, and the calibration and merged catalogues of the DR11 south production run next to
them (defaults below). The markdown avoids run numbers; the cells print them.
"""

import argparse
import sys
import textwrap
import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORDER = ("blind", "pipeline", "scan")


def md(text):
    return new_markdown_cell(textwrap.dedent(text).strip("\n"))


def code(text):
    return new_code_cell(textwrap.dedent(text).strip("\n"))


def notebook(cells):
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "rema", "display_name": "rema", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


# --------------------------------------------------------------------------- shared code
SETUP_COMMON = '''
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # before importing jax

import dataclasses
import logging
import re
import sys
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from matplotlib.colors import LinearSegmentedColormap

# Persistent compilation cache, shared with the rema command line.
jax.config.update("jax_compilation_cache_dir",
                  os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.path.expanduser("~/.cache/rema/jax"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

# Paths: the environment variables, else the DR11 data system at CC-IN2P3 when /sps is mounted
# (the Jupyter kernels there do not read ~/.bashrc), else a local copy with the same layout.
CC = Path("/sps/lsst/datasets/desi/legacysurveys")
LS_DIR = Path(os.environ.get("LEGACYSURVEY_DIR", CC if CC.exists() else Path.home() / "data" / "legacysurvey"))
DR11 = Path(os.environ.get("REMA_DR11_DIR", LS_DIR / "dr11" / "south"))
# The DR11 south production run (rema 0.2.0) next to the sweeps: two parts that meet at RA 0 and
# 240 deg and at Dec -85 deg, with one calibration.
PRODUCTS = Path(os.environ.get("REMA_PRODUCTS", DR11 / "rema"))
RUNS = [PRODUCTS / "rema_dr11_v0.2.0_ra0-240", PRODUCTS / "rema_dr11_v0.2.0_ra240-360"]
CALIB = Path(os.environ.get("REMA_CALIB", RUNS[0] / "calib" / "calib.fits"))
EXAMPLES = Path(os.environ.get("REMA_WORK", PRODUCTS / "notebooks"))   # what the notebooks write
RANDOMS = sorted((DR11 / "randoms").glob("randoms-south-1-*.fits"))     # every randoms file present


def read_all_randoms(cfg, box):
    """The randoms of every file in RANDOMS inside ``box``, and their density per deg²."""
    from rema.sky.maps import read_randoms

    parts = [read_randoms(f, cfg, box) for f in RANDOMS]
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}, cfg.mask.randoms_density * len(parts)
'''

SWEEPS_DEFAULT = '''
# At most 10 sweeps (5 x 5 deg tiles of DR11 south), not necessarily contiguous. The default is
# the 75 deg^2 strip RA 0-5, Dec -15 to 0 (three sweeps).
SWEEPS = ["sweep-000m005-005p000.fits", "sweep-000m010-005m005.fits", "sweep-000m015-005m010.fits"]
assert 1 <= len(SWEEPS) <= 10, "up to 10 sweeps (larger areas: the HPC pipeline)"
'''

STYLE = '''
class StageSummaries(logging.Filter):
    """Drop rema's per-batch progress lines; keep the stage summaries."""

    progress = re.compile(r"seeds \\(K=|round \\d+,")

    def filter(self, record):
        return not self.progress.search(record.getMessage())


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout, force=True)
for handler in logging.getLogger().handlers:
    handler.addFilter(StageSummaries())

# Figures: hairline grid, a one-hue sequential ramp, three categorical colours.
plt.rcParams.update({"figure.dpi": 100, "font.size": 9, "axes.grid": True,
                     "axes.axisbelow": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "legend.frameon": False})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"
RAMP = LinearSegmentedColormap.from_list(
    "ramp", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def outline(ax, box, **kw):
    """Draw a Box (or each box of a BoxUnion) on RA/Dec axes."""
    for b in box.boxes:
        ax.plot([b.ra_min, b.ra_max, b.ra_max, b.ra_min, b.ra_min],
                [b.dec_min, b.dec_min, b.dec_max, b.dec_max, b.dec_min], **kw)


def label_candidates(ax, x, y, p_cen, r=26):
    """Circle the centre candidates and label them with P_CEN.

    Call after the axis limits are set. Labels point away from the plot centre and are
    turned, when needed, so that they do not overlap.
    """
    ax.scatter(x, y, s=160, facecolor="none", edgecolor=ORANGE, lw=1.5, label="centre candidates")
    ax.apply_aspect()
    to_pt = 72 / ax.figure.dpi
    cx, cy = ax.transData.transform((0, 0)) * to_pt
    placed = []
    for xk, yk, pk in zip(x, y, p_cen):
        px, py = ax.transData.transform((xk, yk)) * to_pt
        base = np.arctan2(py - cy, px - cx) if np.hypot(px - cx, py - cy) > 3 else np.pi / 4
        for turn in (0, 0.6, -0.6, 1.2, -1.2, 1.8, -1.8, 2.4, -2.4, np.pi):
            dx, dy = r * np.cos(base + turn), r * np.sin(base + turn)
            if all(np.hypot(px + dx - qx, py + dy - qy) > 18 for qx, qy in placed):
                break
        placed.append((px + dx, py + dy))
        ax.annotate(f"{pk:.2f}", (xk, yk), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="center", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, shrinkA=0, shrinkB=7))


T = {}   # wall-clock time of each step [s]
'''


# --------------------------------------------------------------------------- blind notebook
def blind_cells():
    c = []
    c.append(md("""
    # Blind cluster finding on a set of DR11 sweeps, with spectroscopic post-processing

    This notebook runs rema's blind mode on a set of up to 10 DR11 south sweeps (5° × 5° tiles),
    processed as **one region**: there are no internal boundaries, and the outer edges of the set
    are handled by the footprint (apertures that leave the data get MASKFRAC > 0, and clusters
    with MASKFRAC ≥ 0.2 are dropped).

    The inputs are the sweeps with their row-matched photo-z sweeps (for Z_SPEC), the DR11
    randoms files and the DR11 griz calibration of the production run. The steps are:

    - the galaxy selection, written as one table per sweep (as on the HPC);
    - the mask and depth maps from the randoms;
    - zred for every galaxy;
    - the blind run: redMaPPer's first pass, likelihood pass and percolation, with redMaPPer's
      wcen centring;
    - spectroscopic post-processing (Clerc et al. 2016): cluster redshift and velocity dispersion from
      the members' Z_SPEC.

    Larger areas run as many regions on an HPC (`scripts/slurm/rema_dr11_blind.sh`). The
    pipeline notebook runs the same sweeps that way and compares the two catalogues, and the
    last section compares this catalogue with the DR11 south production catalogue.
    """))

    c.append(md("""
    ## Setup and parameters

    `SWEEPS` lists the sweep files, by name. At CC-IN2P3 the defaults read the data system:
    the sweeps and the randoms of DR11 south, and the calibration and catalogues of the
    production run next to them (`PRODUCTS`). Elsewhere, set the environment variables
    `REMA_DR11_DIR`, `REMA_PRODUCTS` (or `REMA_CALIB`) and `REMA_WORK`. Run products go to
    `WORK`, and the per-sweep galaxy tables to `GALDIR`, which the pipeline notebook shares.
    """))

    c.append(code(SETUP_COMMON + SWEEPS_DEFAULT + '''
import rema
from rema.calibration import Calibration
from rema.config import RemaConfig
from rema.io.legacy import SURVEYS, default_mag_max, ingest_sweeps, read_galaxies
from rema.io.tables import write_catalog
from rema.modes import specpost
from rema.modes.blind import run_blind
from rema.modes.common import Region
from rema.pipeline import file_sha1
from rema.sky.maps import build_footprint, read_randoms
from rema.sky.regions import sweep_union

WORK = EXAMPLES / "blind"
GALDIR = EXAMPLES / "galaxies"
WORK.mkdir(parents=True, exist_ok=True)

SKY = sweep_union(SWEEPS)          # a Box when the sweeps tile a rectangle, else a BoxUnion
BOUND = SKY.bounding()
print(f"rema {rema.__version__}, jax {jax.__version__}, devices: {jax.devices()}")
print(f"{len(SWEEPS)} sweeps, {SKY.area_deg2():.1f} deg², sky: {SKY}")
print(f"data: {DR11}; {len(RANDOMS)} randoms files; calibration: {CALIB}")
'''))

    c.append(code(STYLE))

    c.append(md("""
    ## The calibration

    The calibration file holds the red-sequence model, the zred correction, the χ² and zred
    backgrounds, the z_λ correction (with the z → zred_uncorr mapping of LNCGLIKE), the wcen
    centring model and the configuration it was made with. Its primary header records how it was
    made:

    - NCLUSTER: the spectroscopically seeded clusters with λ ≥ 5;
    - NSPECGAL: the spectroscopic galaxies they came from;
    - ZLNMAD: the NMAD of their z_λ against the seeds' redshifts, before the z_λ correction;
    - NWCEN: the training clusters of wcen.

    The configuration read from the file drives every step below. The default calibration is
    the one of the DR11 south production run, fitted on RA 160–180° and 190–210°,
    Dec −10° to 10°. When the sweeps lie in that area, the comparison of z_λ with spectroscopic
    redshifts below is not independent.
    """))

    c.append(code('''
cal = Calibration.read(CALIB)
cfg = cal.config

print(CALIB.name)
print("  " + ", ".join(f"{k} {cal.meta[k]}" for k in ("REMAVER", "BANDS", "REFBAND", "CHI2MODE",
                                                     "FLXFLOOR", "MSTAR")))
print(f"  NCLUSTER {cal.meta['NCLUSTER']}, NSPECGAL {cal.meta['NSPECGAL']}, "
      f"ZLNMAD {cal.meta['ZLNMAD']:.4f}, NWCEN {cal.meta['NWCEN']}")
w = cal.wcen
print(f"wcen: central magnitude m* {w['DELTA0']:+.3f} {w['DELTA1']:+.3f} ln(λ/{w['PIVOT']:.0f}), "
      f"σ_m {w['SIGMA_M']:.3f}")
print(f"z_lambda correction with a z -> zred_uncorr mapping: {cal.zlcorr.zred_uncorr is not None}")
print(f"model.zrange {cfg.model.zrange}, model.chisq_mode {cfg.model.chisq_mode!r}, "
      f"centering.method {cfg.centering.method!r}, survey.ebv_max {cfg.survey.ebv_max}")
'''))

    c.append(md("""
    ## Galaxies

    `ingest_sweeps` writes one compact table per sweep, named like the sweep. It is the stage the
    HPC pipeline runs once for all 1,600 sweeps, and existing tables made with the same survey
    configuration are kept. The selection is:

    - clean MASKBITS, with large galaxies kept;
    - TYPE ≠ PSF;
    - NOBS ≥ 1 in g, r, i and z;
    - z-band S/N ≥ 5;
    - a z-band magnitude below m*(z = 1) + 2.5.

    Fluxes are dereddened, and Z_SPEC comes from the row-matched photo-z sweep.
    `read_galaxies` then assembles the galaxies of any box, or union of boxes, from the tables.
    """))

    c.append(code('''
t0 = time.perf_counter()
ingest_sweeps([DR11 / "sweep" / "11.0" / s for s in SWEEPS], GALDIR, cfg)
gal = read_galaxies(GALDIR, SKY, cfg)
T["ingest"] = time.perf_counter() - t0

n = gal["ID"].size
spec = gal["ZSPEC"] > 0
src, nsrc = np.unique(gal["ZSPEC_SRC"][spec], return_counts=True)
order = np.argsort(-nsrc)
print(f"{n:,} galaxies ({n / SKY.area_deg2():,.0f} per deg²) down to {default_mag_max(cfg):.2f} "
      f"mag in z, in {T['ingest']:.0f} s")
names = [SURVEYS[s - 1] if 0 < s <= len(SURVEYS) else "other" for s in src[order]]
print(f"{spec.sum():,} with ZSPEC > 0: " + ", ".join(f"{name} {k:,}"
                                                     for name, k in zip(names, nsrc[order])))
'''))

    c.append(md("""
    ## Footprint and depth

    The footprint plays the role of redMaPPer's mask and depth maps. It is built from the DR11
    randoms, at 2,500 per deg² per file, on NESTED HEALPix pixels of nside 1024 (3.4′):

    - FRACGOOD is the fraction of randoms that pass the galaxies' MASKBITS, NOBS and E(B−V)
      cuts, times the fraction of the pixel inside the sky of the run;
    - SIGF_<band> is the median dereddened 1σ flux error of a galaxy, from GALDEPTH.

    Together they set MASKFRAC and the depth correction (SCALEVAL) of λ. A pixel holds about 8
    randoms per file, so FRACGOOD is coarse per pixel, but λ uses it integrated over the cluster
    aperture.

    Every randoms file present is used, as in the production run, and the density scales with
    their number. The rows of a randoms file are in random sky order, so `read_randoms` scans the
    whole 23 GB file. The HPC pipeline indexes every file once (`rema randoms-index`), as the
    pipeline notebook shows.
    """))

    c.append(code('''
t0 = time.perf_counter()
rnd, density = read_all_randoms(cfg, BOUND)
fp = build_footprint(rnd, cfg, box=SKY, density=density)
fp.write(WORK / "footprint.fits")
T["footprint"] = time.perf_counter() - t0

v = fp.fine.values
good = v["FRACGOOD"] > 0.5
depth = {b.lower(): np.median(22.5 - 2.5 * np.log10(5 * v[f"SIGF_{b}"][good])) for b in fp.bands}
print(f"{rnd['RA'].size:,} randoms from {len(RANDOMS)} files in {fp.fine.pixels.size:,} pixels (nside {fp.nside}), "
      f"in {T['footprint']:.0f} s")
print(f"unmasked area {fp.area_deg2():.2f} deg² of {SKY.area_deg2():.2f} deg²")
print("median 5σ depth: " + ", ".join(f"{b} {d:.2f}" for b, d in depth.items()))
'''))

    c.append(code('''
# FRACGOOD and z-band 5σ depth, looked up on a 0.02° grid over the bounding box.
step = 0.02
ra_g, dec_g = np.meshgrid(np.arange(BOUND.ra_min, BOUND.ra_max, step) + step / 2,
                          np.arange(BOUND.dec_min, BOUND.dec_max, step) + step / 2)
frac = fp.fine.lookup_np(ra_g.ravel() % 360, dec_g.ravel(), "FRACGOOD").reshape(ra_g.shape)
sig_z = fp.fine.lookup_np(ra_g.ravel() % 360, dec_g.ravel(), "SIGF_Z").reshape(ra_g.shape)
with np.errstate(divide="ignore", invalid="ignore"):
    depth_z = np.where(frac > 0, 22.5 - 2.5 * np.log10(5 * sig_z), np.nan)

aspect = (BOUND.ra_max - BOUND.ra_min) / (BOUND.dec_max - BOUND.dec_min)
fig, axes = plt.subplots(1, 2, figsize=(9.5, min(6.5, max(3.5, 4.6 / aspect))), constrained_layout=True)
extent = (BOUND.ra_min, BOUND.ra_max, BOUND.dec_min, BOUND.dec_max)
panels = ((frac, "FRACGOOD", (0, 1)), (depth_z, "z-band 5σ depth [AB]", np.nanpercentile(depth_z, [2, 98])))
for ax, (img, label, (lo, hi)) in zip(axes, panels):
    im = ax.imshow(img, origin="lower", extent=extent, cmap=RAMP, vmin=lo, vmax=hi,
                   interpolation="nearest", aspect="auto")
    outline(ax, SKY, color=ORANGE, lw=1)
    ax.invert_xaxis()
    ax.grid(False)
    ax.set(xlabel="RA [deg]", ylabel="Dec [deg]")
    fig.colorbar(im, ax=ax, shrink=0.85, label=label)
axes[0].set_title("fraction of good randoms (sweeps in orange)", fontsize=9)
axes[1].set_title("depth of a canonical galaxy, from GALDEPTH", fontsize=9)
plt.show()
'''))

    c.append(md("""
    ## zred and the run context

    `Region.build` computes zred for every galaxy, with the calibrated zred correction. It also
    sets up:

    - the filter: red sequence, χ² background, m* and cosmology;
    - the footprint lookups;
    - the z_λ correction.

    The calibration holds a wcen model, so `centering.method: auto` resolves to wcen, which is
    redMaPPer's CenteringWcenZred with the calibration's zred background.
    """))

    c.append(code('''
t0 = time.perf_counter()
reg = Region.build(gal, cal.rs, cfg, footprint=fp, zredcorr=cal.zredcorr, bkg=cal.bkg,
                   zlcorr=cal.zlcorr, zbkg=cal.zbkg, wcen_params=cal.wcen)
T["zred"] = time.perf_counter() - t0
method = reg.centering_method()
print(f"zred for {reg.gal['ZRED'].size:,} galaxies in {T['zred']:.1f} s; centring: {method}")
'''))

    c.append(md("""
    ## Blind run

    `run_blind` follows redMaPPer's run mode:

    1. **Seeds:** galaxies with zred in the model range (0.05–0.90), zred χ² < 20 and
       m < m*(zred) + 1.75.
    2. **First pass:** z_λ and λ in a 0.5 h⁻¹Mpc aperture centred on each seed; λ ≥ 3 is
       kept.
    3. **Likelihood pass:** λ with r0 = 1 h⁻¹Mpc and β = 0.2, and
       LNLIKE = LNLAMLIKE + LNCGLIKE. Candidates with λ < 3, or with fewer than 3 members
       besides the seed, are dropped.
    4. **Percolation**, in order of decreasing LNLIKE, with wcen centring and redMaPPer's
       rejections: λ/S < 3 at the seed or at the new centre, z_λ failure, or no free central
       galaxy.
    5. **Consolidation:** λ/S ≥ 3 and MASKFRAC < 0.2. With `own=None` every cluster is kept;
       a region of the HPC pipeline keeps those centred in its own box.

    The percolation is exact: it gives the result of the sequential loop, computed in batches
    of candidates that do not depend on each other. With a checkpoint directory, a rerun resumes
    after the first pass or the likelihood pass.

    The log keeps the stage summaries, with the time since the start of the run in parentheses.
    In the percolation summary, a skipped candidate is one whose seed was already claimed
    (pfree < 0.5). The dropped candidates are counted by rejection; "seed" means λ/S < 3 at the
    seed.
    """))

    c.append(code('''
t0 = time.perf_counter()
info = {}
cat, mem = run_blind(reg, own=None, region_id=0, checkpoint=WORK / "checkpoint", info=info)
T["blind"] = time.perf_counter() - t0
print(f"{cat['LAMBDA'].size:,} clusters, {mem['ID'].size:,} member rows, in {T['blind']:.0f} s")
print(f"{info['n_seeds']:,} seeds, {info['n_firstpass']:,} first-pass and {info['n_candidates']:,} "
      f"likelihood candidates, {info['n_dependencies']:,} percolation dependencies; "
      f"peak memory {info['peak_rss_gb']:.1f} GB")
'''))

    c.append(md("""
    ## Spectroscopic post-processing

    `specpost.process` is the velocity clipping of Clerc et al. (2016):

    - it applies to clusters with at least 3 members with Z_SPEC;
    - the cluster redshift is the biweight location of the member redshifts;
    - members are clipped at 3σ in rest-frame velocity for 20 iterations, with the gapper σ
      below 15 members and the biweight scale above;
    - the errors (SPEC_Z_BOOT, SPEC_ZERR_BOOT, VDISP_ERR) come from 64 bootstrap resamples;
    - BEST_Z is SPEC_Z_BOOT when available and not flagged, else the central galaxy's Z_SPEC,
      else z_λ.

    `write_catalog` then writes the catalogue as `rema blind` does, with CLUSTERS, MEMBERS and
    CONFIG HDUs. The header records the sweeps and the calibration, and the pipeline notebook
    checks both before it compares catalogues.
    """))

    c.append(code('''
t0 = time.perf_counter()
cat, mem = specpost.process(cat, mem, **dataclasses.asdict(cfg.spec))
T["specpost"] = time.perf_counter() - t0
types, ntypes = np.unique(cat["BEST_Z_TYPE"], return_counts=True)
print(f"specpost in {T['specpost']:.1f} s: {np.isfinite(cat['SPEC_Z_BOOT']).sum()} clusters "
      f"with SPEC_Z_BOOT; BEST_Z_TYPE " + ", ".join(f"{t} {k}" for t, k in zip(types, ntypes)))

path = write_catalog(WORK / "clusters.fits", cat, mem, cfg,
                     {"MODE": "blind", "CENTRING": method, "SWEEPS": ",".join(sorted(SWEEPS))[:1000],
                      "CALIB": CALIB.name, "CALSHA1": file_sha1(CALIB), "TBLIND": round(T["blind"], 1)})
print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
'''))

    c.append(md("""
    ## The catalogue

    The cells below show:

    - the counts above a few richness thresholds and the cumulative richness function;
    - the z_λ distribution, which includes the calibrated z_λ correction (Z_LAMBDA_RAW does
      not);
    - for comparison, the redshifts of the spectroscopic galaxies.
    """))

    c.append(code('''
lam, zl = cat["LAMBDA"], cat["Z_LAMBDA"]
area = fp.area_deg2()
print(f"{lam.size:,} clusters over {area:.1f} deg² unmasked (λ ≥ {cfg.richness.minlambda:.0f}, "
      f"λ/S ≥ {cfg.richness.minlambda:.0f}, MASKFRAC < {cfg.mask.max_maskfrac})")
for lmin in (5, 10, 20):
    print(f"  λ ≥ {lmin:2d}: {np.sum(lam >= lmin):6,d}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
srt = np.sort(lam)[::-1]
ax1.step(srt, np.arange(1, srt.size + 1) / area, where="post", color=BLUE, lw=1.5)
ax1.set(xscale="log", yscale="log", xlabel=r"$\\lambda$", ylabel=r"$N(>\\lambda)$ per deg²")
bins = np.arange(0.05, 0.9501, 0.025)
for lmin, color in zip((5, 10, 20), ("#86b6ef", "#2a78d6", "#0d366b")):
    ax2.hist(zl[lam >= lmin], bins=bins, histtype="step", lw=1.5, color=color,
             label=rf"$\\lambda \\geq {lmin}$")
ax2.set(xlabel=r"$z_\\lambda$", ylabel=r"clusters per $\\Delta z = 0.025$", yscale="log")
ax2.legend(loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.8)
zspec = gal["ZSPEC"][gal["ZSPEC"] > 0]
ax3.hist(zspec, bins=bins, color=GREY, edgecolor="white", lw=0.5)
ax3.set(xlabel="ZSPEC", ylabel=r"galaxies per $\\Delta z = 0.025$",
        title=f"{zspec.size:,} galaxies with ZSPEC", xlim=ax2.get_xlim())
ax3.title.set_fontsize(9)
plt.show()
'''))

    c.append(md("""
    Spikes in the z_λ histogram that line up with spikes of the spectroscopic redshifts are
    large-scale structure. At z_λ > 0.75, clusters with λ < 10 are mostly noise, as a null test
    shows; see section 14 of the design notes.
    """))

    c.append(code('''
# The λ ≥ 20 clusters, sized by λ and coloured by z_λ, over the sweeps.
ACT = SkyCoord(3.23333, -8.95419, unit="deg")      # ACT-CL J0012.9-0857, in the default strip
rich = lam >= 20
fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
sc = ax.scatter(cat["RA"][rich], cat["DEC"][rich], s=1.5 * lam[rich], c=zl[rich], cmap=RAMP,
                vmin=0.05, vmax=0.9, edgecolor="white", lw=0.8, zorder=3)
for size in (20, 50, 100):
    ax.scatter([], [], s=1.5 * size, color=GREY, edgecolor="white", label=rf"$\\lambda = {size}$")
if SKY.contains(ACT.ra.deg, ACT.dec.deg):
    ax.scatter(ACT.ra.deg, ACT.dec.deg, s=250, facecolor="none", edgecolor=ORANGE, lw=1.5,
               label="ACT-CL J0012.9-0857", zorder=4)
outline(ax, SKY, color=GREY, lw=1)
ax.set(xlim=(BOUND.ra_max + 0.2, BOUND.ra_min - 0.2), ylim=(BOUND.dec_min - 0.2, BOUND.dec_max + 0.2),
       xlabel="RA [deg]", ylabel="Dec [deg]", title=rf"{rich.sum()} clusters with $\\lambda \\geq 20$")
ax.set_aspect(1 / np.cos(np.radians(0.5 * (BOUND.dec_min + BOUND.dec_max))))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4)
fig.colorbar(sc, cax=ax.inset_axes([1.03, 0.1, 0.035, 0.8]), label=r"$z_\\lambda$")
plt.show()
'''))

    c.append(md("""
    ### z_λ against the spectroscopic cluster redshifts

    The comparison uses clusters with λ ≥ 20 and at least 3 spectroscopic members kept by the
    clipping (N_MEMBERS ≥ 3).

    - Δz = Z_LAMBDA − SPEC_Z_BOOT.
    - The bias is the median of Δz/(1 + z).
    - The NMAD is 1.4826 times the median absolute deviation from it.
    """))

    c.append(code('''
sel = (lam >= 20) & (cat["N_MEMBERS"] >= 3) & np.isfinite(cat["SPEC_Z_BOOT"])
zs, zse = cat["SPEC_Z_BOOT"][sel], cat["SPEC_ZERR_BOOT"][sel]
dz = (zl[sel] - zs) / (1 + zs)
bias = np.median(dz)
nmad = 1.4826 * np.median(np.abs(dz - bias))
print(f"{sel.sum()} clusters: bias {bias:+.4f}, NMAD {nmad:.4f}, "
      f"|Δz/(1 + z)| > 0.05: {np.sum(np.abs(dz) > 0.05)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
ax1.plot([0, 1], [0, 1], color=GREY, lw=1)
ax1.errorbar(zs, zl[sel], xerr=zse, yerr=cat["Z_LAMBDA_E"][sel], fmt="o", ms=4, color=BLUE,
             mec="white", mew=0.8, elinewidth=0.8)
lim = (0.0, max(zs.max(), zl[sel].max()) + 0.05)
ax1.set(xlim=lim, ylim=lim, xlabel="SPEC_Z_BOOT", ylabel="Z_LAMBDA")
ax1.set_aspect("equal")
ax2.axhspan(bias - nmad, bias + nmad, color=BLUE, alpha=0.1, lw=0, label="bias ± NMAD")
ax2.axhline(0, color=GREY, lw=1)
ax2.errorbar(zs, dz, yerr=cat["Z_LAMBDA_E"][sel] / (1 + zs), fmt="o", ms=4, color=BLUE,
             mec="white", mew=0.8, elinewidth=0.8)
ax2.set(xlim=lim, ylim=(-0.045, 0.045), xlabel="SPEC_Z_BOOT", ylabel=r"$\\Delta z / (1 + z)$")
ax2.legend(loc="upper left")
plt.show()
'''))

    c.append(md("""
    ### The richest cluster

    The cell shows its key columns and its members, coloured by PMEM with the symbol size set
    by z-band magnitude, and its five centre candidates with their P_CEN. The circle has radius
    R_LAMBDA.
    """))

    c.append(code('''
i = int(np.argmax(lam))
cl = {k: v[i] for k, v in cat.items()}
print(f"MEM_MATCH_ID {cl['MEM_MATCH_ID']}  RA {cl['RA']:.5f}  Dec {cl['DEC']:.5f}")
print(f"LAMBDA {cl['LAMBDA']:.1f} ± {cl['LAMBDA_E']:.1f}  Z_LAMBDA {cl['Z_LAMBDA']:.4f} ± "
      f"{cl['Z_LAMBDA_E']:.4f}  R_LAMBDA {cl['R_LAMBDA']:.3f} h⁻¹Mpc  SCALEVAL "
      f"{cl['SCALEVAL']:.3f}  MASKFRAC {cl['MASKFRAC']:.3f}")
print("P_CEN " + ", ".join(f"{p:.3f}" for p in cl["P_CEN"])
      + f"  NCENT_GOOD {cl['NCENT_GOOD']}  W {cl['W']:.3f}")
print(f"NSPEC {cl['NSPEC']}  N_MEMBERS {cl['N_MEMBERS']}  SPEC_Z_BOOT {cl['SPEC_Z_BOOT']:.4f} ± "
      f"{cl['SPEC_ZERR_BOOT']:.4f}  VDISP {cl['VDISP']:.0f} ± {cl['VDISP_ERR']:.0f} km/s "
      f"({cl['VDISP_TYPE']})  BEST_Z {cl['BEST_Z']:.4f} ({cl['BEST_Z_TYPE']})")

m = mem["MEM_MATCH_ID"] == cl["MEM_MATCH_ID"]
cosd = np.cos(np.radians(cl["DEC"]))
dx = (mem["RA"][m] - cl["RA"]) * cosd * 60
dy = (mem["DEC"][m] - cl["DEC"]) * 60
pm, mag = mem["PMEM"][m], mem["REFMAG"][m]
o = np.argsort(pm)
r_lam = cl["R_LAMBDA"] / reg.mpc_per_deg_np(cl["Z_LAMBDA"]) * 60

fig, ax = plt.subplots(figsize=(6, 5.2), constrained_layout=True)
sc = ax.scatter(dx[o], dy[o], c=pm[o], cmap=RAMP, vmin=0, vmax=1,
                s=8 + 60 * 10 ** (-0.4 * (mag[o] - mag.min())), edgecolor="white", lw=0.5)
ax.add_patch(plt.Circle((0, 0), r_lam, fill=False, color=GREY, lw=1, label="R_LAMBDA"))
ok = cl["ID_CENT"] >= 0
cx = (cl["RA_CENT"][ok] - cl["RA"]) * cosd * 60
cy = (cl["DEC_CENT"][ok] - cl["DEC"]) * 60
ax.set(xlim=(1.25 * r_lam, -1.25 * r_lam), ylim=(-1.25 * r_lam, 1.25 * r_lam),
       xlabel="ΔRA cos Dec [arcmin]", ylabel="ΔDec [arcmin]",
       title=rf"$\\lambda$ = {cl['LAMBDA']:.0f}, $z_\\lambda$ = {cl['Z_LAMBDA']:.3f}: {m.sum()} members")
ax.set_aspect("equal")
label_candidates(ax, cx, cy, cl["P_CEN"][ok])
ax.scatter(0, 0, marker="*", s=110, color=ORANGE, edgecolor="white", lw=0.5, zorder=4,
           label="centre")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
fig.colorbar(sc, cax=ax.inset_axes([1.03, 0.1, 0.035, 0.8]), label="PMEM")
plt.show()
'''))

    c.append(md("""
    ## Cross-check with ACT

    When the sweeps contain it, the SZ cluster ACT-CL J0012.9-0857 (Hilton et al. 2021,
    z = 0.352) serves as an external check. The cell lists the rema clusters centred within 3′
    of its SZ position, and the centring of the richest of them. The scan notebook measures the
    same cluster from its position alone.
    """))

    c.append(code('''
if SKY.contains(ACT.ra.deg, ACT.dec.deg):
    Z_ACT = 0.352
    sep = SkyCoord(cat["RA"], cat["DEC"], unit="deg").separation(ACT).arcmin
    near = np.flatnonzero(sep < 3)
    near = near[np.argsort(sep[near])]
    cols = ("MEM_MATCH_ID", "LAMBDA", "Z_LAMBDA", "Z_LAMBDA_E", "SPEC_Z_BOOT", "N_MEMBERS", "VDISP")
    tab = Table({"SEP_ARCMIN": sep[near], **{k: cat[k][near] for k in cols}})
    for col, fmt in (("SEP_ARCMIN", ".2f"), ("LAMBDA", ".1f"), ("Z_LAMBDA", ".4f"),
                     ("Z_LAMBDA_E", ".4f"), ("SPEC_Z_BOOT", ".4f"), ("VDISP", ".0f")):
        tab[col].format = fmt
    tab.pprint(max_width=-1)
    j = near[np.argmax(lam[near])]
    off = sep[j] / 60 * reg.mpc_per_deg_np(zl[j])
    ok = cat["ID_CENT"][j] >= 0
    cand = SkyCoord(cat["RA_CENT"][j][ok], cat["DEC_CENT"][j][ok], unit="deg").separation(ACT).arcmin
    print(f"\\nrichest within 3′: λ {lam[j]:.1f}, centred {sep[j]:.2f}′ = {off:.2f} h⁻¹Mpc from the SZ position")
    print("centre candidates: " + ", ".join(f"{s:.2f}′ (P_CEN {p:.3f})"
                                            for s, p in zip(cand, cat["P_CEN"][j][ok])))
    print(f"(Z_LAMBDA − z_ACT)/(1 + z_ACT) = {(zl[j] - Z_ACT) / (1 + Z_ACT):+.4f}, "
          f"(SPEC_Z_BOOT − z_ACT)/(1 + z_ACT) = {(cat['SPEC_Z_BOOT'][j] - Z_ACT) / (1 + Z_ACT):+.4f}")
else:
    print("ACT-CL J0012.9-0857 is not in these sweeps")
'''))

    c.append(md("""
    For this cluster, the members' spectroscopic redshifts and z_λ agree with each other, and
    both lie about 0.01 (1 + z) below the ACT redshift. No spectroscopic redshift within 6′ is
    near the ACT value (see the scan notebook). Blind mode centres the cluster on its brightest
    member, which lies outside the 0.4 h⁻¹Mpc search radius that scan mode uses around an input
    position.
    """))

    c.append(md("""
    ## The production catalogue

    The DR11 south production run used the same calibration, configuration, galaxies and
    randoms over the whole footprint, as regions of about 100 deg² with 2° buffers. Its merged
    catalogues are on the data system next to the sweeps (`PRODUCTS`), in two parts that meet at
    RA 0° and 240° and at Dec −85°. Inside these sweeps, its clusters should be the clusters of
    this notebook. Near the edges of the sweeps they may differ: the production run also had the
    galaxies beyond them, except across the boundaries of the two parts.

    The cell matches the clusters by their central galaxy (ID_CENT[0]) and compares them as a
    function of the distance to the edge of the sweeps (of their bounding box when they do not
    tile a rectangle).
    """))

    c.append(code('''
import json

from rema.io.tables import read_catalog


def production_catalogue(sky):
    """Clusters of the production parts centred in ``sky``, with the part number (PART)."""
    parts = []
    for k, run in enumerate(RUNS):
        f = run / "clusters_dr11.fits"
        if not f.exists():
            print(f"{run.name}: not available yet")
            continue
        pc, _, _ = read_catalog(f, members=False)
        qa = json.loads((run / "clusters_dr11_qa.json").read_text())
        keep = sky.contains(pc["RA"], pc["DEC"])
        parts.append({**{c: np.asarray(v)[keep] for c, v in pc.items()}, "PART": np.full(int(keep.sum()), k)})
        same = qa.get("calibrations") == [file_sha1(CALIB)]
        print(f"{run.name}: {len(pc['RA']):,} clusters, {int(keep.sum()):,} in these sweeps; "
              f"{'same' if same else 'another'} calibration")
    return {c: np.concatenate([q[c] for q in parts]) for c in parts[0]} if parts else None


prod = production_catalogue(SKY)
if prod is not None and len(prod["RA"]):
    where = {k: i for i, k in enumerate(np.asarray(prod["ID_CENT"])[:, 0])}
    j = np.array([where.get(k, -1) for k in np.asarray(cat["ID_CENT"])[:, 0]])
    b = SKY.bounding()
    cosd = np.cos(np.radians(cat["DEC"]))
    edge = np.minimum.reduce([cat["DEC"] - b.dec_min, b.dec_max - cat["DEC"],
                              (cat["RA"] - b.ra_min) * cosd, (b.ra_max - cat["RA"]) * cosd])
    rows = []
    for lo, hi in ((0.0, 1.0), (1.0, 2.3), (2.3, 99.0)):
        s = (lam >= 20) & (edge >= lo) & (edge < hi)
        m = s & (j >= 0)
        rel = np.abs(prod["LAMBDA"][j[m]] / lam[m] - 1)
        dz = np.abs(prod["Z_LAMBDA"][j[m]] - zl[m])
        rows.append((f"{lo:.1f}-{hi:.1f}", int(s.sum()), m.sum() / max(s.sum(), 1),
                     np.median(rel) if m.any() else np.nan, dz.max() if m.any() else np.nan))
    tab = Table(rows=rows, names=("EDGE_DEG", "N_LAMBDA_GE_20", "SAME_CENTRE", "MEDIAN_DLAMBDA_REL", "MAX_DZ"))
    for col, fmt in (("SAME_CENTRE", ".1%"), ("MEDIAN_DLAMBDA_REL", ".2e"), ("MAX_DZ", ".1e")):
        tab[col].format = fmt
    tab.pprint(max_width=-1)
    lp = np.asarray(prod["LAMBDA"])
    print(f"production clusters with λ ≥ 20 in these sweeps: {int(np.sum(lp >= 20))}, "
          f"this notebook: {int(np.sum(lam >= 20))}")
'''))

    c.append(md("""
    ## Runtimes
    """))

    c.append(code('''
for step, sec in T.items():
    print(f"{step:10s} {sec:7.1f} s")
print(f"{'total':10s} {sum(T.values()):7.1f} s on {jax.devices()[0].device_kind}")
'''))

    c.append(md("""
    ## The same run from the command line

    Repeat `--box` once per box of the sky; the sweeps here tile a single rectangle.

    ```bash
    rema ingest DR11/sweep/11.0/sweep-000m005-005p000.fits DR11/sweep/11.0/sweep-000m010-005m005.fits \\
         DR11/sweep/11.0/sweep-000m015-005m010.fits --outdir galaxies
    rema maps DR11/randoms/randoms-south-1-*.fits --box 0 5 -15 0 --out footprint.fits
    rema blind --galaxies galaxies --box 0 5 -15 0 --calib CALIB --footprint footprint.fits \\
         --checkpoint checkpoint --specpost --out clusters.fits
    ```

    - **Configuration:** `rema blind` uses the configuration stored in the calibration unless
      `--config` is given. `rema ingest` and `rema maps` have none to read, so pass them
      `--config` when the calibration was made with a non-default configuration (write it with
      `cal.config.to_yaml("config.yaml")`).
    - **GPU:** set `JAX_PLATFORMS=cuda`.
    """))
    return c


# --------------------------------------------------------------------------- pipeline notebook
def pipeline_cells():
    c = []
    c.append(md("""
    # The HPC pipeline on a set of sweeps, and the region boundaries

    The full DR11 south blind run cuts the sky into about 300–450 regions of about 100 deg²,
    one SLURM array task each (`scripts/slurm/rema_dr11_blind.sh`; see the HPC page of the
    documentation). This notebook runs the same stages, with the same task script
    (`scripts/slurm/rema_task.sh`), on the sweeps of the blind notebook, cut into one region
    per sweep. It then measures what the region boundaries change, by comparing the merged
    catalogue with the blind notebook's one-region catalogue of the same sweeps and calibration.

    **How a cluster near a boundary is handled:**

    - A region reads the galaxies and randoms of its *data box*, which is its *own box* grown by
      a 2° buffer, and keeps the clusters whose final centre lies in its own box.
    - The own boxes tile the sky without overlap, so the merged catalogue has no duplicates, and
      `MEM_MATCH_ID = region << 32 | rank` is unique.
    - Sweeps are only I/O units. A cluster across two sweeps is seen whole by the region that
      owns its centre.
    - The 2° buffer covers what changes a cluster directly: its aperture, plus a higher-ranked
      neighbour's mask radius and seed offset, at most about 1.6° at z = 0.05.
    - Percolation chains that run beyond the buffer can still change a cluster, at second order.

    Run the blind notebook first. With three sweeps the regions take about 1.5 times the
    one-region time, because the buffers are read twice.
    """))

    c.append(md("""
    ## Setup and parameters

    `SWEEPS` must be the blind notebook's. `OUTDIR` is laid out like an HPC run directory, and
    its `galaxies/` points to the per-sweep tables that the blind notebook wrote. The stages run
    the task script with the environment variables the driver would set. `AREA_BOX` restricts
    the run to the sweeps, `TARGET_AREA=25` with `BAND_HEIGHT=5` gives one region per sweep, and
    `NRAND` is the number of randoms files present, as in the production run.
    """))

    c.append(code(SETUP_COMMON + SWEEPS_DEFAULT + '''
import json
import subprocess

import rema
from rema.calibration import Calibration
from rema.io.tables import read_catalog, read_header
from rema.pipeline import file_sha1, plan_boxes, read_plan, region_path, required_buffer
from rema.sky.regions import sweep_union

TASK = Path(rema.__file__).resolve().parents[1] / "scripts" / "slurm" / "rema_task.sh"
OUTDIR = EXAMPLES / "pipeline"
OUTDIR.mkdir(parents=True, exist_ok=True)
GALDIR = EXAMPLES / "galaxies"
if not (OUTDIR / "galaxies").exists():
    (OUTDIR / "galaxies").symlink_to(GALDIR)

SKY = sweep_union(SWEEPS)
BOUND = SKY.bounding()
area_box = ";".join(f"{b.ra_min:g} {b.ra_max:g} {b.dec_min:g} {b.dec_max:g}" for b in SKY.boxes)
# Products stay in OUTDIR, never in the production folders a sourced ccin2p3.env may point to.
ENV = {**os.environ, "DR11": str(DR11), "OUTDIR": str(OUTDIR), "CALIB": str(CALIB),
       "CLUSTERS_DIR": str(OUTDIR), "MEMBERS_DIR": str(OUTDIR),
       "AREA_BOX": area_box, "TARGET_AREA": "25", "BAND_HEIGHT": "5", "BUFFER": "2",
       "NRAND": str(len(RANDOMS)), "CHUNK": "100000", "DEVICE": "gpu" if jax.default_backend() == "gpu" else "cpu",
       "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"}


def stage(*args):
    """Run one stage of the task script, as a SLURM array task would; returns the seconds."""
    t0 = time.perf_counter()
    print("$ rema_task.sh " + " ".join(map(str, args)))
    r = subprocess.run(["bash", str(TASK), *map(str, args)], env=ENV, capture_output=True, text=True)
    lines = [l for l in r.stderr.splitlines() if " rema" in l and "WARNING" not in l
             and not re.search(r"seeds \\(K=|round \\d+,", l)]
    print("\\n".join("  " + l[9:] for l in lines[-6:]))
    if r.returncode:
        print(r.stderr[-3000:])
        raise RuntimeError(f"stage {args} failed")
    return time.perf_counter() - t0


print(f"rema {rema.__version__}; {len(SWEEPS)} sweeps, {SKY.area_deg2():.1f} deg²; AREA_BOX = {area_box}")
print(f"device for the regions: {ENV['DEVICE']}")
'''))

    c.append(code(STYLE))

    c.append(md("""
    ## Galaxies and randoms, once

    On the HPC these two stages are arrays:

    - **ingest:** one task per chunk of 20 sweeps, which reads the 1.8 TB of sweeps once;
    - **randoms:** one task per randoms file, which reads each 23 GB file once and writes a
      copy of the needed columns sorted by HEALPix pixel.

    Afterwards a region's footprint reads only the rows of its data box. Here the per-sweep
    tables already exist, so the ingest stage keeps them, and the randoms index covers only the
    sweeps (`AREA_BOX`).
    """))

    c.append(code('''
T["ingest"] = stage("ingest", 0)
t0 = time.perf_counter()
for k in range(len(RANDOMS)):
    stage("randoms", k)
T["randoms"] = time.perf_counter() - t0
idx = OUTDIR / "randoms_index"
rows = [json.loads((idx / f"randoms-south-1-{k}.json").read_text())["nrows"] for k in range(len(RANDOMS))]
print(f"randoms index: {len(rows)} files, {sum(rows):,} rows")
'''))

    c.append(md("""
    ## The region plan

    `rema regions` plans the regions from the galaxy counts in the per-sweep table headers.

    - Dec bands follow the sweep rows: 10° bands in production, 5° here. The polar cap
      (|Dec| > 85°) is one full-RA ring.
    - In each band, own boxes are runs of whole sweeps, starting after the largest RA gap so
      that RA 0/360 is handled. Each run holds about `TARGET_AREA` deg².
    - Regions above `MAX_GAL` galaxies, or `MAX_PAIRS` estimated percolation pairs, are split.
    - Regions are numbered by decreasing estimated cost.

    Each data box is the own box grown by the buffer, cut here to the sweeps (`AREA_BOX`). The
    planner checks the buffer against `required_buffer`. The *first-order* reach is the
    aperture plus a neighbour's mask radius and seed offset. The *exact* reach is percolation's
    full dependency radius.
    """))

    c.append(code('''
T["plan"] = stage("plan")
plan, meta = read_plan(OUTDIR / "regions.fits")
rids = [int(r) for r in plan["REGION_ID"]]
print(f"{len(rids)} regions, PLANHASH {meta['PLANHASH']}")
for r in rids:
    own, data = plan_boxes(plan, r, meta)
    print(f"  region {r}: own {own.as_tuple()}, data Dec {data.bounding().dec_min:.0f} to "
          f"{data.bounding().dec_max:.0f}, {data.area_deg2():.0f} deg², "
          f"{plan['NGAL_DATA'][rids.index(r)] / 1e6:.2f} M galaxies")
need = {lam: required_buffer(Calibration.read(CALIB).config, lam) for lam in (30, 100, 300)}
for lam, d in need.items():
    print(f"  λ = {lam:3d} at z = 0.05: first-order reach {d['first_order']:.2f}°, exact {d['exact']:.2f}°")

fig, ax = plt.subplots(figsize=(5.2, 6.2), constrained_layout=True)
for k, r in enumerate(rids):
    own, data = plan_boxes(plan, r, meta)
    col = (BLUE, ORANGE, AQUA)[k % 3]
    outline(ax, data, color=col, lw=1, ls=(0, (2, 2)))
    outline(ax, own, color=col, lw=2.5)
    ax.text(0.5 * (own.ra_min + own.ra_max), 0.5 * (own.dec_min + own.dec_max), f"region {r}",
            ha="center", va="center", color=col)
outline(ax, SKY, color="black", lw=0.8)
ax.set(xlim=(BOUND.ra_max + 2.5, BOUND.ra_min - 2.5), ylim=(BOUND.dec_min - 2.5, BOUND.dec_max + 2.5),
       xlabel="RA [deg]", ylabel="Dec [deg]", title="own boxes (solid) and data boxes (dashed)")
ax.set_aspect(1 / np.cos(np.radians(0.5 * (BOUND.dec_min + BOUND.dec_max))))
plt.show()
'''))

    c.append(md("""
    ## The regions

    Each region is one task: `rema maps` builds its footprint from the randoms index, and
    `rema blind` runs with `--regions/--region-id`. On the HPC the first region runs alone to
    fill the shared JAX compilation cache, then the others run in parallel. Here they run one
    after the other.

    As with the driver, only the missing or stale regions run: those without a catalogue, or
    whose catalogue was made with another plan or calibration (`rema status`). Each catalogue's
    header records the plan, the calibration, the device and the run statistics.
    """))

    c.append(code('''
status = subprocess.run(["rema", "status", "--plan", str(OUTDIR / "regions.fits"), "--runs",
                         str(OUTDIR / "regions"), "--calib", str(CALIB)], env=ENV,
                        capture_output=True, text=True).stdout
todo = [int(i) for i in re.search(r"^todo_ids=(.*)$", status, re.M).group(1).split(",") if i]
print(status.splitlines()[0])
for r in rids:
    if r in todo:
        T[f"region {r}"] = stage("region", r)
rows = []
for r in rids:
    h = read_header(region_path(OUTDIR / "regions", r), 0)
    rows.append((r, h["NCLUSTER"], h["NGAL"], h["NCAND"], h["NDEP"], h["TTOTAL"], h["PEAKRSS"], h["DEVICE"]))
Table(rows=rows, names=("REGION", "NCLUSTER", "NGAL", "NCAND", "NDEP", "TTOTAL", "PEAKRSS", "DEVICE")).pprint(max_width=-1)
'''))

    c.append(md("""
    ## Merge and QA

    `rema merge` concatenates the region catalogues and checks them:

    - every MEM_MATCH_ID is unique;
    - every centre lies in its region's own box;
    - clusters within 1′ and |Δz| < 0.02(1 + z) of each other in different regions are listed;
    - no member lacks a cluster;
    - galaxies whose total membership exceeds 1 are counted, within one region and across
      regions;
    - all regions used the same rema version, calibration and configuration;
    - the cluster density is compared with uniform points, against the distance to internal
      edges.
    """))

    c.append(code('''
T["merge"] = stage("merge")
qa = json.loads((OUTDIR / "clusters_dr11_qa.json").read_text())
for k in ("n_regions", "n_done", "n_clusters", "duplicate_ids", "centres_outside_own", "close_pairs",
          "close_pairs_cross_region", "members_without_cluster", "galaxies_pmem_over_1",
          "galaxies_pmem_over_1_cross_region", "versions", "calibrations"):
    print(f"{k:34s} {qa.get(k)}")
merged, mmem, _ = read_catalog(OUTDIR / "clusters_dr11.fits", members=False)
'''))

    c.append(md("""
    ## Boundaries: regions against one region

    The blind notebook processed the same sweeps as one region, with the same calibration, so
    its catalogue is the reference: a run without internal boundaries. The comparison:

    - Clusters are matched by their central galaxy (ID_CENT[0]).
    - For each reference cluster, the cell gives the distance to the nearest internal boundary
      between regions.
    - It then shows, against that distance, whether the cluster is found with the same centre,
      and how much its λ and z_λ change.

    Far from the boundaries, only floating-point noise should remain. The catalogues are made on
    the same device, so the arrays have the same shapes, but regions batch their candidates
    differently.
    """))

    c.append(code('''
ref_path = EXAMPLES / "blind" / "clusters.fits"
ref, _, rh = read_catalog(ref_path, members=False)
assert rh.get("SWEEPS") == ",".join(sorted(SWEEPS))[:1000], "run the blind notebook with the same SWEEPS"
assert rh.get("CALSHA1") == file_sha1(CALIB), "run the blind notebook with the same calibration"

# Distance (deg) to the nearest internal boundary: own-box edges shared by two regions.
owns = [plan_boxes(plan, r, meta)[0] for r in rids]
edges_dec = sorted({b.dec_min for b in owns} & {b.dec_max for b in owns})
edges_ra = sorted({b.ra_min % 360 for b in owns} & {b.ra_max % 360 for b in owns})


def boundary_distance(ra, dec):
    d = np.full(np.size(ra), np.inf)
    for e in edges_dec:
        d = np.minimum(d, np.abs(dec - e))
    for e in edges_ra:
        dra = np.abs((ra - e + 180) % 360 - 180) * np.cos(np.radians(dec))
        d = np.minimum(d, dra)
    return d


dist = boundary_distance(ref["RA"], ref["DEC"])
pos = {int(c): k for k, c in enumerate(merged["ID_CENT"][:, 0])}
found = np.array([int(c) in pos for c in ref["ID_CENT"][:, 0]])
k = np.array([pos.get(int(c), 0) for c in ref["ID_CENT"][:, 0]])
dlam = np.where(found, merged["LAMBDA"][k] / ref["LAMBDA"] - 1, np.nan)
dzl = np.where(found, merged["Z_LAMBDA"][k] - ref["Z_LAMBDA"], np.nan)
print(f"reference: {ref['LAMBDA'].size:,} clusters; merged: {merged['LAMBDA'].size:,}; "
      f"internal boundaries at Dec {edges_dec} and RA {edges_ra}")
bins = np.array([0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 99])
print(" distance [deg]   N(λ≥5)  same centre   N(λ≥20)  same centre   median |Δλ/λ|   max |Δz|")
for lo, hi in zip(bins[:-1], bins[1:]):
    s = (dist >= lo) & (dist < hi)
    s5, s20 = s & (ref["LAMBDA"] >= 5), s & (ref["LAMBDA"] >= 20)
    if not s5.any():
        continue
    f5 = found[s5].mean()
    f20 = found[s20].mean() if s20.any() else np.nan
    g = s5 & found
    print(f"  {lo:4.2f}-{hi:5.2f}     {s5.sum():6d}     {f5:6.1%}     {s20.sum():6d}     {f20:6.1%}"
          f"        {np.nanmedian(np.abs(dlam[g])) if g.any() else np.nan:9.2e}   "
          f"{np.nanmax(np.abs(dzl[g])) if g.any() else np.nan:8.1e}")
'''))

    c.append(code('''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
mid = 0.5 * (bins[:-2] + bins[1:-1])
for lmin, col in ((5, "#86b6ef"), (10, BLUE), (20, "#0d366b")):
    s = ref["LAMBDA"] >= lmin
    frac = [found[s & (dist >= lo) & (dist < hi)].mean() if np.any(s & (dist >= lo) & (dist < hi))
            else np.nan for lo, hi in zip(bins[:-2], bins[1:-1])]
    ax1.plot(mid, frac, marker="o", color=col, label=rf"$\\lambda \\geq {lmin}$")
ax1.axvline(2.0, color=GREY, lw=1, ls=(0, (2, 2)), label="buffer (2°)")
ax1.axvline(need[100]["first_order"], color=ORANGE, lw=1, label="first-order reach, λ = 100")
ax1.set(xlabel="distance to the nearest internal boundary [deg]", ylabel="same centre as one region",
        ylim=(0, 1.05))
ax1.legend(loc="lower right")
s = (ref["LAMBDA"] >= 5) & found
ax2.scatter(dist[s], np.abs(dlam[s]) + 1e-7, s=4, color=BLUE, alpha=0.5, lw=0)
ax2.axvline(2.0, color=GREY, lw=1, ls=(0, (2, 2)))
ax2.set(yscale="log", xlabel="distance to the nearest internal boundary [deg]",
        ylabel=r"$|\\Delta\\lambda / \\lambda|$ (+1e-7)", xlim=(0, min(5, dist[np.isfinite(dist)].max())))
plt.show()
'''))

    c.append(md("""
    Clusters whose z_λ changes by more than 0.01, and how their z_λ iteration ended in the
    one-region run (Z_LAMBDA_NITER; the iteration stops at `zlambda.maxiter` = 5 without
    converging):
    """))

    c.append(code('''
changed = found & (np.abs(dzl) > 0.01)
nit = ref["Z_LAMBDA_NITER"]
cap = Calibration.read(CALIB).config.zlambda.maxiter
print(f"{changed.sum()} clusters with |Δz_λ| > 0.01 ({changed.mean():.2%}); "
      f"{np.sum(changed & (nit >= cap))} of them stopped at maxiter = {cap}, as did "
      f"{np.mean(nit[ref['LAMBDA'] >= 5] >= cap):.0%} of the λ ≥ 5 clusters")
if changed.any():
    t = Table({"LAMBDA": ref["LAMBDA"][changed], "LAMBDA_REGIONS": merged["LAMBDA"][k[changed]],
               "Z_LAMBDA": ref["Z_LAMBDA"][changed], "Z_LAMBDA_REGIONS": merged["Z_LAMBDA"][k[changed]],
               "NITER": nit[changed], "DIST_DEG": dist[changed]})
    for col in t.colnames:
        if t[col].dtype.kind == "f":
            t[col].format = ".3f" if col.startswith("Z") or col == "DIST_DEG" else ".1f"
    t.sort("DIST_DEG")
    t.pprint(max_width=-1, max_lines=30)
'''))

    c.append(md("""
    ### Reading the comparison

    - **Same centre** gives the fraction of reference clusters whose central galaxy is the
      central of a merged cluster. Close to a boundary, a region does not see the percolation
      history beyond its buffer, so a low-λ cluster there can be percolated differently: it may
      lose its candidacy, or be absorbed by, or absorb, a neighbour. The effect fades with
      distance and with richness.
    - **Δλ and Δz** for matched clusters measure the same differences at the level of the
      values. Far from every boundary only floating-point noise remains: the regions batch
      their candidates differently, and float32 results change at about 1e-4 with the batch
      composition, which can reorder near-tied candidates.
    - A z_λ iteration that stops at the iteration cap has not converged, and the float32 noise
      can send it to another solution, at any distance from a boundary. The table above shows
      whether the changed clusters are of that kind. Z_LAMBDA_NITER flags them in the
      catalogue.

    In production, a region is 10° × 10° and its buffer is 2°. Close to the boundaries, the
    density profile in the merge QA (`edge_profile`) shows whether clusters are lost or doubled.
    """))

    c.append(md("""
    ## Runtimes

    Regions reread their buffers, so with one region per sweep the total area processed is
    larger than the sweeps themselves. With 100 deg² regions and 2° buffers the overhead is
    about 2×.
    """))

    c.append(code('''
for step, sec in T.items():
    print(f"{step:12s} {sec:7.1f} s")
regions_total = sum(float(read_header(region_path(OUTDIR / "regions", r), 0)["TTOTAL"]) for r in rids)
t_one = rh.get("TBLIND")
print(f"blind runs of the regions (headers): {regions_total:.0f} s against {t_one:.0f} s for the "
      f"one-region run" if t_one else f"blind runs of the regions: {regions_total:.0f} s")
'''))

    c.append(md("""
    ## The same on an HPC

    The driver submits these stages as SLURM jobs. Each phase submits only what is missing, and
    the run waits for a person to check the calibration between the two phases:

    ```bash
    export DR11=/path/to/legacysurvey/dr11/south OUTDIR=$SCRATCH/rema_dr11 DEVICE=gpu PART_GPU=gpu
    scripts/slurm/rema_dr11_blind.sh prepare          # ingest + randoms index (arrays)
    rema regions --galaxies $OUTDIR/galaxies --calib-suggest 400 --config scripts/slurm/dr11_south.yaml
    CALIB_BOX="150 170 -5 15" scripts/slurm/rema_dr11_blind.sh prepare    # calibration
    # check $OUTDIR/calib/plots and the calibration header, then:
    CALIB=$OUTDIR/calib/calib.fits scripts/slurm/rema_dr11_blind.sh run   # plan, regions, merge
    CALIB=$OUTDIR/calib/calib.fits scripts/slurm/rema_dr11_blind.sh status
    ```

    The calibration box above is only an example: use one that `--calib-suggest` proposes. After
    failures or timeouts, rerun the same command, and only the missing or stale regions are
    submitted.
    """))
    return c


# --------------------------------------------------------------------------- scan notebook
def scan_cells():
    c = []
    c.append(md(f"""
    # One cluster in scan mode, with spectroscopic post-processing

    Scan mode is redMaPPer's zscan, run at given positions (X-ray or SZ candidates). At the input
    position, λ(z) and the likelihood LNLAMLIKE(z) are computed on Δz = 0.005 steps from
    z = 0.05 to 1.0 in a fixed 0.5 h⁻¹Mpc aperture, and ZMAX is the likelihood peak. Starting
    from ZMAX, z_λ is refined with the percolation aperture (r0 = 1 h⁻¹Mpc, β = 0.2). An
    optical centre is then chosen within 0.4 h⁻¹Mpc of the input position (wcen centring), and
    λ and z_λ are recomputed there (LAMBDA_OPT, Z_LAMBDA_OPT). The spectroscopic
    post-processing adds the cluster redshift and velocity dispersion from the members' Z_SPEC.
    The target is the SZ cluster ACT-CL J0012.9-0857, given by its position only. The notebook
    runs in under a minute on an RTX 3060 laptop GPU.
    """))

    c.append(md("""
    ## Setup and parameters

    At CC-IN2P3 the defaults read the DR11 data system (sweeps, randoms, and the calibration and
    catalogues of the production run); elsewhere set `REMA_DR11_DIR`, `REMA_PRODUCTS` (or
    `REMA_CALIB`) and `REMA_WORK`. The ACT redshift is used only for the comparisons at the end.
    """))

    c.append(code(SETUP_COMMON + '''
import rema
from rema.calibration import Calibration
from rema.io.legacy import ingest
from rema.io.tables import write_catalog
from rema.model.cosmo import CosmoTable
from rema.modes import specpost
from rema.modes.common import Region
from rema.modes.scan import run_scan
from rema.sky.maps import build_footprint, read_randoms
from rema.sky.regions import Box, sweeps_overlapping

WORK = EXAMPLES / "scan"
WORK.mkdir(parents=True, exist_ok=True)

NAME = "ACT-CL J0012.9-0857"
RA, DEC = 3.23333, -8.95419
Z_ACT = 0.352          # ACT DR5 (spectroscopic), for the comparison at the end only

print(f"rema {rema.__version__}, jax {jax.__version__}, devices: {jax.devices()}")
'''))

    c.append(code(STYLE))

    c.append(md("""
    ## The region around the target

    The galaxies and the footprint are needed only around the target: the box
    `Box(RA, RA, DEC, DEC).buffered(1.0)`, about 4 deg². The largest aperture is the
    neighbour radius of the percolation aperture, 1.2 r0 3^β = 1.5 h⁻¹Mpc, plus the 0.4 h⁻¹Mpc
    of the centre search. At the lowest redshift of the scan, z = 0.05, these 1.9 h⁻¹Mpc
    subtend 0.77°, so a 1° buffer is enough. The configuration comes from the calibration
    file.

    `read_randoms` scans the whole 23 GB randoms file and keeps the randoms in the box. Here the
    file was in the page cache; read from disk it takes about a minute and dominates the
    runtime of a single scan. On an HPC, footprints are made once per region and read back
    with `Footprint.read`.
    """))

    c.append(code('''
cal = Calibration.read(CALIB)
cfg = cal.config

box = Box(RA, RA, DEC, DEC).buffered(1.0)
cosmo = CosmoTable.create(cfg.cosmology.Omega_m, cfg.cosmology.h)
rc = cfg.richness
r_max = rc.maxrad_factor * rc.percolation.r0 * 3 ** rc.percolation.beta + cfg.centering.scan_maxrad
z_min = cfg.scan.zrange[0]
print(f"box RA {box.ra_min:.3f} to {box.ra_max:.3f}, Dec {box.dec_min:.3f} to {box.dec_max:.3f}: "
      f"{box.area_deg2():.2f} deg²")
print(f"largest aperture {r_max:.2f} h⁻¹Mpc = {r_max / float(cosmo.mpc_per_deg(z_min)):.2f}° "
      f"at z = {z_min}")

sweeps = sweeps_overlapping(box, DR11 / "sweep" / "11.0")
t0 = time.perf_counter()
gal = ingest(sweeps, WORK / "galaxies.fits", cfg, box=box)
T["ingest"] = time.perf_counter() - t0
print(f"{gal['ID'].size:,} galaxies, {np.sum(gal['ZSPEC'] > 0):,} with ZSPEC > 0, "
      f"in {T['ingest']:.0f} s")

t0 = time.perf_counter()
rnd, density = read_all_randoms(cfg, box)
fp = build_footprint(rnd, cfg, box=box, density=density)
fp.write(WORK / "footprint.fits")
T["footprint"] = time.perf_counter() - t0
print(f"footprint: {rnd['RA'].size:,} randoms, unmasked area {fp.area_deg2():.2f} deg², "
      f"in {T['footprint']:.0f} s")
'''))

    c.append(md("""
    ## Calibration and zred

    The DR11 griz calibration of the production run, fitted on RA 160–180° and 190–210°,
    Dec −10° to 10°: red sequence, zred correction, χ² and zred backgrounds, z_λ correction and
    the wcen centring model. `Region.build` computes zred for every galaxy.
    """))

    c.append(code('''
print(f"{CALIB.name}: rema {cal.meta['REMAVER']}, bands {cal.meta['BANDS']}, reference "
      f"{cal.meta['REFBAND']}, χ² mode {cal.meta['CHI2MODE']}, wcen from {cal.meta['NWCEN']} clusters")

t0 = time.perf_counter()
reg = Region.build(gal, cal.rs, cfg, footprint=fp, zredcorr=cal.zredcorr, bkg=cal.bkg,
                   zlcorr=cal.zlcorr, zbkg=cal.zbkg, wcen_params=cal.wcen)
T["zred"] = time.perf_counter() - t0
method = reg.centering_method()
print(f"zred for {reg.gal['ZRED'].size:,} galaxies in {T['zred']:.1f} s; centring: {method}")
'''))

    c.append(md("""
    ## Scan and spectroscopic post-processing

    `run_scan` takes lists of positions (and identifiers), processed in batches of 128. The
    spectroscopic post-processing uses the members of the fit at the input position (PMEM ≥
    0.01). The output is written as `rema scan` writes it.
    """))

    c.append(code('''
t0 = time.perf_counter()
cat, mem = run_scan(reg, [RA], [DEC], ids=[0])
T["scan"] = time.perf_counter() - t0
nz = int(round((cfg.scan.zrange[1] - cfg.scan.zrange[0]) / cfg.scan.zstep)) + 1
assert cat["Z_STEPS"].shape == cat["LAMBDA_STEPS"].shape == cat["LIKELIHOOD_STEPS"].shape == (1, nz)
print(f"scan in {T['scan']:.1f} s: λ(z) at {nz} redshifts, z = {cat['Z_STEPS'][0, 0]:.3f} "
      f"to {cat['Z_STEPS'][0, -1]:.3f}; {mem['ID'].size} members")

t0 = time.perf_counter()
cat, mem = specpost.process(cat, mem, **dataclasses.asdict(cfg.spec))
T["specpost"] = time.perf_counter() - t0
path = write_catalog(WORK / "scan.fits", cat, mem, cfg, {"MODE": "scan", "CENTRING": method})
print(f"wrote {path}")
'''))

    c.append(md("""
    ## Results

    LMAX is λ at ZMAX in the 0.5 h⁻¹Mpc scan aperture. LAMBDA, Z_LAMBDA, SCALEVAL and MASKFRAC
    are those of the percolation aperture at the input position, and the …_OPT columns those at
    the optical centre. The offset in h⁻¹Mpc is at Z_LAMBDA_OPT.
    """))

    c.append(code('''
cl = {k: v[0] for k, v in cat.items()}
off = SkyCoord(RA, DEC, unit="deg").separation(SkyCoord(cl["RA_OPT"], cl["DEC_OPT"], unit="deg"))
off_mpc = off.deg * reg.mpc_per_deg_np(cl["Z_LAMBDA_OPT"])
rows = [
    ("input position (RA, Dec)", f"{cl['RA']:.5f}, {cl['DEC']:.5f}"),
    ("ZMAX, LMAX", f"{cl['ZMAX']:.3f}, {cl['LMAX']:.1f}" + (" (grid edge)" if cl["ZMAX_EDGE"] else "")),
    ("Z_LAMBDA", f"{cl['Z_LAMBDA']:.4f} ± {cl['Z_LAMBDA_E']:.4f}"),
    ("LAMBDA", f"{cl['LAMBDA']:.1f} ± {cl['LAMBDA_E']:.1f}"),
    ("RA_OPT, DEC_OPT", f"{cl['RA_OPT']:.5f}, {cl['DEC_OPT']:.5f}"),
    ("offset of the optical centre", f"{off.arcsec:.1f}″ = {off_mpc:.3f} h⁻¹Mpc"),
    ("Z_LAMBDA_OPT", f"{cl['Z_LAMBDA_OPT']:.4f} ± {cl['Z_LAMBDA_OPT_E']:.4f}"),
    ("LAMBDA_OPT", f"{cl['LAMBDA_OPT']:.1f} ± {cl['LAMBDA_OPT_E']:.1f}"),
    ("R_LAMBDA", f"{cl['R_LAMBDA']:.3f} h⁻¹Mpc"),
    ("SCALEVAL, MASKFRAC", f"{cl['SCALEVAL']:.3f}, {cl['MASKFRAC']:.3f}"),
    ("P_CEN[0], NCENT_GOOD", f"{cl['P_CEN'][0]:.3f}, {cl['NCENT_GOOD']}"),
    ("NSPEC, N_MEMBERS", f"{cl['NSPEC']}, {cl['N_MEMBERS']}"),
    ("SPEC_Z_BOOT", f"{cl['SPEC_Z_BOOT']:.4f} ± {cl['SPEC_ZERR_BOOT']:.4f}"),
    ("VDISP", f"{cl['VDISP']:.0f} ± {cl['VDISP_ERR']:.0f} km/s ({cl['VDISP_TYPE']})"),
    ("BEST_Z", f"{cl['BEST_Z']:.4f} ({cl['BEST_Z_TYPE']})"),
]
width = max(len(label) for label, _ in rows)
for label, value in rows:
    print(f"{label:<{width}}  {value}")

print(f"\\nagainst z_ACT = {Z_ACT}:")
for key in ("ZMAX", "Z_LAMBDA_OPT", "SPEC_Z_BOOT"):
    print(f"  ({key} − z_ACT)/(1 + z_ACT) = {(cl[key] - Z_ACT) / (1 + Z_ACT):+.4f}")

# Every spectroscopic redshift near the target between 0.30 and 0.42, members or not.
near = SkyCoord(gal["RA"], gal["DEC"], unit="deg").separation(SkyCoord(RA, DEC, unit="deg")).arcmin < 6
zs = np.sort(gal["ZSPEC"][near & (gal["ZSPEC"] > 0.30) & (gal["ZSPEC"] < 0.42)])
print(f"{zs.size} galaxies within 6′ with 0.30 < ZSPEC < 0.42: ZSPEC {zs.min():.4f} to "
      f"{zs.max():.4f}, {np.sum(np.abs(zs - Z_ACT) < 0.005)} within 0.005 of z_ACT")
'''))

    c.append(md("""
    z_λ at the optical centre (0.336) agrees with the spectroscopic redshift of the 14 kept
    members (0.337 ± 0.002). ZMAX, z_λ and SPEC_Z_BOOT all lie 0.009–0.012 (1 + z) below the
    ACT redshift, and no spectroscopic redshift within 6′ is near it: the 17 between 0.30 and
    0.42 span 0.327–0.341. The optical centre is 79″ (0.27 h⁻¹Mpc) from the SZ position, with
    the P_CEN printed above. Blind mode centres this cluster on its brightest member, 0.49 h⁻¹Mpc from the
    SZ position, outside the 0.4 h⁻¹Mpc search radius of scan mode (see the blind notebook).
    """))

    c.append(md("""
    ## Figures

    λ(z) and LNLAMLIKE(z) along the scan, and p(z) of the refined z_λ at the input position.
    LNLAMLIKE is undefined where λ ≤ 0; those steps are left out. Both curves have a second
    maximum near z = 0.39. λ is higher there (32.3 against 30.9), but the likelihood is higher
    at z = 0.340 (42.5 against 41.0), which sets ZMAX.
    """))

    c.append(code('''
z, lz, like = cl["Z_STEPS"], cl["LAMBDA_STEPS"], cl["LIKELIHOOD_STEPS"]
ok = (lz > 0) & np.isfinite(like)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), constrained_layout=True)
axes[0].plot(z, np.where(lz > 0, lz, np.nan), color=BLUE, lw=1.5)
axes[1].plot(z[ok], like[ok], color=BLUE, lw=1.5)
for ax in axes[:2]:
    ax.axvline(cl["ZMAX"], color=ORANGE, lw=1, label=f"ZMAX {cl['ZMAX']:.3f}")
    ax.axvline(cl["Z_LAMBDA"], color=AQUA, lw=1, ls=(0, (1, 1.5)), label=f"Z_LAMBDA {cl['Z_LAMBDA']:.3f}")
    ax.set_xlabel("z")
axes[0].set_ylabel(r"$\\lambda$ in the 0.5 h⁻¹Mpc aperture")
axes[1].set_ylabel("LNLAMLIKE")
axes[1].legend(loc="upper right")
axes[2].plot(cl["PZBINS"], cl["PZ"], color=BLUE, lw=1.5, marker="o", ms=3)
axes[2].axvline(cl["Z_LAMBDA_RAW"], color=AQUA, lw=1, ls=(0, (1, 1.5)),
                label=f"Z_LAMBDA_RAW {cl['Z_LAMBDA_RAW']:.3f}")
axes[2].set(xlabel="z", ylabel="p(z)", ylim=(0, 1.3 * cl["PZ"].max()))
axes[2].legend(loc="upper left")
plt.show()
'''))

    c.append(md("""
    Left: the members on the sky around the input position, coloured by PMEM, with symbol
    size by z-band magnitude. The circles have radius R_LAMBDA and 0.4 h⁻¹Mpc (the centre
    search). The centre candidates are labelled with P_CEN; the optical centre is the most
    probable one. Right: the rest-frame velocities of the members with a spectroscopic redshift,
    relative to SPEC_Z, with a Gaussian of width VDISP.
    """))

    c.append(code('''
cosd = np.cos(np.radians(DEC))


def offsets(ra, dec):
    """Offsets from the input position [arcmin]."""
    return (np.asarray(ra) - RA) * cosd * 60, (np.asarray(dec) - DEC) * 60


dx, dy = offsets(mem["RA"], mem["DEC"])
pm, mag = mem["PMEM"], mem["REFMAG"]
o = np.argsort(pm)
arcmin_per_mpc = 60 / reg.mpc_per_deg_np(cl["Z_LAMBDA"])
r_lam = cl["R_LAMBDA"] * arcmin_per_mpc

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True,
                               gridspec_kw={"width_ratios": [1.15, 1]})
sc = ax1.scatter(dx[o], dy[o], c=pm[o], cmap=RAMP, vmin=0, vmax=1,
                 s=8 + 60 * 10 ** (-0.4 * (mag[o] - mag.min())), edgecolor="white", lw=0.5)
ax1.add_patch(plt.Circle((0, 0), r_lam, fill=False, color=GREY, lw=1, label="R_LAMBDA"))
ax1.add_patch(plt.Circle((0, 0), cfg.centering.scan_maxrad * arcmin_per_mpc, fill=False,
                         color=GREY, lw=1, ls=(0, (1, 1.5)), label="0.4 h⁻¹Mpc"))
ax1.scatter(0, 0, marker="+", s=150, color="black", lw=1.5, label="input position")
okc = cl["ID_CENT"] >= 0
cx, cy = offsets(cl["RA_CENT"][okc], cl["DEC_CENT"][okc])
ax1.set(xlim=(1.2 * r_lam, -1.2 * r_lam), ylim=(-1.2 * r_lam, 1.2 * r_lam),
        xlabel="ΔRA cos Dec [arcmin]", ylabel="ΔDec [arcmin]", title=f"{NAME}: {pm.size} members")
ax1.set_aspect("equal")
label_candidates(ax1, cx, cy, cl["P_CEN"][okc])
ox, oy = offsets(cl["RA_OPT"], cl["DEC_OPT"])
ax1.scatter(ox, oy, marker="*", s=110, color=ORANGE, edgecolor="white", lw=0.5, zorder=4,
            label="optical centre")
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
fig.colorbar(sc, cax=ax1.inset_axes([1.03, 0.1, 0.035, 0.8]), label="PMEM")

vel, kept = mem["VEL"], mem["ISMEMBER_SPEC"]
has = np.isfinite(vel)
sig = cl["VDISP"]
if np.isfinite(sig):
    width = max(100.0, 100.0 * round(sig / 200))       # about σ/2, in steps of 100 km/s
    vmax = width * np.ceil(4 * sig / width)
    bins = np.arange(-vmax, vmax + width / 2, width)
    ax2.hist(vel[has & ~kept], bins=bins, color=GREY, edgecolor="white", lw=1,
             label=f"clipped ({np.sum(has & ~kept)})")
    ax2.hist(vel[kept], bins=bins, color=BLUE, edgecolor="white", lw=1, label=f"kept ({kept.sum()})")
    vv = np.linspace(-vmax, vmax, 300)
    norm = kept.sum() * width / (np.sqrt(2 * np.pi) * sig)
    ax2.plot(vv, norm * np.exp(-0.5 * (vv / sig) ** 2), color=ORANGE, lw=1.5,
             label=f"VDISP {sig:.0f} km/s")
    out = has & (np.abs(vel) > vmax)
    if out.any():
        ax2.text(0.98, 0.97, "not shown: " + ", ".join(f"{v:+,.0f}" for v in vel[out]) + " km/s",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=8)
    ax2.yaxis.get_major_locator().set_params(integer=True)
    ax2.set(xlabel="rest-frame velocity [km/s]", ylabel="members",
            title=f"{has.sum()} members with ZSPEC, SPEC_Z {cl['SPEC_Z']:.4f}")
    ax2.legend(loc="upper left")
else:
    ax2.text(0.5, 0.5, "fewer than 3 spectroscopic members", ha="center", transform=ax2.transAxes)
plt.show()
'''))

    c.append(md("""
    ## Members

    The ten members with the highest PMEM. R is the distance from the input position in
    h⁻¹Mpc; ZSPEC is −1 without a spectroscopic redshift.
    """))

    c.append(code('''
top = np.argsort(-mem["PMEM"])[:10]
members = Table({k: mem[k][top] for k in ("REFMAG", "ZRED", "ZSPEC", "R", "PMEM")})
for col, fmt in (("REFMAG", ".2f"), ("ZRED", ".3f"), ("ZSPEC", ".4f"), ("R", ".3f"), ("PMEM", ".3f")):
    members[col].format = fmt
members.pprint(max_width=-1)
'''))

    c.append(md("""
    ## The production catalogue

    The DR11 south production run found clusters blind over the whole footprint with the same
    calibration (merged catalogues on the data system, `PRODUCTS`). The cell lists the
    production clusters centred within 3′ of the input position: scan mode, which starts from
    the position, and blind mode, which starts from its own seeds, should find the same system.
    """))

    c.append(code('''
from rema.io.tables import read_catalog

target = SkyCoord(RA, DEC, unit="deg")
found = False
for run in RUNS:
    f = run / "clusters_dr11.fits"
    if not f.exists():
        print(f"{run.name}: not available yet")
        continue
    pc, _, _ = read_catalog(f, members=False)
    sep = SkyCoord(pc["RA"], pc["DEC"], unit="deg").separation(target).arcmin
    near = np.flatnonzero(sep < 3)
    if near.size == 0:
        continue
    found = True
    near = near[np.argsort(sep[near])]
    tab = Table({"PART": [run.name] * near.size, "SEP_ARCMIN": sep[near],
                 **{k: np.asarray(pc[k])[near] for k in ("LAMBDA", "Z_LAMBDA", "SPEC_Z_BOOT", "N_MEMBERS")}})
    for col, fmt in (("SEP_ARCMIN", ".2f"), ("LAMBDA", ".1f"), ("Z_LAMBDA", ".4f"), ("SPEC_Z_BOOT", ".4f")):
        tab[col].format = fmt
    tab.pprint(max_width=-1)
print(f"scan mode: LAMBDA_OPT {cl['LAMBDA_OPT']:.1f}, Z_LAMBDA_OPT {cl['Z_LAMBDA_OPT']:.4f}" if found
      else "no production cluster within 3′")
'''))

    c.append(md("""
    ## Runtimes
    """))

    c.append(code('''
for step, sec in T.items():
    print(f"{step:10s} {sec:6.1f} s")
print(f"{'total':10s} {sum(T.values()):6.1f} s on {jax.devices()[0].device_kind}")
'''))

    c.append(md("""
    ## The same run from the command line

    The positions file is a FITS table with columns RA, DEC and an identifier:
    """))

    c.append(code('''
positions = Table({"RA": [RA], "DEC": [DEC], "NAME": [NAME]})
positions.write(WORK / "positions.fits", overwrite=True)
print(positions)
'''))

    c.append(md("""
    ```bash
    rema ingest DR11/sweep/11.0/sweep-000m010-005m005.fits --box 2.221 4.246 -9.955 -7.954 \\
         --out galaxies.fits
    rema maps DR11/randoms/randoms-south-1-0.fits --box 2.221 4.246 -9.955 -7.954 --out footprint.fits
    rema scan --galaxies galaxies.fits --calib CALIB --footprint footprint.fits \\
         --positions positions.fits --id-col NAME --specpost --out scan.fits
    ```

    MEM_MATCH_ID is then the NAME column. `rema scan` uses the calibration's configuration
    unless `--config` is given. As for blind mode, pass `--config` to `rema ingest` and
    `rema maps` when the calibration was made with a non-default configuration, and set
    `JAX_PLATFORMS=cuda` to run on a GPU.
    """))
    return c



# --------------------------------------------------------------------------- write, run, check
BUILDERS = {"blind": blind_cells, "pipeline": pipeline_cells, "scan": scan_cells}
FORBIDDEN = ("cla" + "ude", "/tmp/")


def check_source(nb):
    """Every code cell must compile, and no cell may contain a forbidden string."""
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<cell>", "exec")
        for word in FORBIDDEN:
            assert word not in cell.source.lower(), word
    return nb


def check_executed(path: Path) -> list[str]:
    """Problems of an executed notebook: errors, gaps in the execution counts, forbidden strings."""
    nb = nbformat.read(path, as_version=4)
    problems = []
    counts = [c.execution_count for c in nb.cells if c.cell_type == "code"]
    if counts != list(range(1, len(counts) + 1)):
        problems.append(f"execution counts {counts}")
    for c in nb.cells:
        for o in c.get("outputs", []):
            if o.output_type == "error":
                problems.append(f"error {o.ename}: {o.evalue}")
    text = path.read_text().lower()
    problems += [f"contains {w!r}" for w in FORBIDDEN if w in text]
    return problems


def execute(path: Path, timeout: int = -1):
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="rema", resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("names", nargs="*", help=f"notebooks among {ORDER} (default all)")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    unknown = set(args.names) - set(ORDER)
    if unknown:
        p.error(f"unknown notebooks {sorted(unknown)}")
    names = [n for n in ORDER if n in (args.names or ORDER)]
    if args.check:
        bad = 0
        for n in names:
            probs = check_executed(HERE / f"{n}_dr11.ipynb")
            print(f"{n}_dr11.ipynb: " + ("ok" if not probs else "; ".join(probs)))
            bad += bool(probs)
        return 1 if bad else 0
    for n in names:
        path = HERE / f"{n}_dr11.ipynb"
        nbformat.write(check_source(notebook(BUILDERS[n]())), path)
        print("wrote", path)
        if args.execute:
            t0 = time.time()
            execute(path)
            print(f"executed {path.name} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
