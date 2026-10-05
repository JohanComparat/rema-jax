"""Hands-on tour of rema on mock data: blind mode, scan mode, spectroscopic post-processing and
derivatives of the richness. About two minutes on a laptop CPU with a cold compilation cache.

Run: python docs/examples/mock_tour.py   (GPU: JAX_PLATFORMS=cuda python docs/examples/mock_tour.py)
"""

# [setup]
import dataclasses
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/rema/jax"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

import jax.numpy as jnp
import numpy as np

from rema.config import RemaConfig
from rema.model.cosmo import CosmoTable
from rema.model.profiles import MStar
from rema.model.redsequence import RSModel
from rema.modes.common import Region
from rema.sky.regions import Box
from rema.validate import mocks
# [/setup]

# [mock]
rng = np.random.default_rng(2)
cfg = RemaConfig()                                  # defaults: griz, z reference band
rs = RSModel.from_template()                        # BC03 griz red sequence (no calibration)
mstar, cosmo = MStar(cfg.model.mstar), CosmoTable.create()
depth5 = np.array([24.9, 24.7, 24.2, 23.6])         # 5-sigma griz depths of the mock
box = Box(10.0, 11.5, -0.75, 0.75)                  # RA0, RA1, Dec0, Dec1 in degrees

field = mocks.mock_field(rng, rs, box, density=12000, depth5=depth5, mag_range=(12.0, 22.5))
truth = [(10.5, -0.3, 0.30, 40.0), (11.0, 0.3, 0.55, 30.0)]     # RA, Dec, z, lambda
clusters = [mocks.mock_cluster(rng, rs, mstar, cosmo.mpc_per_deg, ra, dec, z, lam, depth5,
                               poisson=False, central_dmag=-1.5)
            for ra, dec, z, lam in truth]
gal = mocks.concat(field, *clusters)
gal["ID"] = np.arange(gal["RA"].size, dtype=np.int64)
gal["ZSPEC"] = np.full(gal["RA"].size, -1.0, np.float32)

# Spectroscopic redshifts for the 15 brightest members of each cluster (sigma_v = 700 km/s).
start = field["RA"].size
for (_, _, z, _), cl in zip(truth, clusters):
    rows = start + np.argsort(cl["REFMAG"])[:15]
    gal["ZSPEC"][rows] = z + (1 + z) * rng.normal(0.0, 700.0, rows.size) / 299792.458
    start += cl["RA"].size

reg = Region.build(gal, rs, cfg, area_deg2=box.area_deg2())   # zred and chi^2 background
print(f"{gal['RA'].size} galaxies; centring: {reg.centering_method()}")
# [/mock]

# [blind]
from rema.modes import specpost
from rema.modes.blind import run_blind

cat, mem = run_blind(reg, own=box)
cat, mem = specpost.process(cat, mem, **dataclasses.asdict(cfg.spec))
print(f"{cat['LAMBDA'].size} clusters (lambda/S >= 3, MASKFRAC < 0.2), {mem['ID'].size} members")
print("    RA     Dec   Z_LAMBDA  LAMBDA  SPEC_Z_BOOT  VDISP +- ERR  N_MEMBERS")
for i in np.argsort(-cat["LAMBDA"])[:3]:
    print(f"{cat['RA'][i]:7.3f} {cat['DEC'][i]:6.3f}  {cat['Z_LAMBDA'][i]:7.4f}  {cat['LAMBDA'][i]:6.1f}"
          f"  {cat['SPEC_Z_BOOT'][i]:11.4f}  {cat['VDISP'][i]:5.0f} +- {cat['VDISP_ERR'][i]:3.0f}"
          f"  {cat['N_MEMBERS'][i]:9d}")
# [/blind]

# [scan]
from rema.modes.scan import run_scan

ra, dec = np.array([t[0] for t in truth]), np.array([t[1] for t in truth])
scat, smem = run_scan(reg, ra, dec)
for i in range(ra.size):
    off = 3600 * np.hypot((scat["RA_OPT"][i] - ra[i]) * np.cos(np.radians(dec[i])), scat["DEC_OPT"][i] - dec[i])
    print(f"input {i}: ZMAX {scat['ZMAX'][i]:.3f}  Z_LAMBDA_OPT {scat['Z_LAMBDA_OPT'][i]:.4f}"
          f"  LAMBDA_OPT {scat['LAMBDA_OPT'][i]:.1f}  centre offset {off:.1f} arcsec")
# [/scan]

# [grad]
from rema.core.richness import RadialQuad, Stage, richness

st, quad = Stage.make(1.0, 0.2), RadialQuad.make()          # percolation aperture: r0 = 1, beta = 0.2
ra0, dec0, z0 = truth[0][:3]
pad = reg.query([ra0], [dec0], reg.radius_deg(float(st.maxrad), z0))   # neighbours, padded [1, K]
nb = reg.neighbors(pad)
frad, fgeo = reg.completeness([ra0], [dec0], [z0], quad)    # all ones without a footprint


def lam(z, model=reg.model):
    return richness(nb, z, frad, fgeo, quad, model, st).lam[0]


z = jnp.array([z0], jnp.float32)
h = 3e-5                                                     # see the note on finite differences
print(f"lambda {lam(z):.2f}  dlambda/dz {jax.grad(lam)(z)[0]:.1f}"
      f"  (finite difference {(lam(z + h) - lam(z - h)) / (2 * h):.1f})")


def lam_of_colours(mean):                                    # mean colours at the z nodes [nz, ncol]
    model = dataclasses.replace(reg.model, rs=dataclasses.replace(reg.model.rs, mean=mean))
    return lam(z, model)


g = np.asarray(jax.grad(lam_of_colours)(reg.model.rs.mean))   # same shape as the nodes
k = int(np.argmin(np.abs(np.asarray(reg.model.rs.z_mean) - z0)))
print(f"dlambda/d(mean colour) at the z = {float(reg.model.rs.z_mean[k]):.2f} node:"
      f"  g-r {g[k, 0]:.1f}  r-i {g[k, 1]:.1f}  i-z {g[k, 2]:.1f}")
# [/grad]
