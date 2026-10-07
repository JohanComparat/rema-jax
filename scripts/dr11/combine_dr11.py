"""Combine the two parts of the DR11 south rema 0.2.0 run into one catalogue.

Adds PART (0: RA 0-240, Dec > -85; 1: RA 240-360 and the polar cap Dec < -85), GLON, GLAT,
SEAM_DIST (deg to the nearest boundary between the parts) and FLAG_SEAM (SEAM_DIST < 2.33 deg),
and makes MEM_MATCH_ID unique: PART << 40 | the part's MEM_MATCH_ID (region << 32 | rank).

The script of the DR11 south production of rema 0.2.0 at CC-IN2P3 (docs: "The DR11 south catalogue
at CC-IN2P3"); it reads and writes the data system directly. Run it in the rema environment with
JAX_PLATFORMS=cpu, in a job with enough memory for the members (128 GB on htc).
"""
import json
import sys
import time
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

R = Path("/sps/lsst/datasets/desi/legacysurveys/dr11/south/rema")
PARTS = ["rema_dr11_v0.2.0_ra0-240", "rema_dr11_v0.2.0_ra240-360"]
OUT = R / "rema_dr11_v0.2.0"
SEAM = 2.33          # deg: the exact reach of a lambda = 100 cluster at z = 0.05 (region buffer 2 deg)
t0 = time.time()
OUT.mkdir(exist_ok=True)


def seam_distance(ra, dec):
    """Angular distance [deg] to the boundaries between the parts: the meridians RA = 0 and 240
    above Dec -85, and the parallel Dec = -85 between RA 0 and 240."""
    ra, dec = np.radians(ra), np.radians(dec)
    d = np.full(ra.shape, np.inf)
    for ra0 in (0.0, 240.0):
        # distance to a meridian: arcsin(cos(dec) |sin(ra - ra0)|), valid on the near side
        dra = (ra - np.radians(ra0) + np.pi) % (2 * np.pi) - np.pi
        dm = np.degrees(np.arcsin(np.clip(np.cos(dec) * np.abs(np.sin(dra)), 0, 1)))
        dm = np.where(np.abs(dra) < np.pi / 2, dm, np.inf)
        # the meridian segment ends at Dec -85: below it, distance to its end point
        end = np.degrees(np.arccos(np.clip(np.sin(dec) * np.sin(np.radians(-85)) +
                                           np.cos(dec) * np.cos(np.radians(-85)) * np.cos(dra), -1, 1)))
        d = np.minimum(d, np.where(np.degrees(dec) >= -85, dm, end))
    # the parallel Dec = -85 between RA 0 and 240
    ra_deg = np.degrees(ra) % 360
    dp = np.abs(np.degrees(dec) + 85.0)
    d = np.minimum(d, np.where(ra_deg < 240, dp, np.inf))
    return d


clusters, members, configs, qas, hdrs = [], [], [], [], []
for k, p in enumerate(PARTS):
    with fits.open(R / p / "clusters_dr11.fits") as h:
        c = np.array(h["CLUSTERS"].data)
        hdrs.append(h[0].header.copy())
        configs.append(h["CONFIG"].copy() if "CONFIG" in h else None)
    with fits.open(R / p / "clusters_dr11_members.fits") as h:
        m = np.array(h["MEMBERS"].data)
    qas.append(json.loads((R / p / "clusters_dr11_qa.json").read_text()))
    off = np.int64(k) << np.int64(40)
    assert c["MEM_MATCH_ID"].max() < (np.int64(1) << np.int64(40))
    c["MEM_MATCH_ID"] = c["MEM_MATCH_ID"] + off
    m["MEM_MATCH_ID"] = m["MEM_MATCH_ID"] + off
    clusters.append((c, k)); members.append((m, k))
    print(f"{p}: {len(c):,} clusters, {len(m):,} members ({time.time() - t0:.0f}s)", flush=True)

cat = np.concatenate([c for c, _ in clusters])
part = np.concatenate([np.full(len(c), k, np.int16) for c, k in clusters])
mem = np.concatenate([m for m, _ in members])
mpart = np.concatenate([np.full(len(m), k, np.int16) for m, k in members])
gal = SkyCoord(cat["RA"] * u.deg, cat["DEC"] * u.deg).galactic
seam = seam_distance(cat["RA"], cat["DEC"]).astype(np.float32)

# Checks before writing.
ids = cat["MEM_MATCH_ID"]
assert np.unique(ids).size == ids.size, "duplicate MEM_MATCH_ID"
assert np.isin(mem["MEM_MATCH_ID"], ids).all(), "members without a cluster"
assert np.abs(gal.b.deg).min() >= 15.0 - 1e-6
print(f"checks ok: {len(cat):,} unique clusters, every member has its cluster ({time.time() - t0:.0f}s)", flush=True)

ccols = fits.ColDefs(fits.BinTableHDU(cat).columns) + fits.ColDefs([
    fits.Column("PART", "I", array=part), fits.Column("GLON", "D", array=gal.l.deg),
    fits.Column("GLAT", "D", array=gal.b.deg), fits.Column("SEAM_DIST", "E", array=seam),
    fits.Column("FLAG_SEAM", "L", array=seam < SEAM)])
mcols = fits.ColDefs(fits.BinTableHDU(mem).columns) + fits.ColDefs([fits.Column("PART", "I", array=mpart)])

ph = hdrs[0].copy()
for key, val, com in (("MODE", "combined", "the two parts of the DR11 south run"),
                      ("NPART", 2, "PART 0: RA 0-240 Dec>-85; PART 1: RA 240-360 + Dec<-85"),
                      ("PART0", PARTS[0], ""), ("PART1", PARTS[1], ""),
                      ("NREGION", int(sum(q["n_done"] for q in qas)), "regions merged, both parts"),
                      ("IDOFFSET", 40, "MEM_MATCH_ID = PART << 40 | region << 32 | rank"),
                      ("SEAMDIST", SEAM, "FLAG_SEAM: deg from a part boundary below this"),
                      ("GLATMIN", 15.0, "clusters at |b| >= GLATMIN")):
    ph[key] = (val, com)
hdus = [fits.PrimaryHDU(header=ph), fits.BinTableHDU.from_columns(ccols, name="CLUSTERS")]
if configs[0] is not None:
    hdus.append(configs[0])
tmp = OUT / ".clusters_dr11.fits.part"
fits.HDUList(hdus).writeto(tmp, overwrite=True); tmp.rename(OUT / "clusters_dr11.fits")
tmp = OUT / ".clusters_dr11_members.fits.part"
fits.HDUList([fits.PrimaryHDU(header=ph), fits.BinTableHDU.from_columns(mcols, name="MEMBERS")]).writeto(tmp, overwrite=True)
tmp.rename(OUT / "clusters_dr11_members.fits")

lam = cat["LAMBDA"]
qa = {"parts": PARTS, "n_clusters": int(len(cat)), "n_members": int(len(mem)),
      "n_clusters_part": [int(np.sum(part == k)) for k in (0, 1)],
      "n_regions_done": [q["n_done"] for q in qas], "n_regions": [q["n_regions"] for q in qas],
      "missing_regions": [q["missing"] for q in qas], "glat_min": 15.0,
      "n_flag_seam": int(np.sum(seam < SEAM)), "n_lambda_ge": {str(t): int(np.sum(lam >= t)) for t in (5, 10, 20, 50, 100)},
      "id": "MEM_MATCH_ID = PART << 40 | region << 32 | rank", "calibrations": qas[0].get("calibrations"),
      "versions": qas[0].get("versions")}
(OUT / "clusters_dr11_qa.json").write_text(json.dumps(qa, indent=1))
# Read back.
with fits.open(OUT / "clusters_dr11.fits", memmap=True) as h:
    assert h["CLUSTERS"].header["NAXIS2"] == len(cat)
with fits.open(OUT / "clusters_dr11_members.fits", memmap=True) as h:
    assert h["MEMBERS"].header["NAXIS2"] == len(mem)
print(json.dumps({k: qa[k] for k in ("n_clusters", "n_members", "n_clusters_part", "n_flag_seam", "n_lambda_ge")}))
print(f"done in {time.time() - t0:.0f}s")
