"""Reduced tables and maps for the figures of docs/redmapper_dr11.rst.

    python docs/figures/redmapper_dr11/prepare.py [--force]

Writes into $REMA_WORK (see common.py), each table as a directory of .npy columns:

clusters/   every cluster of the production parts present, with PART, REGION, GLAT, IN_CALIB,
            SEAM_DIST and the depth-map values at its position (DEPTH_Z10, ZVLIM, ...)
members/    members with P >= 0.05 or a spectroscopic redshift, with CL_ROW (row in clusters/)
            and their colours
maps/       HEALPix NEST maps (NSIDE 256) from the DR11 randoms: area of the run, depths, E(B-V),
            PSF size, masked fraction and z_vlim
external/   public cluster catalogues with common column names (RA, DEC, Z, ...)
meta.json   parts, areas and the signatures that decide what is up to date
"""

# %% [script-only]
from common import *  # noqa: F403

# %% [markdown]
# ## Reduced tables and maps (`prepare.py`)
#
# The production catalogue is read once and reduced to the columns the figures use. The DR11
# randoms give the footprint of the run, cut as the galaxies were (`MASKBITS`, at least one
# exposure in g, r, i and z, E(B−V) < 0.2), to |b| ≥ 15° and to the own boxes of the regions
# that finished. Each step is skipped when its products are newer than its inputs.

# %%
from rema.calibration import Calibration
from rema.config import EXTINCTION_DECAM
from rema.io.tables import read_table
from rema.model.profiles import MStar, maxmag_from_mstar
from rema.pipeline import galactic_latitude, read_plan
from rema.sky.maps import good_randoms

FORCE = "--force" in sys.argv[1:] if FIGDIR is not None else False
NSIDE = 256
CAL = Calibration.read(CALIB)
CFG = CAL.config
BANDS = [b.upper() for b in CFG.survey.bands]
DENSITY = CFG.mask.randoms_density                    # randoms per deg2 in one DR11 randoms file
PARTS = [k for k, run in enumerate(RUNS) if (run / "clusters_dr11.fits").exists()]
print("parts present:", [RUNS[k].name for k in PARTS])


def signature(paths):
    return [[str(Path(p).name), os.path.getsize(p), int(os.path.getmtime(p))] for p in paths]


def write_columns(name, cols):
    d = WORK / name
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.npy"):
        old.unlink()
    for k, v in cols.items():
        np.save(d / f"{k}.npy", np.ascontiguousarray(v))


def up_to_date(key, sig):
    m = json.loads((WORK / "meta.json").read_text()) if (WORK / "meta.json").exists() else {}
    return not FORCE and m.get(key) == sig


def update_meta(**kw):
    m = json.loads((WORK / "meta.json").read_text()) if (WORK / "meta.json").exists() else {}
    m.update(kw)
    (WORK / "meta.json").write_text(json.dumps(m, indent=1))


# %% [markdown]
# ### The footprint of each part
#
# The own boxes of the regions that finished (the region plan minus the regions listed as
# missing in the QA file): a random point belongs to the run if it falls in one of them.

# %%
OWN = {}
for k in PARTS:
    plan, _ = read_plan(RUNS[k] / "regions.fits")
    qa = json.loads((RUNS[k] / "clusters_dr11_qa.json").read_text())
    done = ~np.isin(plan["REGION_ID"], qa["missing"])
    OWN[k] = np.stack([plan[f"OWN_{a}"][done] for a in ("RA0", "RA1", "DEC0", "DEC1")], axis=1)
    print(f"{RUNS[k].name}: {done.sum()} of {done.size} regions done, missing {qa['missing']}")


def part_of(ra, dec):
    """Part whose finished own boxes contain each position (-1 for none)."""
    out = np.full(ra.size, -1, np.int8)
    for k, boxes in OWN.items():
        for r0, r1, d0, d1 in boxes:
            rr = (ra - r0) % 360.0
            inside = (rr < (r1 - r0)) & (dec >= d0) & (dec < d1)
            out[inside] = k
    return out


# %% [markdown]
# ### Depth and z_vlim maps from the randoms
#
# Per HEALPix pixel (NSIDE 256, 0.05 deg²): the number of randoms in the run (its area at 2500
# randoms per deg²), the mean 10σ galaxy depth in each band corrected for Galactic extinction,
# the mean E(B−V) and z-band PSF size, and the fraction of observed randoms removed by the mask
# bits. As in Kluge et al. (2024, Fig. B.3), z_vlim is the redshift at which a 0.2 L* galaxy,
# m*(z) + 1.75 in the reference band (z), reaches the 10σ depth.

# %%
MSTAR = MStar(CFG.model.mstar)
ZGRID = np.asarray(MSTAR.z)
MLIM = np.asarray(maxmag_from_mstar(np.asarray(MSTAR.m), CFG.model.lval_reference))   # m* + 1.75


def zvlim_of(depth10):
    return np.interp(depth10, MLIM, ZGRID, left=np.nan, right=ZGRID[-1])


def read_randoms_chunks(path, columns, chunk=4_000_000):
    """Yield dicts of columns of a randoms file, chunk by chunk."""
    try:
        import fitsio

        with fitsio.FITS(str(path)) as f:
            n = f[1].get_nrows()
            for lo in range(0, n, chunk):
                d = f[1].read(columns=columns, rows=np.arange(lo, min(n, lo + chunk)))
                yield {c: d[c].astype(d[c].dtype.newbyteorder("=")) for c in columns}
    except ImportError:
        from astropy.io import fits

        with fits.open(path, memmap=True) as h:
            data = h[1].data
            for lo in range(0, len(data), chunk):
                sl = slice(lo, lo + chunk)
                yield {c: np.asarray(data[c][sl]).astype(data[c].dtype.newbyteorder("=")) for c in columns}


maps_sig = {"randoms": signature([RANDOMS]), "parts": [RUNS[k].name for k in PARTS],
            "qa": signature([RUNS[k] / "clusters_dr11_qa.json" for k in PARTS])}
if up_to_date("maps", maps_sig):
    print("maps: up to date")
else:
    t0 = time.time()
    cols = ["RA", "DEC", "MASKBITS", "EBV"] + [f"NOBS_{b}" for b in BANDS] + \
           [f"GALDEPTH_{b}" for b in BANDS] + ["PSFSIZE_Z"]
    npix = hp.nside2npix(NSIDE)
    acc = {k: np.zeros(npix) for k in ["N_ALL", "N_OBS", "N_RUNOBS", "N_RUN", "EBV", "PSFSIZE_Z"]
           + [f"N_RUN_P{k}" for k in (0, 1)] + [f"DEPTH_{b}" for b in BANDS]}
    reject = np.int64(sum(1 << int(b) for b in CFG.survey.maskbits_reject))
    nread = 0
    for r in read_randoms_chunks(RANDOMS, cols):
        ra, dec = r["RA"].astype(np.float64), r["DEC"].astype(np.float64)
        pix = hp.ang2pix(NSIDE, ra, dec, lonlat=True, nest=True)
        observed = np.ones(ra.size, bool)
        for b in BANDS:
            observed &= (r[f"NOBS_{b}"] >= CFG.survey.nobs_min) & (r[f"GALDEPTH_{b}"] > 0)
        good = good_randoms(r, CFG)
        part = part_of(ra, dec)
        sky = (np.abs(galactic_latitude(ra, dec)) >= 15.0) & (r["EBV"] < CFG.survey.ebv_max) & (part >= 0)
        run = good & sky
        acc["N_ALL"] += np.bincount(pix, minlength=npix)
        acc["N_OBS"] += np.bincount(pix, weights=observed, minlength=npix)
        acc["N_RUNOBS"] += np.bincount(pix, weights=observed & sky, minlength=npix)
        acc["N_RUN"] += np.bincount(pix, weights=run, minlength=npix)
        for k in (0, 1):
            acc[f"N_RUN_P{k}"] += np.bincount(pix, weights=run & (part == k), minlength=npix)
        w = run.astype(float)
        for b in BANDS:
            m10 = 22.5 - 2.5 * np.log10(10.0 / np.sqrt(np.where(run, r[f"GALDEPTH_{b}"], 1.0)))
            m10 -= EXTINCTION_DECAM[b.lower()] * r["EBV"]
            acc[f"DEPTH_{b}"] += np.bincount(pix, weights=np.where(run, m10, 0.0), minlength=npix)
        acc["EBV"] += np.bincount(pix, weights=w * r["EBV"], minlength=npix)
        acc["PSFSIZE_Z"] += np.bincount(pix, weights=w * r["PSFSIZE_Z"], minlength=npix)
        nread += ra.size
        print(f"  {nread / 1e6:.0f} M randoms read ({time.time() - t0:.0f} s)", end="\r")
    print()
    keep = np.flatnonzero(acc["N_RUN"] > 0)
    n = acc["N_RUN"][keep]
    maps = {"PIX": keep.astype(np.int64), "N_ALL": acc["N_ALL"][keep], "N_RUN": n,
            "AREA": n / DENSITY,
            "MASKED_FRAC": 1.0 - n / np.maximum(acc["N_RUNOBS"][keep], 1),
            "EBV": acc["EBV"][keep] / n, "PSFSIZE_Z": acc["PSFSIZE_Z"][keep] / n}
    for k in (0, 1):
        maps[f"AREA_P{k}"] = acc[f"N_RUN_P{k}"][keep] / DENSITY
    for b in BANDS:
        maps[f"DEPTH_{b}10"] = acc[f"DEPTH_{b}"][keep] / n
    maps["ZVLIM"] = zvlim_of(maps["DEPTH_Z10"])
    # The 5 sigma depth (rema's S/N >= 5 cut in the reference band) for comparison.
    maps["ZVLIM5"] = zvlim_of(maps["DEPTH_Z10"] + 2.5 * np.log10(2.0))
    write_columns("maps", maps)
    area = {RUNS[k].name: float(maps[f"AREA_P{k}"].sum()) for k in PARTS}
    update_meta(maps=maps_sig, area=area, nside=NSIDE, density=DENSITY)
    print(f"maps: {keep.size} pixels, run area {sum(area.values()):.0f} deg2 {area} "
          f"({time.time() - t0:.0f} s)")

# %% [markdown]
# ### Clusters
#
# All clusters of the parts present, with the part, the region (`MEM_MATCH_ID >> 32`), the
# Galactic latitude, a flag for the calibration area (whose spectroscopic redshifts trained the
# red-sequence model), the distance to the seams between the parts (RA 0° and 240°, Dec −85°)
# and the depth-map values at the cluster position.

# %%
CL_COLS = ["RA", "DEC", "LAMBDA", "LAMBDA_E", "Z_LAMBDA", "Z_LAMBDA_E", "Z_LAMBDA_RAW", "R_LAMBDA",
           "SCALEVAL", "MASKFRAC", "NCENT_GOOD", "REFMAG", "ZRED", "ZRED_E", "ZRED_CHISQ", "MEM_MATCH_ID",
           "NSPEC", "N_MEMBERS", "SPEC_Z_BOOT", "SPEC_ZERR_BOOT", "VDISP_BOOT", "VDISP_ERR_BOOT",
           "VDISP_FLAG", "CG_SPEC_Z", "BEST_Z", "LNLAMLIKE", "P_CEN", "ID_CENT", "RA_CENT", "DEC_CENT",
           "PZBINS", "PZ"]


def seam_distance(ra, dec):
    """Angular distance [deg] to the seams of the two parts: RA = 0 and 240 deg, Dec = -85 deg."""
    d = np.full(ra.size, np.inf)
    for ra0 in (0.0, 240.0):
        dra = np.radians((ra - ra0 + 180.0) % 360.0 - 180.0)
        d = np.minimum(d, np.degrees(np.arcsin(np.minimum(1.0, np.abs(np.sin(dra)) * np.cos(np.radians(dec))))))
    return np.minimum(d, np.abs(dec + 85.0))


cl_sig = signature([RUNS[k] / "clusters_dr11.fits" for k in PARTS])
if up_to_date("clusters", {"cat": cl_sig, "maps": maps_sig}):
    print("clusters: up to date")
else:
    t0 = time.time()
    parts = []
    for k in PARTS:
        c = read_table(RUNS[k] / "clusters_dr11.fits", columns=CL_COLS, hdu="CLUSTERS")
        c["PART"] = np.full(c["RA"].size, k, np.int8)
        parts.append(c)
    cl = {c: np.concatenate([q[c] for q in parts]) for c in parts[0]}
    cl["ID_CG"] = cl.pop("ID_CENT")[:, 0]
    cl["REGION"] = (cl["MEM_MATCH_ID"] >> 32).astype(np.int32)
    cl["GLAT"] = galactic_latitude(cl["RA"], cl["DEC"]).astype(np.float32)
    cl["IN_CALIB"] = in_calib(cl["RA"], cl["DEC"])
    cl["SEAM_DIST"] = seam_distance(cl["RA"], cl["DEC"]).astype(np.float32)
    mp = load("maps")
    pix = hp.ang2pix(NSIDE, cl["RA"], cl["DEC"], lonlat=True, nest=True)
    j = np.searchsorted(mp["PIX"], pix).clip(0, mp["PIX"].size - 1)
    found = mp["PIX"][j] == pix
    for name in ("DEPTH_Z10", "ZVLIM", "ZVLIM5", "EBV", "PSFSIZE_Z", "MASKED_FRAC"):
        cl[name] = np.where(found, np.asarray(mp[name])[j], np.nan).astype(np.float32)
    cl["PIX"] = pix
    write_columns("clusters", cl)
    update_meta(clusters={"cat": cl_sig, "maps": maps_sig}, n_clusters=int(cl["RA"].size))
    print(f"clusters: {cl['RA'].size:,} ({time.time() - t0:.0f} s); "
          f"{(~found).sum()} outside the randoms map")

# %% [markdown]
# ### Members
#
# Members with P ≥ 0.05, plus every member with a spectroscopic redshift, with the row of
# their cluster and their colours (dereddened g−r, r−i, i−z; NaN when a flux is not positive).

# %%
MEM_COLS = ["MEM_MATCH_ID", "RA", "DEC", "Z", "R", "P", "PFREE", "REFMAG", "ZRED", "ZRED_E", "CHISQ",
            "ZSPEC", "VEL", "ISMEMBER_SPEC", "CENT_RANK", "FLUX"]
mem_sig = signature([RUNS[k] / "clusters_dr11_members.fits" for k in PARTS])
if up_to_date("members", {"mem": mem_sig, "cat": cl_sig}):
    print("members: up to date")
else:
    t0 = time.time()
    cl = load("clusters", ["MEM_MATCH_ID", "PART"])
    key_cl = (np.asarray(cl["PART"]).astype(np.int64) << 50) | np.asarray(cl["MEM_MATCH_ID"])
    order = np.argsort(key_cl)
    parts = []
    for k in PARTS:
        f = RUNS[k] / "clusters_dr11_members.fits"
        sel = read_table(f, columns=["P", "ZSPEC"], hdu="MEMBERS")
        rows = np.flatnonzero((sel["P"] >= 0.05) | (sel["ZSPEC"] > 0))
        m = read_table(f, columns=MEM_COLS, rows=rows, hdu="MEMBERS")
        m["PART"] = np.full(rows.size, k, np.int8)
        parts.append(m)
        print(f"  {f.name}: {rows.size:,} of {sel['P'].size:,} rows kept")
    mem = {c: np.concatenate([q[c] for q in parts]) for c in parts[0]}
    key = (mem["PART"].astype(np.int64) << 50) | mem["MEM_MATCH_ID"]
    j = np.searchsorted(key_cl, key, sorter=order)
    mem["CL_ROW"] = order[j.clip(0, order.size - 1)].astype(np.int32)
    assert np.all(key_cl[mem["CL_ROW"]] == key), "members without cluster"
    flux = mem.pop("FLUX").astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(flux > 0, 22.5 - 2.5 * np.log10(flux), np.nan)
    for i, name in enumerate(("GR", "RI", "IZ")):
        mem[name] = (mag[:, i] - mag[:, i + 1]).astype(np.float32)
    write_columns("members", mem)
    update_meta(members={"mem": mem_sig, "cat": cl_sig}, n_members=int(mem["P"].size))
    print(f"members: {mem['P'].size:,} ({time.time() - t0:.0f} s)")

# %% [markdown]
# ### External cluster catalogues
#
# Public catalogues, renamed to common columns: RA, DEC, Z (best redshift), and where they
# exist LAMBDA (redMaPPer-like richness), M500 [10¹⁴ M☉], LX [10⁴⁴ erg/s], centre positions
# and quality columns. Each is read once from `REMA_EXTERNAL` (sources in its README).

# %%
def read_cds(path, columns):
    """Fixed-width CDS table: columns = {name: (first byte, last byte)} (1-based, inclusive)."""
    import pandas as pd

    names = list(columns)
    spec = [(columns[c][0] - 1, columns[c][1]) for c in names]
    df = pd.read_fwf(path, colspecs=spec, names=names, header=None, compression="infer")
    return {c: df[c].to_numpy() for c in names}


def fits_cols(path, mapping, hdu=1):
    t = read_table(path, columns=list(mapping.values()), hdu=hdu)
    return {k: t[v.upper()] for k, v in mapping.items()}


ERO = Path.home() / "data" / "eromapper" / "data" / "external"
SOURCES = {
    "sdss_dr8": lambda: read_cds(EXTERNAL / "cat_dr8.dat", {
        "RA": (29, 39), "DEC": (41, 51), "Z": (53, 58), "Z_E": (60, 65), "LAMBDA": (67, 72),
        "LAMBDA_E": (74, 78), "S": (80, 84), "ZSPEC": (86, 93), "P_CEN0": (202, 210)}),
    "des_y1": lambda: fits_cols(Path.home() / "data" / "ACT" / "redmapper_y1a1_public_v6.4_catalog.fits", {
        "RA": "RA", "DEC": "DEC", "Z": "Z_LAMBDA", "Z_E": "Z_LAMBDA_ERR", "LAMBDA": "LAMBDA",
        "LAMBDA_E": "LAMBDA_ERR", "S": "S", "ZSPEC": "Z_SPEC"}),
    "erass1": lambda: {**fits_cols(EXTERNAL / "erass1cl_primary_v3.2.fits", {
        "RA": "RA", "DEC": "DEC", "Z": "BEST_Z", "PCONT": "PCONT", "M500": "M500", "LX": "L500",
        "EXT_LIKE": "EXT_LIKE"}), **fits_cols(EXTERNAL / "eRASS1_clusters_optical.fits", {
        "RA_OPT": "RA_OPT", "DEC_OPT": "DEC_OPT", "RA_BCG": "RA_BCG", "DEC_BCG": "DEC_BCG",
        "LAMBDA": "LAMBDA_NORM", "Z_LAMBDA": "Z_LAMBDA", "BEST_Z_TYPE": "BEST_Z_TYPE",
        "IN_ZVLIM": "IN_ZVLIM", "ZVLIM_02": "ZVLIM_02", "LIMMAG_Z": "LIMMAG_Z",
        "IN_FOOTPRINT": "IN_FOOTPRINT"})},
    "act_dr6": lambda: fits_cols(EXTERNAL / "DR6_cluster-catalog_v1.0.fits", {
        "RA": "RADeg", "DEC": "decDeg", "Z": "redshift", "M500": "M500c", "SNR": "SNR",
        "RA_OPT": "opt_RADeg", "DEC_OPT": "opt_decDeg"}),
    "act_dr5": lambda: fits_cols(ERO / "ACT_DR5_cluster-catalog_v1.1.fits", {
        "RA": "ra", "DEC": "dec", "Z": "redshift", "M500": "M500cCal", "SNR": "SNR"}),
    "spt_2500d": lambda: fits_cols(ERO / "2500d_cluster_sample_Bocquet19.fits", {
        "RA": "RA", "DEC": "DEC", "Z": "REDSHIFT", "M500": "M500", "SNR": "XI"}),
    "mcxc": lambda: fits_cols(ERO / "mcxc.fits", {
        "RA": "RA", "DEC": "DEC", "Z": "REDSHIFT", "M500": "MASS_500", "LX": "LX_500"}),
    "psz2": lambda: fits_cols(EXTERNAL / "HFI_PCCS_SZ-union_R2.08.fits.gz", {
        "RA": "RA", "DEC": "DEC", "Z": "REDSHIFT", "M500": "MSZ", "SNR": "SNR"}),
    "wen_han_2024": lambda: read_cds(EXTERNAL / "wenhan2024_table2.dat.gz", {
        "RA": (29, 37), "DEC": (39, 47), "Z": (50, 55), "ZSPEC_FLAG": (58, 58), "LAMBDA": (88, 93),
        "M500": (96, 100)}),
}

# Units: M500 in 10^14 Msun, LX in 10^44 erg/s.
SCALE = {"erass1": {"M500": 0.1, "LX": 0.01}, "mcxc": {"M500": 1e-14, "LX": 1e-44}}

for name, reader in SOURCES.items():
    d = WORK / "external" / name
    if d.exists() and not FORCE:
        continue
    try:
        cols = reader()
    except (FileNotFoundError, OSError, ImportError) as e:      # ImportError: pandas, for the CDS tables
        print(f"{name}: not available ({e.__class__.__name__}), skipped")
        continue
    cols = {k: np.asarray(v) for k, v in cols.items()}
    cols = {k: (v.astype(np.float64) if v.dtype.kind in "fiu" and k != "BEST_Z_TYPE" else v)
            for k, v in cols.items()}
    for k, f in SCALE.get(name, {}).items():
        cols[k] = cols[k] * f
    cols["Z"] = np.where(np.isfinite(cols["Z"]) & (cols["Z"] > 0), cols["Z"], np.nan)
    write_columns(f"external/{name}", cols)
    print(f"{name}: {cols['RA'].size:,} clusters")
