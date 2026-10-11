"""Inputs for rema scan-positions from the public cluster catalogues kept with the DR11 products.

    python scripts/scan/public_catalogues.py OUTDIR [--external DIR] [--dr11 DIR]
        [--only act_dr6 spt_2500d ...] [--sample N] [--seed 1] [--split-above 200000 --split-ra 15]

Writes ``OUTDIR/<name>.fits`` per catalogue: RA, DEC, Z and the catalogue's own mass, richness or
S/N columns under common names (the loaders of docs/figures/redmapper_dr11/prepare.py), plus
``ROW``, the row in the original file. ``--sample N`` keeps N random clusters whose position falls
in a DR11 south sweep (a quick validation set); without it every cluster is kept (the ones
outside the sweeps come back unscanned). A catalogue of more than ``--split-above`` rows is written
as slices of ``--split-ra`` deg in RA, ``<name>_ra<RA0>.fits``, one scan job each; join their
outputs with ``scripts/scan/merge_parts.py``.

Catalogues (sources in ``EXTERNAL/README.txt``): act_dr6, act_dr5, spt_2500d, psz2, mcxc,
erass1, sdss_dr8, des_y1, wen_han_2024.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np

CC = Path("/sps/lsst/datasets/desi/legacysurveys/dr11/south")


def read_cds(path, columns):
    """Fixed-width CDS table: columns = {name: (first byte, last byte)} (1-based, inclusive)."""
    import pandas as pd

    names = list(columns)
    spec = [(columns[c][0] - 1, columns[c][1]) for c in names]
    df = pd.read_fwf(path, colspecs=spec, names=names, header=None, compression="infer")
    return {c: df[c].to_numpy() for c in names}


def fits_cols(path, mapping, hdu=1):
    from astropy.table import Table

    t = Table.read(path, hdu=hdu)
    up = {c.upper(): c for c in t.colnames}
    return {k: np.asarray(t[up[v.upper()]]) for k, v in mapping.items()}


def sources(ext: Path):
    return {
        "act_dr6": lambda: fits_cols(ext / "DR6_cluster-catalog_v1.0.fits", {
            "NAME": "name", "RA": "RADeg", "DEC": "decDeg", "Z": "redshift", "M500": "M500c",
            "SNR": "SNR", "RA_OPT": "opt_RADeg", "DEC_OPT": "opt_decDeg"}),
        "act_dr5": lambda: fits_cols(ext / "ACT_DR5_cluster-catalog_v1.1.fits", {
            "NAME": "name", "RA": "ra", "DEC": "dec", "Z": "redshift", "M500": "M500cCal",
            "SNR": "SNR"}),
        "spt_2500d": lambda: fits_cols(ext / "2500d_cluster_sample_Bocquet19.fits", {
            "NAME": "SPT_ID", "RA": "RA", "DEC": "DEC", "Z": "REDSHIFT", "M500": "M500", "SNR": "XI"}),
        "psz2": lambda: fits_cols(ext / "HFI_PCCS_SZ-union_R2.08.fits.gz", {
            "NAME": "NAME", "RA": "RA", "DEC": "DEC", "Z": "REDSHIFT", "M500": "MSZ", "SNR": "SNR"}),
        "mcxc": lambda: fits_cols(ext / "mcxc.fits", {
            "NAME": "NAME", "RA": "RA", "DEC": "DEC", "Z": "REDSHIFT", "M500": "MASS_500",
            "LX": "LX_500"}),
        "erass1": lambda: {**fits_cols(ext / "erass1cl_primary_v3.2.fits", {
            "NAME": "NAME", "RA": "RA", "DEC": "DEC", "Z": "BEST_Z", "PCONT": "PCONT",
            "M500": "M500", "LX": "L500", "EXT_LIKE": "EXT_LIKE"}),
            **fits_cols(ext / "eRASS1_clusters_optical.fits", {
                "LAMBDA": "LAMBDA_NORM", "Z_LAMBDA": "Z_LAMBDA", "RA_OPT": "RA_OPT",
                "DEC_OPT": "DEC_OPT"})},
        "sdss_dr8": lambda: read_cds(ext / "cat_dr8.dat", {
            "RA": (29, 39), "DEC": (41, 51), "Z": (53, 58), "Z_E": (60, 65), "LAMBDA": (67, 72),
            "LAMBDA_E": (74, 78), "S": (80, 84), "ZSPEC": (86, 93)}),
        "des_y1": lambda: fits_cols(ext / "redmapper_y1a1_public_v6.4_catalog.fits", {
            "RA": "RA", "DEC": "DEC", "Z": "Z_LAMBDA", "Z_E": "Z_LAMBDA_ERR", "LAMBDA": "LAMBDA",
            "LAMBDA_E": "LAMBDA_ERR", "ZSPEC": "Z_SPEC"}),
        "wen_han_2024": lambda: read_cds(ext / "wenhan2024_table2.dat.gz", {
            "RA": (29, 37), "DEC": (39, 47), "Z": (50, 55), "ZSPEC_FLAG": (58, 58),
            "LAMBDA": (88, 93), "M500": (96, 100)}),
    }


# Units: M500 in 10^14 Msun, LX in 10^44 erg/s.
SCALE = {"erass1": {"M500": 0.1, "LX": 0.01}, "mcxc": {"M500": 1e-14, "LX": 1e-44}}


def in_sweeps(ra, dec, sweep_dir: Path) -> np.ndarray:
    """Positions inside one of the sweep files of ``sweep_dir`` (sweeps tile the sky in 5 deg)."""
    from rema.sky.regions import sweep_box

    names = [p.name for p in sweep_dir.glob("sweep-*.fits") if not p.name.endswith("-pz.fits")]
    boxes = [sweep_box(n) for n in names]
    keep = np.zeros(np.size(ra), bool)
    for b in boxes:
        keep |= b.contains(ra, dec)
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir")
    dr11 = os.environ.get("DR11") or (CC if CC.exists() else None)
    ap.add_argument("--dr11", default=dr11)
    ap.add_argument("--external", help="default DR11/rema/external")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--sample", type=int, help="N random clusters in the DR11 south sweeps")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split-above", type=int, default=200_000, help="rows above which to split in RA")
    ap.add_argument("--split-ra", type=float, default=15.0, help="deg of RA per slice")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", message=".*did not parse as fits unit")
    from rema.io.tables import write_table

    ext = Path(a.external or Path(a.dr11) / "rema" / "external")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for name, reader in sources(ext).items():
        if a.only and name not in a.only:
            continue
        try:
            cols = reader()
        except (FileNotFoundError, OSError, ImportError, KeyError) as e:
            print(f"{name}: not available ({e.__class__.__name__}: {e}), skipped")
            continue
        cols = {k: np.asarray(v) for k, v in cols.items()}
        for k, v in list(cols.items()):
            if v.dtype.kind in "fiu":
                cols[k] = v.astype(np.float64)
            elif v.dtype.kind in "SUO":
                cols[k] = np.char.strip(v.astype(str))
        for k, f in SCALE.get(name, {}).items():
            cols[k] = cols[k] * f
        cols["Z"] = np.where(np.isfinite(cols["Z"]) & (cols["Z"] > 0), cols["Z"], np.nan)
        cols["ROW"] = np.arange(cols["RA"].size)
        n = cols["RA"].size
        if a.sample:
            good = np.flatnonzero(in_sweeps(cols["RA"], cols["DEC"], Path(a.dr11) / "sweep" / "11.0"))
            pick = np.sort(rng.choice(good, min(a.sample, good.size), replace=False))
            cols = {k: v[pick] for k, v in cols.items()}
        if cols["RA"].size > a.split_above:
            sl = np.floor(np.mod(cols["RA"], 360.0) / a.split_ra).astype(int)
            for k in np.unique(sl):
                f = out / f"{name}_ra{k * a.split_ra:05.1f}.fits"
                write_table(f, {c: v[sl == k] for c, v in cols.items()}, extname="POSITIONS")
            print(f"{name}: {cols['RA'].size:,} of {n:,} clusters -> {np.unique(sl).size} RA slices "
                  f"{out / (name + '_ra*.fits')}")
            continue
        write_table(out / f"{name}.fits", cols, extname="POSITIONS")
        print(f"{name}: {cols['RA'].size:,} of {n:,} clusters -> {out / (name + '.fits')}")


if __name__ == "__main__":
    sys.exit(main())
