"""Join the rema scan-positions outputs of the slices of one input table (public_catalogues.py
--split-above) into one catalogue.

    python scripts/scan/merge_parts.py OUT.fits PART.fits [PART.fits ...] [--row-col INPUT_ROW]

The clusters are sorted by ``--row-col`` (the row in the original table) and get MEM_MATCH_ID =
that row; the members follow. The primary header and CONFIG HDU are those of the first part, with
NPOS, NSCANNED, NFOUND summed and TTOTAL the total time of the parts.
"""

import argparse
import sys

import numpy as np
from astropy.io import fits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--row-col", default="INPUT_ROW")
    a = ap.parse_args(argv)
    from rema.io.tables import read_catalog, table_hdu, write_fits

    cats, mems, hdr0, config = [], [], None, None
    tot = dict.fromkeys(("NPOS", "NSCANNED", "NFOUND", "TTOTAL"), 0)
    for p in sorted(a.parts):
        cat, mem, hdr = read_catalog(p)
        row = np.asarray(cat[a.row_col], np.int64)
        if mem:
            mem = dict(mem)
            mem["MEM_MATCH_ID"] = row[np.asarray(mem["MEM_MATCH_ID"], np.int64)]
            mems.append(mem)
        cat = dict(cat)
        cat["MEM_MATCH_ID"] = row
        cats.append(cat)
        for k in tot:
            tot[k] += hdr.get(k, 0)
        if hdr0 is None:
            hdr0 = hdr
            with fits.open(p) as h:
                config = h["CONFIG"].copy() if "CONFIG" in h else None

    def concat(parts):
        keys = [k for k in parts[0] if all(k in q for q in parts)]
        return {k: np.concatenate([np.asarray(q[k]) for q in parts]) for k in keys}

    cat = concat(cats)
    order = np.argsort(cat["MEM_MATCH_ID"], kind="stable")
    cat = {k: v[order] for k, v in cat.items()}
    if np.unique(cat["MEM_MATCH_ID"]).size != cat["MEM_MATCH_ID"].size:
        raise SystemExit(f"{a.row_col} repeats across the parts")
    mem = concat(mems) if mems else {}
    if mem:
        order = np.argsort(mem["MEM_MATCH_ID"], kind="stable")
        mem = {k: v[order] for k, v in mem.items()}
    for k in ("NAXIS", "EXTEND", "SIMPLE", "BITPIX"):
        hdr0.remove(k, ignore_missing=True)
    hdr0.update(tot)
    hdr0["NCLUSTER"] = cat["MEM_MATCH_ID"].size
    hdr0["NMEMBER"] = mem["MEM_MATCH_ID"].size if mem else 0
    hdr0["NPARTS"] = len(a.parts)
    hdus = [table_hdu(cat, extname="CLUSTERS"), table_hdu(mem, extname="MEMBERS")]
    if config is not None:
        hdus.append(config)
    write_fits(a.out, hdus, dict(hdr0))
    print(f"{a.out}: {hdr0['NCLUSTER']:,} rows ({tot['NSCANNED']:,} scanned, {tot['NFOUND']:,} with "
          f"lambda > 0), {hdr0['NMEMBER']:,} members from {len(a.parts)} parts")


if __name__ == "__main__":
    sys.exit(main())
