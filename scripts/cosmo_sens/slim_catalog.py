"""The clusters and members of a catalogue in some boxes, with the columns ``rema remeasure``
reads (a small copy of the combined DR11 catalogue for the study regions).

    python scripts/cosmo_sens/slim_catalog.py CLUSTERS.fits MEMBERS.fits --box RA0 RA1 DEC0 DEC1 [...]
        --out-clusters OUT.fits --out-members OUT_MEMBERS.fits
"""

from __future__ import annotations

import argparse

import numpy as np

CLUSTER_COLUMNS = ("MEM_MATCH_ID", "SEED_ID", "ID_CENT", "RA", "DEC", "LNLIKE", "LAMBDA", "LAMBDA_E",
                   "Z_LAMBDA", "Z_LAMBDA_E", "Z_LAMBDA_RAW", "R_LAMBDA", "SCALEVAL", "MASKFRAC", "P_CEN")
MEMBER_COLUMNS = ("MEM_MATCH_ID", "ID", "P", "PMEM", "PFREE", "R")


def main(argv=None):
    from astropy.io import fits

    from rema.io.tables import read_table, write_table
    from rema.sky.regions import Box, sky_union

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("clusters")
    p.add_argument("members")
    p.add_argument("--box", nargs=4, type=float, action="append", required=True)
    p.add_argument("--out-clusters", required=True)
    p.add_argument("--out-members", required=True)
    a = p.parse_args(argv)
    sky = sky_union([Box(*b) for b in a.box])
    pos = read_table(a.clusters, ["RA", "DEC"], hdu="CLUSTERS")
    rows = np.flatnonzero(sky.contains(pos["RA"], pos["DEC"]))
    with fits.open(a.clusters, memmap=True) as h:
        names = set(h["CLUSTERS"].columns.names)
        prim = {k: v for k, v in h[0].header.items() if k not in ("SIMPLE", "BITPIX", "NAXIS", "EXTEND")}
    cat = read_table(a.clusters, [c for c in CLUSTER_COLUMNS if c in names], rows=rows, hdu="CLUSTERS")
    write_table(a.out_clusters, cat, extname="CLUSTERS", primary_header={**prim, "NCLUSTER": rows.size})
    mm = read_table(a.members, ["MEM_MATCH_ID"], hdu="MEMBERS")["MEM_MATCH_ID"]
    mrows = np.flatnonzero(np.isin(mm, cat["MEM_MATCH_ID"]))
    mem = read_table(a.members, list(MEMBER_COLUMNS), rows=mrows, hdu="MEMBERS")
    write_table(a.out_members, mem, extname="MEMBERS", primary_header={"NMEMBER": mrows.size})
    print(f"{rows.size} clusters, {mrows.size} members")


if __name__ == "__main__":
    main()
