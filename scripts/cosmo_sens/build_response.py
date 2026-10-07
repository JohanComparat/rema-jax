"""Response table of the finder (rema.abundance.response) from ``rema remeasure`` files.

    python scripts/cosmo_sens/build_response.py REMEASURE.fits [...] --out RESPONSE.fits
        [--z-edges 0.05,0.2,0.35,0.5,0.7,0.95] [--lam-edges 10,20,40,300] [--min-count 15]
        [--degree 2,1]

Prints the binned medians, their NMAD and the counts per bin, and writes the table and a JSON
summary next to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    from rema.abundance.response import from_remeasure

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("files", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--z-edges", default="0.05,0.2,0.35,0.5,0.7,0.95")
    p.add_argument("--lam-edges", default="10,20,40,300")
    p.add_argument("--min-count", type=int, default=15)
    p.add_argument("--degree", default="2,1", help="polynomial degrees in z and ln lambda")
    a = p.parse_args(argv)
    ze = [float(x) for x in a.z_edges.split(",")]
    le = [float(x) for x in a.lam_edges.split(",")]
    deg = tuple(int(x) for x in a.degree.split(","))
    tab, st = from_remeasure(a.files, ze, le, min_count=a.min_count, degree=deg)
    tab.write(a.out, header={"NFILES": len(a.files)})
    summary = {"files": [str(f) for f in a.files], "params": list(tab.params), "z_edges": ze,
               "lam_edges": le, "N": st["N"].tolist()}
    for k, par in enumerate(tab.params):
        summary[par] = {key: np.where(np.isfinite(st[key][k]), st[key][k], None).tolist()
                        for key in ("D1", "D2", "DZ", "D1_NMAD")}
        summary[par]["D1_SMOOTH"] = np.asarray(tab.d1[k]).tolist()
        print(f"{par}: d ln(lambda)/d{par} (rows z, columns lambda)\n{np.round(st['D1'][k], 4)}")
    print("clusters per bin\n", st["N"])
    Path(a.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
