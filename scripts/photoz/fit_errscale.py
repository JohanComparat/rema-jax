"""Photo-z width calibration of a galaxy table: the photoz section of a configuration overlay.

    python scripts/photoz/fit_errscale.py GALAXIES [--edges 14 19 20 21 22 24] [--out overlay.yaml]

GALAXIES is a galaxy table or a directory of per-sweep tables (with --box). Only the galaxies with
a spectroscopic redshift (ZSPEC > 0) and a photo-z enter. Prints the per-bin statistics (scale =
NMAD of (ZPHOT - ZSPEC)/ZPHOT_STD, NMAD and outliers of dz/(1+z), bias), split into red and blue
galaxies when the table has ZRED_CHISQ, and writes the YAML overlay for ``--set @overlay.yaml``.
"""

import argparse
import json
import sys

import numpy as np

from rema.io.legacy import read_galaxies
from rema.sky.regions import Box, sky_union
from rema.validate.photoz import fit_err_scale, yaml_overlay


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("galaxies")
    ap.add_argument("--box", nargs=4, type=float, action="append")
    ap.add_argument("--edges", nargs="+", type=float, default=[14, 19, 20, 21, 22, 24])
    ap.add_argument("--out", help="YAML overlay")
    ap.add_argument("--json", help="the fit, as JSON")
    a = ap.parse_args(argv)
    sky = sky_union([Box(*b) for b in a.box]) if a.box else None
    cols = ("ZPHOT", "ZPHOT_STD", "ZSPEC", "REFMAG")
    g = read_galaxies(a.galaxies, sky)
    spec = np.asarray(g["ZSPEC"]) > 0
    print(f"{spec.sum()} spectroscopic galaxies of {spec.size}")
    fit = fit_err_scale(*(np.asarray(g[c])[spec] for c in cols), edges=a.edges)
    for b in fit["bins"]:
        print(f"  {b['lo']:5.1f}-{b['hi']:4.1f}  n {b['n']:6d}  scale {b['scale']:.3f}  "
              f"NMAD dz/(1+z) {b['nmad_dz']:.4f}  outliers {b['outliers']:.3f}  bias {b['bias']:+.4f}")
    if "ZRED_CHISQ" in g:
        red = np.asarray(g["ZRED_CHISQ"])[spec] < 20
        for name, sel in (("red", red), ("blue", ~red)):
            f = fit_err_scale(*(np.asarray(g[c])[spec][sel] for c in cols), edges=a.edges)
            print(f"  {name}: " + ", ".join(f"{b['mag']:.1f}: {b['scale']:.2f}" for b in f["bins"]))
    text = yaml_overlay(fit)
    print(text, end="")
    if a.out:
        open(a.out, "w").write(text)
    if a.json:
        json.dump(fit, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
