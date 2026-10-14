"""Galaxy tracer of the cluster x galaxy cross-correlation from the LS DR11 r < 21 sample of
make-sample-ls11 (tag ``bgsr21``: every object with 13 < r <= 21, dereddened, in the footprint
NOBS / GALDEPTH / MASKBITS 0, 1, 11, 12, 13 / E(B-V) < 0.1; its randoms are
``<LS11_OUT>/bgsr21/LS11_bgsr21_RAND.fits``).

Reads every per-sweep selection file with :func:`ls11samples.catalog.load_sweep` (positions and
g, r, z fluxes back from the sweeps), leaves out the stars (STAR_FLAG & ``--star-flag``: bit 0
TYPE = PSF, bit 1 Gaia G - r_raw <= 0.6) and writes RA, DEC, MAG_G, MAG_R, MAG_Z, BEST_Z,
BEST_Z_ERR, Z_SOURCE, keeping MAG_R <= ``--rmax``. Run with the make-sample-ls11 environment
(``source cc_in2p3/env_ccin2p3.sh``).

    python scripts/randoms/extract_ls11_tracer.py OUT.fits [--rmax 21] [--tag bgsr21]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--tag", default="bgsr21")
    ap.add_argument("--rmax", type=float, default=21.0)
    ap.add_argument("--star-flag", type=int, default=3)
    a = ap.parse_args(argv)

    from astropy.io import fits
    from ls11samples.catalog import load_sweep
    from ls11samples.env import get_paths

    paths = get_paths()
    sweeps = [s for s in paths.sweeps() if paths.product(s, a.tag).exists()]
    cols = ("RA", "DEC", "MAG_G", "MAG_R", "MAG_Z", "BEST_Z", "BEST_Z_ERR", "Z_SOURCE")
    parts, n_in = {c: [] for c in cols}, 0
    for i, s in enumerate(sweeps):
        t = load_sweep(paths, s, a.tag, sweep_columns=("RA", "DEC"), mag_bands=("G", "R", "Z"))
        n_in += len(t["RA"])
        keep = ((np.asarray(t["STAR_FLAG"]) & a.star_flag) == 0) & (t["MAG_R"] <= a.rmax)
        for c in cols:
            parts[c].append(np.asarray(t[c])[keep])
        if i % 100 == 0:
            print(f"{i}/{len(sweeps)} sweeps, {sum(p.size for p in parts['RA'])} of {n_in} kept", flush=True)
    out = {c: np.concatenate(v) for c, v in parts.items()}
    fmt = {"RA": "D", "DEC": "D", "Z_SOURCE": "I"}
    hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name=c, format=fmt.get(c, "E"), array=out[c].astype(np.int16 if c == "Z_SOURCE" else
                                                                          (np.float64 if c in fmt else np.float32)))
         for c in cols], name="TRACER")
    hdu.header.update(TAG=a.tag, RMAX=a.rmax, STARFLAG=a.star_flag, NIN=n_in, NSWEEP=len(sweeps))
    hdu.writeto(a.out, overwrite=True)
    print(f"wrote {a.out}: {out['RA'].size} of {n_in} objects from {len(sweeps)} sweeps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
