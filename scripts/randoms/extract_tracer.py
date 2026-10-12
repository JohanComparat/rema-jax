"""Galaxy tracer of the cluster x galaxy cross-correlation, from the per-sweep galaxy tables of
``rema ingest --outdir`` (DR11 south: MASKBITS, NOBS, TYPE != PSF, z-band S/N >= 5, E(B-V) < 0.2).

Keeps the galaxies brighter than ``lstar`` L* at their photo-z (REFMAG < m*(ZPHOT) -
2.5 log10(lstar), the m*(z) of the catalogue's configuration) with zmin <= ZPHOT < zmax, and
writes RA, DEC, ZPHOT, ZPHOT_STD, REFMAG, ZSPEC and EBV to one FITS table. The tracer's volume
limit (where a galaxy of ``lstar`` L* is detected) is applied later with a z_vlim map made with
the same ``lstar``, to galaxies and randoms alike.

    python scripts/randoms/extract_tracer.py GALAXY_DIR OUT.fits --config catalog_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("galaxies", help="directory of per-sweep galaxy tables")
    ap.add_argument("out")
    ap.add_argument("--config", help="configuration (m*(z) model); default rema's")
    ap.add_argument("--lstar", type=float, default=0.5)
    ap.add_argument("--zmin", type=float, default=0.05)
    ap.add_argument("--zmax", type=float, default=0.95)
    a = ap.parse_args(argv)

    from rema.config import RemaConfig
    from rema.io.tables import read_table, write_table
    from rema.model.profiles import MStar, maxmag_from_mstar

    cfg = RemaConfig.from_yaml(a.config) if a.config else RemaConfig()
    ms = MStar(cfg.model.mstar)
    zt, mt = np.asarray(ms.z), np.asarray(maxmag_from_mstar(np.asarray(ms.m), a.lstar))
    cols = ["RA", "DEC", "ZPHOT", "ZPHOT_STD", "REFMAG", "ZSPEC", "EBV"]
    parts, n_in = {c: [] for c in cols}, 0
    files = sorted(Path(a.galaxies).glob("sweep-*.fits"))
    for i, f in enumerate(files):
        t = read_table(f, columns=cols)
        n_in += t["RA"].size
        z = np.asarray(t["ZPHOT"], np.float64)
        keep = (z >= a.zmin) & (z < a.zmax) & (t["ZPHOT_STD"] > 0)
        keep &= t["REFMAG"] < np.interp(z, zt, mt)
        for c in cols:
            parts[c].append(np.asarray(t[c])[keep])
        if i % 100 == 0:
            print(f"{i}/{len(files)} sweeps, {sum(p.size for p in parts['RA'])} of {n_in} galaxies kept", flush=True)
    out = {c: np.concatenate(v) for c, v in parts.items()}
    write_table(a.out, out, header={"LSTAR": a.lstar, "ZMIN": a.zmin, "ZMAX": a.zmax, "MSTAR": cfg.model.mstar,
                                    "NIN": n_in, "NSWEEP": len(files)}, extname="TRACER")
    print(f"wrote {a.out}: {out['RA'].size} of {n_in} galaxies from {len(files)} sweeps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
