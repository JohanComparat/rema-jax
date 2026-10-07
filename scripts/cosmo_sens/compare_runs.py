"""Tiers B and C: compare the blind re-runs in other cosmologies with the fiducial re-run.

    python scripts/cosmo_sens/compare_runs.py OUTDIR/tierB [--response RESPONSE.fits] [--out B.json]

DIR holds one directory per cosmology (``fiducial``, ``Omega_m=0.25``, ...) with one
``NNNN/clusters.fits`` per region (``cosmo_sens.sh``). For each cosmology and region: the
changes of matched clusters (rema.validate.compare.rerun_summary), and, with ``--response``, the
ratio of the counts above each lambda edge in the re-run to the fiducial counts with lambda
shifted by the tier-A response, whose log divided by the parameter step is the selection
correction c(lambda, z) of the counts model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cosmo_common as C  # noqa: E402

LAM_EDGES = [10.0, 20.0, 30.0, 45.0, 60.0]


def main(argv=None):
    from rema.abundance.response import ResponseTable, delta_lnlam
    from rema.config import CosmologyConfig, parse_cosmology_overrides
    from rema.io.tables import read_catalog
    from rema.validate.compare import rerun_summary

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dir")
    p.add_argument("--response")
    p.add_argument("--out")
    a = p.parse_args(argv)
    d = Path(a.dir)
    resp = ResponseTable.read(a.response) if a.response else None
    fid = {r.parent.name: r for r in sorted((d / "fiducial").glob("*/clusters.fits"))}
    out = {"lam_edges": LAM_EDGES, "z_edges": C.Z_EDGES, "runs": []}
    for cdir in sorted(x for x in d.iterdir() if x.is_dir() and x.name != "fiducial"):
        theta = parse_cosmology_overrides([cdir.name])
        for rf in sorted(cdir.glob("*/clusters.fits")):
            rid = rf.parent.name
            if rid not in fid:
                continue
            ca, ma, _ = read_catalog(fid[rid])
            cb, mb, _ = read_catalog(rf)
            s = rerun_summary(ca, cb, LAM_EDGES, C.Z_EDGES, lam_min=20.0, mem_a=ma, mem_b=mb)
            s.update(cosmology=cdir.name, region=rid)
            if resp is not None:
                dth = np.asarray(resp.dtheta({**CosmologyConfig().__dict__, **theta}))
                shifted = np.exp(np.log(ca["LAMBDA"]) + np.asarray(delta_lnlam(resp, np.log(ca["LAMBDA"]),
                                                                             ca["Z_LAMBDA"], dth)))
                ze = np.asarray(C.Z_EDGES)
                ratio = np.full((len(LAM_EDGES), ze.size - 1), np.nan)
                for k, lmin in enumerate(LAM_EDGES):
                    for m in range(ze.size - 1):
                        na = np.sum((shifted >= lmin) & (ca["Z_LAMBDA"] >= ze[m]) & (ca["Z_LAMBDA"] < ze[m + 1]))
                        nb = np.sum((cb["LAMBDA"] >= lmin) & (cb["Z_LAMBDA"] >= ze[m]) & (cb["Z_LAMBDA"] < ze[m + 1]))
                        ratio[k, m] = nb / na if na else np.nan
                s["ncum_ratio_vs_tierA"] = ratio.tolist()
                step = sum(v - getattr(CosmologyConfig(), k) for k, v in theta.items())
                s["selection_c"] = (np.log(ratio) / step).tolist()
            out["runs"].append(s)
            print(f"{cdir.name:14s} region {rid}: matched {s['n_matched']}, d ln lambda {s['dlnlam_median']:+.4f} "
                  f"(NMAD {s['dlnlam_nmad']:.4f}), dz {s['dz_median']:+.5f}, centre changed {s['centre_changed']:.3f}, "
                  f"lost {s['lost']}, gained {s['gained']}")
    path = Path(a.out) if a.out else d / "compare.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", path)


if __name__ == "__main__":
    main()
