"""Tier C: how much the calibration changes with the cosmology it is made in. Compares the
calibrations in $REMA_COSMO/tierC/<cosmology>/calib.fits with the fiducial one and writes
calib_table.rst (the largest and median changes over 0.1 < z < 0.8)."""

import numpy as np
from astropy.io import fits

from common import HERE, RESULTS, need, record

COSMOS = ["Omega_m=0.25", "Omega_m=0.35"]
ROWS = [("RS_MEAN", "VALUE", "red-sequence mean colours [mag]", 1.0),
        ("RS_SLOPE", "VALUE", "red-sequence slopes", 1.0),
        ("RS_LOG_SIGMA", "VALUE", "intrinsic scatter [relative]", 1.0),
        ("RS_PIVOT", "VALUE", "pivot magnitude [mag]", 1.0),
        ("ZREDCORR", "CORR", "zred correction", 1.0),
        ("ZLAMBDACORR", "OFFSET", "z_λ correction: offset", 1.0),
        ("ZLAMBDACORR", "SCATTER", "z_λ correction: scatter", 1.0)]
paths = {c: RESULTS / "tierC" / c / "calib.fits" for c in ["fiducial"] + COSMOS}
if need(*paths.values()):
    h = {c: fits.open(p) for c, p in paths.items()}
    f = h["fiducial"]
    lines = [".. list-table::", "   :header-rows: 1", "", "   * - re-calibrated in"]
    lines += [f"     - {c.replace('Omega_m', 'Ω_m')}: largest (median) change" for c in COSMOS]
    for ext, col, label, _ in ROWS:
        z = np.asarray(f[ext].data["Z"], float)
        sel = (z >= 0.1) & (z <= 0.8)
        ref = np.asarray(f[ext].data[col], float)
        cells = []
        for c in COSMOS:
            d = np.abs(np.asarray(h[c][ext].data[col], float) - ref)[sel]
            cells.append(f"{d.max():.4f} ({np.median(d):.4f})")
            record(f"calib_{ext}_{col}_{c}_max", float(d.max()))
        lines += [f"   * - {label}"] + [f"     - {x}" for x in cells]
    nc = [int(h[c][0].header["NCLUSTER"]) for c in ["fiducial"] + COSMOS]
    lines += ["   * - calibration clusters (fiducial: {:,})".format(nc[0])] + [f"     - {n:,}" for n in nc[1:]]
    zl = [float(h[c][0].header["ZLNMAD"]) for c in ["fiducial"] + COSMOS]
    lines += [f"   * - z_λ against spectroscopic z, NMAD (fiducial: {zl[0]:.4f})"] + [f"     - {v:.4f}" for v in zl[1:]]
    wf = h["fiducial"]["WCEN"].data
    wmax = [max(abs(float(h[c]["WCEN"].data[n][0]) - float(wf[n][0])) for n in wf.columns.names) for c in COSMOS]
    lines += ["   * - centring model: largest change of a parameter"] + [f"     - {v:.3f}" for v in wmax]
    (HERE / "calib_table.rst").write_text("\n".join(lines + [""]) + "\n")
    print("wrote calib_table.rst")
