"""Write the tables of docs/cosmology_sensitivity.rst from the results in $REMA_WORK/cosmo_sens:
fit_table.rst (fit_<tag>.json) and forecast_table.rst (forecast_<variant>.json)."""

import json
import os

from common import HERE, RESULTS

FITS = [("all_cc", "free"), ("all_cc_sig025", "0.25 (fixed)")]
FC = os.environ.get("COSMO_FORECAST", "sig025_cc")
VARIANTS = [(FC, "free", "fixed (catalogue)"),
            (f"{FC}_fixednorm", "normalisation fixed", "fixed (catalogue)"),
            (f"{FC}_fixedmor", "fixed", "fixed (catalogue)"),
            (f"{FC}_mstar", "free", "follows D_L"),
            (f"{FC}_mstar_fixednorm", "normalisation fixed", "follows D_L"),
            (f"{FC}_mstar_fixedmor", "fixed", "follows D_L")]
OFFSETS = [("Omega_m", -0.05, "ΔΩ_m = −0.05"), ("Omega_m", 0.05, "ΔΩ_m = +0.05"),
           ("w0", -0.2, "Δw0 = −0.2"), ("w0", 0.2, "Δw0 = +0.2")]


def table(header, rows):
    out = [".. list-table::", "   :header-rows: 1", ""]
    for r in [header] + rows:
        out += [f"   * - {r[0]}"] + [f"     - {c}" for c in r[1:]]
    return out + [""]


fits = [(json.loads((RESULTS / f"fit_{t}.json").read_text()), lab) for t, lab in FITS
        if (RESULTS / f"fit_{t}.json").exists()]
rows = []
for fit, slab in fits:
    for f in fit["fits"]:
        b, e = f["best"], f["fisher_err"]
        rows.append(["with" if f["model"] == "resp" else "without", slab,
                     f"{b['Omega_m']:.3f} ± {e['Omega_m']:.3f}", f"{f['sigma8']:.3f} ± {f['sigma8_err']:.3f}",
                     f"{f['chi2']['counts']:.1f}", f"{f['chi2']['wl']:.1f}"])
lines = table(["finder's response", "σ_int", "Ω_m", "σ_8", "χ² counts (20 bins)", "χ² lensing (15 bins)"],
              rows) if rows else ["(the fits are not available)", ""]
(HERE / "fit_table.rst").write_text("\n".join(lines) + "\n")

rows = []
for tag, mor, mstar in VARIANTS:
    p = RESULTS / f"forecast_{tag}.json"
    if not p.exists():
        continue
    fc = json.loads(p.read_text())
    eo, es = fc["noresp"]["err"]["Omega_m"], fc["noresp"]["sigma8_err"]
    sh = {(s["param"], round(s["delta"], 3)): s for s in fc["shifts"]}
    row = [mor, mstar, f"{eo:.3f}", f"{es:.3f}"]
    for par, d, _ in OFFSETS:
        s = sh.get((par, d))
        row.append("—" if s is None else f"{s['shift']['Omega_m'] / eo:+.2f}, {s['sigma8_shift'] / es:+.2f}")
    rows.append(row)
lines = table(["mass–richness", "m*", "σ(Ω_m)", "σ(σ_8)"]
              + [f"{lab}: shifts of Ω_m, σ_8 [σ]" for *_, lab in OFFSETS], rows) \
    if rows else ["(the forecasts are not available)", ""]
(HERE / "forecast_table.rst").write_text("\n".join(lines) + "\n")
old = HERE / "results.rst"
if old.exists():
    old.unlink()
print("wrote fit_table.rst, forecast_table.rst")
