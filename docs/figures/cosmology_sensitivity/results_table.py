"""Write results.rst (included by docs/cosmology_sensitivity.rst) from fit_<tag>.json and the
forecast_<variant>.json files in $REMA_COSMO."""

import json
import os

from common import HERE, RESULTS

FITS = [("all", "σ_int free"), ("all_sig025", "σ_int = 0.25")]
FC = os.environ.get("COSMO_FORECAST", "sig025_strip")
VARIANTS = [(FC, "catalogue's (m* fixed), richness normalisation free"),
            (f"{FC}_fixednorm", "catalogue's (m* fixed), normalisation fixed"),
            (f"{FC}_mstar", "m* following D_L, normalisation free"),
            (f"{FC}_mstar_fixednorm", "m* following D_L, normalisation fixed")]
lines = []
fits = [(json.loads((RESULTS / f"fit_{t}.json").read_text()), lab) for t, lab in FITS
        if (RESULTS / f"fit_{t}.json").exists()]
if fits:
    lines += [
        "Best fits to the DR11 counts (Ω_m, ln 10¹⁰A_s, h, n_s, Ω_b, the mass–richness parameters and",
        "the z_λ bias free; σ_int free or fixed at 0.25; errors from the Fisher matrix at the best fit):",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - model",
        "     - scatter",
        "     - Ω_m",
        "     - σ_8",
        "     - σ_int",
        "     - χ² counts (20 bins)",
        "     - χ² weak lensing",
    ]
    for fit, slab in fits:
      for f in fit["fits"]:
        b, e = f["best"], f["fisher_err"]
        name = "without the finder's response" if f["model"] == "noresp" else "with the response (strip table)"
        lines += [f"   * - {name}", f"     - {slab}",
                  f"     - {b['Omega_m']:.3f} ± {e['Omega_m']:.3f}",
                  f"     - {f['sigma8']:.3f} ± {f['sigma8_err']:.3f}",
                  f"     - {b.get('sigma_int', 0.25):.3f}",
                  f"     - {f['chi2']['counts']:.1f}",
                  f"     - {f['chi2']['wl']:.1f} ({f['chi2']['n_wl']} bins)"]
    lines += ["", ".. figure:: /figures/cosmology_sensitivity/counts_fit.png",
              "   :alt: DR11 counts against redshift in five richness bins, with the best-fitting models", "",
              "   The DR11 counts in the volume-limited sky (points, Poisson errors) and the best fits",
              "   without (solid) and with (dashed) the finder's response; right: residuals in units of the",
              "   Poisson error (dots: without, squares: with the response).", "",
              ".. figure:: /figures/cosmology_sensitivity/constraints.png",
              "   :alt: Omega_m - sigma_8 ellipses with and without the finder's response", "",
              "   Ω_m–σ_8 (68 and 95 %, Fisher matrix at each best fit) without and with the finder's",
              "   response.", ""]
rows = []
for v, label in VARIANTS:
    p = RESULTS / f"forecast_{v}.json"
    if not p.exists():
        continue
    fc = json.loads(p.read_text())
    eo, es = fc["noresp"]["err"]["Omega_m"], fc["noresp"]["sigma8_err"]
    sh = {f"{s['param']} truth − finder = {s['delta']:+g}": s for s in fc["shifts"]}
    rows.append((label, eo, es, fc["resp"]["err"]["Omega_m"], sh))
if rows:
    keys = list(rows[0][4])
    lines += [
        "Forecast at the best fit without response, with the DR11 binning, area and covariance:",
        "the shift of the best fit, in units of its error, when the counts of a catalogue made in",
        "a cosmology offset from the true one (column heads: truth − finder) are fitted by the",
        "model without the response.",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - response",
        "     - σ(Ω_m)",
        "     - σ(σ_8)",
        "     - σ(Ω_m) with it",
    ] + [f"     - {k}: ΔΩ_m, Δσ_8 [σ]" for k in keys]
    for label, eo, es, eo1, sh in rows:
        lines += [f"   * - {label}", f"     - {eo:.4f}", f"     - {es:.4f}", f"     - {eo1:.4f}"]
        lines += [f"     - {sh[k]['shift']['Omega_m'] / eo:+.2f}, {sh[k]['sigma8_shift'] / es:+.2f}" for k in keys]
    lines += ["", ".. figure:: /figures/cosmology_sensitivity/counts_dlnN.png",
              "   :alt: d ln N / d Omega_m in each bin, total and through the finder", "",
              "   d ln N/dΩ_m in each bin (lines: richness bins, as in the counts figure): total (left) and",
              "   the part due to the finder's response (right).", "",
              ".. figure:: /figures/cosmology_sensitivity/shifts.png",
              "   :alt: Shifts of Omega_m and sigma_8 when the response is ignored", "",
              "   Shifts of the best fit when the response (m* fixed) is ignored, in units of the error.", ""]
(HERE / "results.rst").write_text("\n".join(lines) + "\n" if lines else
                                  "The fit and forecast results are not available yet.\n")
print("wrote results.rst")
