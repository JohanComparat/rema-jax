Best fits to the DR11 counts (Ω_m, ln 10¹⁰A_s, h, n_s, Ω_b, the mass–richness parameters and
the z_λ bias free; σ_int free or fixed at 0.25; errors from the Fisher matrix at the best fit):

.. list-table::
   :header-rows: 1

   * - model
     - scatter
     - Ω_m
     - σ_8
     - σ_int
     - χ² counts (20 bins)
     - χ² weak lensing
   * - without the finder's response
     - σ_int free
     - 0.275 ± 0.034
     - 0.867 ± 0.041
     - 0.020
     - 82.8
     - 15.8 (15 bins)
   * - with the response (strip table)
     - σ_int free
     - 0.274 ± 0.034
     - 0.869 ± 0.041
     - 0.020
     - 84.9
     - 15.6 (15 bins)
   * - without the finder's response
     - σ_int = 0.25
     - 0.276 ± 0.035
     - 0.899 ± 0.037
     - 0.250
     - 92.4
     - 21.4 (15 bins)
   * - with the response (strip table)
     - σ_int = 0.25
     - 0.279 ± 0.035
     - 0.898 ± 0.037
     - 0.250
     - 92.1
     - 21.8 (15 bins)

.. figure:: /figures/cosmology_sensitivity/counts_fit.png
   :alt: DR11 counts against redshift in five richness bins, with the best-fitting models

   The DR11 counts in the volume-limited sky (points, Poisson errors) and the best fits
   without (solid) and with (dashed) the finder's response; right: residuals in units of the
   Poisson error (dots: without, squares: with the response).

.. figure:: /figures/cosmology_sensitivity/constraints.png
   :alt: Omega_m - sigma_8 ellipses with and without the finder's response

   Ω_m–σ_8 (68 and 95 %, Fisher matrix at each best fit) without and with the finder's
   response.

Forecast at the best fit without response, with the DR11 binning, area and covariance:
the shift of the best fit, in units of its error, when the counts of a catalogue made in
a cosmology offset from the true one (column heads: truth − finder) are fitted by the
model without the response.

.. list-table::
   :header-rows: 1

   * - response
     - σ(Ω_m)
     - σ(σ_8)
     - σ(Ω_m) with it
     - Omega_m truth − finder = -0.05: ΔΩ_m, Δσ_8 [σ]
     - Omega_m truth − finder = -0.02: ΔΩ_m, Δσ_8 [σ]
     - Omega_m truth − finder = +0.02: ΔΩ_m, Δσ_8 [σ]
     - Omega_m truth − finder = +0.05: ΔΩ_m, Δσ_8 [σ]
     - w0 truth − finder = -0.2: ΔΩ_m, Δσ_8 [σ]
     - w0 truth − finder = +0.2: ΔΩ_m, Δσ_8 [σ]
   * - catalogue's (m* fixed), richness normalisation free
     - 0.0348
     - 0.0449
     - 0.0344
     - -0.01, +0.01
     - -0.01, +0.00
     - +0.00, -0.00
     - +0.01, -0.01
     - -0.06, +0.05
     - +0.03, -0.02
   * - catalogue's (m* fixed), normalisation fixed
     - 0.0327
     - 0.0357
     - 0.0323
     - -0.01, +0.02
     - -0.00, +0.01
     - +0.00, -0.01
     - +0.01, -0.01
     - -0.06, +0.06
     - +0.03, -0.03
   * - m* following D_L, normalisation free
     - 0.0348
     - 0.0449
     - 0.0329
     - +0.19, -0.13
     - +0.08, -0.05
     - -0.08, +0.06
     - -0.21, +0.15
     - +0.25, -0.17
     - -0.41, +0.29
   * - m* following D_L, normalisation fixed
     - 0.0327
     - 0.0357
     - 0.0315
     - +0.21, -0.13
     - +0.08, -0.06
     - -0.09, +0.06
     - -0.22, +0.16
     - +0.27, -0.18
     - -0.43, +0.32

.. figure:: /figures/cosmology_sensitivity/counts_dlnN.png
   :alt: d ln N / d Omega_m in each bin, total and through the finder

   d ln N/dΩ_m in each bin (lines: richness bins, as in the counts figure): total (left) and
   the part due to the finder's response (right).

.. figure:: /figures/cosmology_sensitivity/shifts.png
   :alt: Shifts of Omega_m and sigma_8 when the response is ignored

   Shifts of the best fit when the response (m* fixed) is ignored, in units of the error.

