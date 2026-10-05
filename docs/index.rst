rema
====

**Red-sequence matched-filter galaxy cluster finding in JAX, for the Legacy Surveys DR11.**

rema reimplements redMaPPer (Rykoff et al. 2014, 2016). It keeps redMaPPer's red-sequence
model, zred, richness λ, z_λ, BCG and wcen centring, percolation and redshift scans at given
positions (zscan), and adds the Clerc et al. (2016) velocity clipping. The kernels are batched JAX that run on CPU or GPU, λ has exact
derivatives, and the package installs with ``pip``.

These pages are for readers who know redMaPPer, and they are hands-on: every command and code
block has been run. Start with :doc:`tour` (mock data, two minutes on a laptop). Then follow
:doc:`dr11` for the real data, the :ref:`notebooks <notebooks>` (a set of sweeps as one region,
the same sweeps through the HPC pipeline, one cluster in scan mode) and :doc:`hpc` for the full
DR11 run. :doc:`design` gives the algorithm, the deviations from redMaPPer and the
validation.

What differs from redMaPPer
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * -
     - redMaPPer v0.7.7
     - rema
   * - Colour likelihood
     - χ² of the colours in magnitudes
     - χ² in asinh magnitudes, softened at each band's 1σ flux error, so faint and negative
       g fluxes stay usable (``model.chisq_mode: lupt``). ``mag`` reproduces redMaPPer.
   * - Selection and depth
     - The redMaPPer DR10 input catalogue drops galaxies with a flux ≤ 0 or σ_m > 1 in any band. The local depth
       enters θ_i and the maskgals.
     - An explicit cut at S/N ≥ 5 in the reference band. Mask and depth enter only the
       completeness K(r_λ) = 1/SCALEVAL.
   * - Mask correction
     - Monte-Carlo maskgals
     - Deterministic quadrature over a footprint map made from the DR11 randoms
   * - λ solve
     - Iterated per cluster
     - Batched bracket and Newton steps inside ``jax.lax.custom_root``, so λ has exact
       implicit derivatives
   * - Percolation
     - Sequential loop
     - The same result as the sequential loop, computed in batches over a dependency graph
   * - Calibration
     - Per-node fits, iterated
     - EM: one L-BFGS fit of all red-sequence nodes with a smoothness prior. wcen is fitted
       as in redMaPPer's WcenCalibrator.
   * - Infrastructure
     - C extensions, esutil, healsparse
     - numpy, scipy, astropy, healpy and JAX, with one process per sky region (SLURM arrays)

.. toctree::
   :maxdepth: 2
   :caption: Hands-on

   install
   tour
   dr11
   hpc

.. _notebooks:

.. toctree::
   :maxdepth: 1
   :caption: Notebooks (DR11)

   notebooks/blind_dr11
   notebooks/pipeline_dr11
   notebooks/scan_dr11

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference
   api
   design
   changelog
