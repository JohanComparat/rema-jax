rema
====

**Red-sequence matched-filter galaxy cluster finding in JAX, for the Legacy Surveys DR11.**

``rema`` finds galaxy clusters in a multi-band photometric galaxy catalogue and returns their
richness λ, redshift z_λ, centre candidates and members.
It reimplements redMaPPer :cite:p:`Rykoff2014,Rykoff2016`: the red-sequence model, zred, λ, z_λ,
brightest cluster galaxy (BCG) and wcen centring, percolation, and redshift scans at given
positions (zscan).
It adds the velocity clipping of :cite:t:`Clerc2016`.
The kernels are batched JAX functions that run on a CPU or a GPU, λ has exact derivatives, and the
package installs with ``pip``.

We ran ``rema`` on the south part of the Legacy Surveys data release 11 (DR11 south).
The catalogue holds 3 481 608 clusters with λ ≥ 3 (145 133 with λ ≥ 20) over 18 500 deg² and is
on the data system of the IN2P3 computing centre (CC-IN2P3, :doc:`ccin2p3`).
We validated it against the figures of the redMaPPer papers (:doc:`redmapper_dr11`).
At the 1.64 M clusters of nine public Sunyaev–Zel'dovich, X-ray and optical catalogues, we
measured λ, z_λ and the spectroscopic redshift (:doc:`public_catalogues_scan`).

The pages are for readers who know redMaPPer, and we ran every command and code block in them:

- :doc:`install`, including :ref:`the notebooks at CC-IN2P3 <cc-notebooks>` on the Jupyter
  platform https://notebook.cc.in2p3.fr;
- :doc:`tour`: two minutes on a laptop, without data;
- :doc:`dr11`: the chain on three DR11 sweeps, step by step;
- the :ref:`notebooks <notebooks>`: a set of sweeps as one region, the same sweeps through the
  high-performance computing (HPC) pipeline, one cluster in scan mode, and the figures of the
  production catalogue;
- :doc:`hpc` and :doc:`ccin2p3`: production runs on an HPC system, and the DR11 catalogue at
  CC-IN2P3;
- :doc:`design`: the algorithm, the deviations from redMaPPer and the validation.

Differences from redMaPPer
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * -
     - redMaPPer v0.7.7
     - rema
   * - Colour likelihood
     - χ² of the colours in magnitudes
     - χ² in asinh magnitudes :cite:p:`Lupton1999`, softened at each band's 1σ flux error, so that
       faint and negative g fluxes stay usable (``model.chisq_mode: lupt``).
       ``mag`` reproduces redMaPPer.
   * - Selection and depth
     - The redMaPPer DR10 input catalogue drops galaxies with a flux ≤ 0 or σ_m > 1 in any band.
       The local depth enters θ_i and the maskgals.
     - An explicit cut at signal-to-noise ratio S/N ≥ 5 in the reference band.
       Mask and depth enter only the completeness K(r_λ) = 1/SCALEVAL.
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
     - Expectation–maximisation: one L-BFGS fit of all red-sequence nodes with a smoothness
       prior.
       wcen is fitted as in redMaPPer's WcenCalibrator.
   * - Infrastructure
     - C extensions, esutil, healsparse
     - numpy, scipy, astropy, healpy and JAX, with one process per sky region (SLURM arrays)

.. toctree::
   :maxdepth: 2
   :caption: The DR11 south catalogue

   ccin2p3
   redmapper_dr11
   cosmology_sensitivity
   photoz_finders
   photoz_study_findings
   public_catalogues_scan

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
   notebooks/redmapper_dr11

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference
   tables
   api
   design
   literature
   changelog
