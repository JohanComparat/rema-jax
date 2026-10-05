Two minutes on mock data
========================

:download:`examples/mock_tour.py` runs everything on this page. It injects two clusters into a
mock field, then runs blind mode, spectroscopic post-processing and scan mode, and takes
derivatives of λ. It needs no data. On a laptop CPU it takes about two minutes with a cold
compilation cache and 1.5 minutes with a warm one:

.. code-block:: bash

   python docs/examples/mock_tour.py                       # CPU
   JAX_PLATFORMS=cuda python docs/examples/mock_tour.py    # GPU

The script starts with the JAX settings of :doc:`install`.

A mock region
-------------

.. literalinclude:: examples/mock_tour.py
   :language: python
   :start-after: # [mock]
   :end-before: # [/mock]

.. code-block:: text

   26988 galaxies; centring: bcg

- A galaxy table is a dict of columns: ``ID``, ``RA``, ``DEC``, ``FLUX`` and ``FLUX_IVAR``
  (dereddened nanomaggies, in the bands of ``cfg.survey.bands``), ``REFMAG``, ``REFMAG_ERR``
  and ``ZSPEC`` (−1 where there is none). ``rema ingest`` writes the same columns for DR11.
- ``RSModel.from_template()`` is the BC03 griz red sequence before any calibration. With real
  data, use ``Calibration.read(path).rs``.
- A :class:`~rema.modes.common.Region` holds the galaxies, a neighbour index, the optional
  footprint and the filter model. The filter model is a single pytree holding the red
  sequence, the χ² background, the cosmology table and m*(z). ``Region.build`` computes zred
  and the χ² background when the table and the calibration lack them. Without a footprint,
  ``area_deg2`` sets the background normalisation.

Blind mode and spectroscopic post-processing
--------------------------------------------

.. literalinclude:: examples/mock_tour.py
   :language: python
   :start-after: # [blind]
   :end-before: # [/blind]

.. code-block:: text

   53 clusters (lambda/S >= 3, MASKFRAC < 0.2), 543 members
       RA     Dec   Z_LAMBDA  LAMBDA  SPEC_Z_BOOT  VDISP +- ERR  N_MEMBERS
    10.500 -0.300   0.3014    39.6       0.2989    788 +- 322         15
    11.000  0.300   0.5598    31.7       0.5519    800 +- 143         15
    11.416 -0.283   0.8806     6.1          nan    nan +- nan          0

The injected clusters had (z, λ) = (0.30, 40) and (0.55, 30), and σ_v = 700 km/s.

- :func:`~rema.modes.blind.run_blind` runs four stages:

  - seeds;
  - the first pass (r0 = 0.5, β = 0; λ ≥ 3 is kept);
  - the likelihood pass (r0 = 1, β = 0.2);
  - the exact percolation, then consolidation: λ/S ≥ 3, MASKFRAC < 0.2, and the centre in
    ``own``.

  The returned catalogue and members are dicts of numpy arrays with redMaPPer's column names
  (:doc:`reference`).
- :func:`~rema.modes.specpost.process` is the Clerc et al. (2016) velocity clipping,
  run on the members with ``ZSPEC > 0``. It uses a biweight location, a 5000 km/s window and
  3σ clipping. σ_v is the gapper estimate below 15 members and the biweight scale above. 64
  bootstrap resamples give the errors. The keyword arguments are those of
  ``cfg.spec``.
- There is no wcen model here, so the centre is redMaPPer's CenteringBCG: the brightest
  galaxy within r_λ with pmem > 0.8 or \|zred − z\| < 2 zred_e. That rule has no χ² cut, so a
  bright interloper with a matching zred can take the centre. With another random seed, the
  first mock cluster gets centred on a 14.7 mag field galaxy with χ² = 70, 0.03° from the
  injected central galaxy. wcen centring, which ``rema calibrate`` fits, weighs each
  candidate's magnitude against m* + Δ0, its zred and its connectivity to the members. A
  galaxy 3.7 mag brighter than m* is then a 5σ outlier in magnitude.

Scan mode
---------

.. literalinclude:: examples/mock_tour.py
   :language: python
   :start-after: # [scan]
   :end-before: # [/scan]

.. code-block:: text

   input 0: ZMAX 0.315  Z_LAMBDA_OPT 0.3014  LAMBDA_OPT 39.6  centre offset 0.0 arcsec
   input 1: ZMAX 0.555  Z_LAMBDA_OPT 0.5598  LAMBDA_OPT 31.7  centre offset 0.0 arcsec

:func:`~rema.modes.scan.run_scan` is redMaPPer's zscan, in four steps:

1. λ(z) and the likelihood are computed on Δz = 0.005 steps from z = 0.05 to 1.0, in a
   0.5 h⁻¹Mpc aperture at the input position.
2. ZMAX is the likelihood peak.
3. z_λ is refined from ZMAX with the percolation aperture.
4. The optical centre is chosen within 0.4 h⁻¹Mpc. λ and z_λ are recomputed there and stored
   as ``LAMBDA_OPT`` and ``Z_LAMBDA_OPT``.

Every input position is kept, including those without a cluster.

Derivatives of λ
----------------

.. literalinclude:: examples/mock_tour.py
   :language: python
   :start-after: # [grad]
   :end-before: # [/grad]

.. code-block:: text

   lambda 39.65  dlambda/dz -18.3  (finite difference -18.1)
   dlambda/d(mean colour) at the z = 0.30 node:  g-r -1.3  r-i -12.5  i-z -15.1

- :func:`~rema.core.richness.richness` solves Σ pmem = λ K(r_λ) inside ``jax.lax.custom_root``.
  ``jax.grad`` therefore gives the implicit derivative of λ with respect to anything the
  filter depends on: the redshift, the red-sequence nodes, the background or the cosmology. It
  does not differentiate through the iterations. ``richness`` is batched and jitted, so arrays
  are [B, K] and one compilation serves every cluster of the same shape.
- z = 0.30 is a node of the red-sequence splines, so there λ depends only on that node's
  colours. Between nodes, the gradient spreads over the neighbouring nodes.

.. note::

   **Finite differences need small steps.** Each bright galaxy enters θ_i, the soft
   luminosity cut, over a width set by its photometric error (≈ 0.01 mag, or Δz ≈ 0.002).
   λ(z) is smooth but varies on scales of Δz ~ 10⁻³. At z = 0.32 in this mock, a centred
   difference with h = 10⁻³ gives +126, while the derivative is −11.6. In float32 the finite
   difference converges to the autodiff value for h ≲ 3·10⁻⁵, and for the colour nodes at
   h ~ 10⁻⁴. A gradient-based fit through λ(z) has to deal with this structure.

Next, :doc:`dr11` runs the same steps on the Legacy Surveys.
