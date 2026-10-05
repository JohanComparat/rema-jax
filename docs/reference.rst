Reference
=========

Configuration
-------------

Every setting lives in one YAML file. The defaults follow the redMaPPer v0.7.7 configuration of
the DR10 run (``run_zred_iter1.yml``). Write them out, edit the file, and pass it with ``--config``:

.. code-block:: bash

   python -c "from rema.config import RemaConfig; RemaConfig().to_yaml('run.yaml')"
   rema ingest --config run.yaml ...
   rema calibrate --config run.yaml ...

Each command chooses its configuration in this order:

1. ``--config``, if it is given;
2. the configuration stored with the input:
   - the calibration's, for ``scan``, ``blind``, ``zred`` and ``background``;
   - the catalogue's ``CONFIG`` HDU, for ``specpost``;
3. the defaults.

``rema calibrate`` stores its configuration in the calibration, so the runs inherit it
without a flag. When ``--config`` differs from the stored configuration, the log lists the
keys that differ. ``ingest``, ``maps`` and ``calibrate`` have no stored configuration to fall
back on, so give them ``--config`` whenever you change the defaults.

In Python, use ``RemaConfig.from_yaml(path)``, ``cfg.to_yaml(path)``,
``cfg.replace(model={"chisq_mode": "mag"})`` and ``Calibration.read(path).config``. Unknown
keys raise an error, so a typo cannot pass silently.

The keys that are most often changed:

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - Key
     - Default
     - Meaning (redMaPPer name)
   * - ``survey.bands``, ``survey.ref_band``
     - griz, z
     - Bands of the galaxy table and the reference band
   * - ``survey.ref_snr_min``
     - 5
     - Selection: S/N in the reference band
   * - ``survey.mag_max``
     - null
     - Faint limit of the table. null means m*(1.0) + 2.5, the faint end of the background.
   * - ``survey.reject_psf``
     - true
     - Drop ``TYPE == PSF``
   * - ``survey.flux_floor``
     - 0.015
     - Fractional flux error added in quadrature
   * - ``survey.ebv_max``
     - null
     - E(B−V) cut on the galaxies and the randoms alike. The DR11 production uses 0.2
       (``scripts/slurm/dr11_south.yaml``).
   * - ``model.zrange``
     - [0.05, 0.90]
     - Redshift range of blind mode and of the calibration
   * - ``model.chisq_mode``
     - lupt
     - ``lupt``: asinh-magnitude χ². ``mag``: redMaPPer's colour χ².
   * - ``model.mstar``
     - des_z03
     - m*(z) table: ``des_z03``, ``legacy_z_ezgal``, ``lsst_{i,r,z}03``,
       ``lsst_{u,g,r,i,z,y}_ezgal``, ``euclid_{vis,y,j,h}_ezgal`` (``rema/data/README.md``), or a
       FITS file with Z and MSTAR
   * - ``model.lval_reference``
     - 0.2
     - Luminosity cut of λ in L*, so m < m* + 1.75 (``lval_reference``)
   * - ``model.chisq_max``
     - 20
     - Largest χ² of a member (``chisq_max``)
   * - ``richness.firstpass``, ``.likelihoods``, ``.percolation``
     - (0.5, 0), (1, 0.2), (1, 0.2)
     - r0 and β of each stage, with r_λ = r0 (λ/100)^β (``firstpass_r0``, ...)
   * - ``richness.zscan``
     - (0.5, 0)
     - Aperture of the scan-mode λ(z)
   * - ``richness.minlambda``
     - 3
     - Smallest λ, and smallest λ/S in the catalogue
   * - ``percolation.rmask_0``, ``.rmask_beta``
     - 1.5, 0.2
     - Claim radius R_MASK = max(r_λ, rmask_0 (λ/100)^rmask_beta ((1+z)/1.3)^rmask_gamma),
       with ``rmask_gamma`` 0 (``percolation_rmask_*``)
   * - ``percolation.member_pmin``
     - 0.01
     - Members are written when pmem ≥ this
   * - ``centering.method``
     - auto
     - ``auto`` (wcen when calibrated), ``bcg`` or ``wcen``
   * - ``centering.maxcen``
     - 5
     - Number of centre candidates stored (``percolation_maxcen``)
   * - ``centering.scan_maxrad``
     - 0.4
     - Search radius of the optical centre in scan mode, h⁻¹Mpc
   * - ``mask.nside_fine``, ``mask.max_maskfrac``
     - 1024, 0.2
     - Footprint resolution, and the MASKFRAC cut of the catalogue
   * - ``scan.zrange``, ``scan.zstep``
     - [0.05, 1.0], 0.005
     - Scan grid
   * - ``spec.*``
     - see the file
     - Velocity clipping: ``min_members`` 3, ``vmax_init`` 5000 km/s, ``nsigma_clip`` 3,
       ``niter`` 20, ``gapper_nmax`` 15, ``nboot`` 64, ``seed``
   * - ``calib.niter``
     - 3
     - EM iterations of the calibration
   * - ``calib.wcen``, ``calib.wcen_niter``
     - true, 1
     - Whether to fit wcen. With ``wcen_niter`` 2, wcen is refitted on wcen-centred clusters.

Products
--------

All products are FITS files with upper-case column names. Positions are in degrees, distances
in h⁻¹Mpc (Ω_m = 0.3, flat, from ggah_mod), and fluxes in dereddened nanomaggies.

``rema ingest`` → ``GALAXIES``
    ``ID`` (LS_ID_DR11), ``RA``, ``DEC``, ``FLUX[4]`` and ``FLUX_IVAR[4]`` (griz), ``REFMAG``,
    ``REFMAG_ERR``, ``EBV``, ``TYPE``, ``MASKBITS``, ``ZSPEC`` (−1: none), ``ZSPEC_SRC``,
    ``ZPHOT``, ``ZPHOT_STD``. The header records the bands, ``MAGMAX``, ``SNRMIN`` and the box.

``rema maps`` → ``FINE``
    ``PIXEL`` (NESTED, nside 1024), ``NRAND``, ``COVER``, ``FRACGOOD``, ``SIGF_G`` … ``SIGF_Z``.

``rema calibrate`` → calibration
    - Red sequence: ``RS_MEAN``, ``RS_SLOPE``, ``RS_LOG_SIGMA``, ``RS_CORR`` and ``RS_PIVOT``
      (spline nodes in z).
    - zred correction: ``ZREDCORR`` and ``ZREDCORR_SLOPE``.
    - χ² background: ``BKG_CHISQ``, with axes ``BKG_Z``, ``BKG_C`` and ``BKG_M``.
    - zred background: ``BKG_ZRED``, with axes ``BKG_ZRED_Z`` and ``BKG_ZRED_M``.
    - ``ZLAMBDACORR``, ``WCEN`` (one row of parameters) and ``CONFIG``.
    - The primary header holds ``REMAVER``, ``BANDS``, ``REFBAND``, ``CHI2MODE``, ``MSTAR``,
      ``NCLUSTER``, ``NSPECGAL``, ``ZLNMAD`` and ``NWCEN``.

``rema blind`` and ``rema scan`` → ``CLUSTERS``, ``MEMBERS``, ``CONFIG``
    - The primary header holds ``REMAVER``, ``MODE``, ``CENTRING``, ``NCLUSTER`` and
      ``NMEMBER``.
    - Blind runs add their provenance: ``REGION``, ``PLANHASH``, ``CALSHA1`` (the calibration
      file), ``CFGSHA1`` and ``DEVICE``.
    - They also add the data box (``BOXRA0``…, or ``NBOX`` with ``B01RA0``…), the own box
      (``OWNRA0``…) and the run statistics: ``NGAL``, ``NSEED``, ``NFIRST``, ``NCAND``,
      ``NDEP``, ``TFIRST``, ``TLIKE``, ``TTOTAL`` and ``PEAKRSS``.
    - An empty catalogue keeps empty HDUs, with ``NCLUSTER = 0``.

Large runs (:doc:`hpc`):

``rema ingest --outdir`` → one ``GALAXIES`` table per sweep, named like the sweep
    The header records ``SWEEP``, ``NGAL``, ``NZSPEC``, ``EBVMEAN``, ``SURVHASH``, ``MAGMAX``
    and the sweep's box.
``rema randoms-index`` → ``<stem>.npy``, ``<stem>.offsets.npy``, ``<stem>.json``
    The first file holds the randoms columns sorted by NESTED pixel at nside 1024. The second
    gives the first row of each nside-64 pixel. The third, written last, holds the source,
    the settings and the number of rows.
``rema regions`` → ``REGIONS``
    - Columns: ``REGION_ID``, ``OWN_RA0``…, ``DATA_RA0``…, ``AREA_OWN``, ``AREA_COVERED``,
      ``NGAL_OWN``, ``NGAL_DATA`` and ``PAIRS_EST``.
    - Header: the settings and ``PLANHASH``.
``rema merge`` → ``<out>.fits``, ``<out>_members.fits``, ``<out>_regions.fits``, ``<out>_qa.json``
    The merged clusters, the members, the per-region statistics and the QA.

``CLUSTERS``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Columns
     - Meaning
   * - ``MEM_MATCH_ID``
     - Blind: (region << 32) | rank in decreasing λ. Scan: the ``--id-col`` value.
   * - ``RA``, ``DEC``
     - Centre: the most probable central galaxy, ``ID_CENT[0]``. Scan: the input position.
   * - ``SEED_ID``, ``RA_SEED``, ``DEC_SEED``, ``Z_INIT``
     - Blind: the seed galaxy and its zred
   * - ``LAMBDA``, ``LAMBDA_E``, ``R_LAMBDA``, ``SCALEVAL``, ``MASKFRAC``
     - Richness, its error, its radius, the completeness correction 1/K(r_λ), and the masked
       fraction of the aperture
   * - ``Z_LAMBDA``, ``Z_LAMBDA_E``
     - z_λ and its error, after the calibrated correction. ``…_RAW``: before it.
   * - ``PZBINS[21]``, ``PZ[21]``, ``Z_LAMBDA_NITER``
     - p(z), and the number of z_λ iterations
   * - ``LNLAMLIKE``, ``LNCGLIKE``, ``LNLIKE``
     - Richness likelihood, central-galaxy likelihood (wcen; 0 with BCG), and their sum,
       which ranks the percolation
   * - ``R_MASK``
     - Radius within which the cluster claims galaxies in percolation
   * - ``ID_CENT[5]``, ``RA_CENT[5]``, ``DEC_CENT[5]``
     - Centre candidates, best first. Unused slots hold −1 and −400.
   * - ``P_CEN[5]``, ``Q_CEN[5]``, ``P_SAT[5]``, ``P_FG[5]``, ``P_C[5]``
     - Centring probabilities, as in redMaPPer. BCG gives ``P_CEN[0] = 1``.
   * - ``NCENT_GOOD``, ``Q_MISS``, ``W``
     - Number of candidates with P_C > 0, the probability that the centre is missed, and the
       connectivity of the centre
   * - ``REFMAG``, ``REFMAG_ERR``, ``ZRED``, ``ZRED_E``, ``ZRED_CHISQ``, ``CHISQ``
     - The central galaxy (blind)
   * - ``Z_STEPS[191]``, ``LAMBDA_STEPS``, ``LIKELIHOOD_STEPS``, ``ZMAX``, ``LMAX``, ``MAX_IND``,
       ``ZMAX_EDGE``
     - Scan: λ(z) and the likelihood on the grid, and its peak. ``ZMAX_EDGE`` flags a peak at
       the grid edge.
   * - ``RA_OPT``, ``DEC_OPT``, ``LAMBDA_OPT(_E)``, ``Z_LAMBDA_OPT(_E)``, ``Z_LAMBDA_OPT_RAW``,
       ``IN_CALIB_ZRANGE``
     - Scan: optical centre, λ and z_λ there, and whether z_λ is inside ``model.zrange``
   * - ``NSPEC``, ``N_MEMBERS``
     - Members with a spectroscopic redshift, and those kept by the velocity clipping
   * - ``SPEC_Z``, ``SPEC_ZERR_RUEL``, ``SPEC_Z_BOOT``, ``SPEC_ZERR_BOOT``
     - Biweight cluster redshift, its error σ_v (1+z)/(c √N), and its bootstrap mean and error
   * - ``VDISP``, ``VDISP_ERR``, ``VDISP_BOOT``, ``VDISP_ERR_BOOT``, ``VDISP_CLIP``,
       ``VDISP_TYPE``, ``VDISP_FLAG``, ``VDISP_FLAG_BOOT``, ``BOOTNUM``
     - Velocity dispersion (km/s): the gapper or biweight estimate, the bootstrap results,
       the clipping radius and the flags
   * - ``CG_SPEC_Z``, ``BEST_Z``, ``BEST_ZERR``, ``BEST_Z_TYPE``
     - Spec-z of the central galaxy, and the best redshift: ``spec_z_boot``, else
       ``cg_spec_z``, else ``photo_z``

``MEMBERS``: one row per galaxy with pmem ≥ ``percolation.member_pmin``, plus the centre
candidates:

- ``MEM_MATCH_ID``, ``ID``, ``RA``, ``DEC``; ``Z`` is the cluster's z_λ before the
  correction;
- ``R``: the projected distance to the centre;
- ``P``, ``PFREE``, ``THETA_I``, ``THETA_R`` and ``PMEM`` = P · PFREE · THETA_I · THETA_R.
  ``PCOL`` is redMaPPer's colour-only probability within r_λ;
- ``CHISQ`` at the cluster redshift; ``REFMAG``, ``REFMAG_ERR``, ``ZRED``, ``ZRED_E``,
  ``ZSPEC``, ``FLUX``, ``FLUX_IVAR``;
- blind mode: ``CG`` (the central galaxy) and ``CENT_RANK`` (0–4 for the centre candidates,
  −1 otherwise);
- spectroscopic post-processing: ``VEL`` (rest-frame velocity relative to ``SPEC_Z``, km/s)
  and ``ISMEMBER_SPEC`` (kept by the clipping).
