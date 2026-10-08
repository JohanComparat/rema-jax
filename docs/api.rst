Python API
==========

The modules most used in scripts. Every function works on plain dicts of numpy arrays (tables)
or on JAX pytrees (models), and the command-line interface (``rema.cli``) is a thin layer over
them.

Run modes
---------

.. automodule:: rema.modes.common
   :members: Region

.. automodule:: rema.modes.blind
   :members: run_blind, select_seeds, percolate, consolidate

.. automodule:: rema.modes.scan
   :members: run_scan

.. automodule:: rema.modes.specpost
   :members: process, clip_velocity_batch, bootstrap_clip

.. automodule:: rema.modes.remeasure
   :members: Centres, centres_from_catalog, MemberPfree, remeasure, remeasure_cosmologies,
             cosmology_grid, response_jvp, response_fd, galaxies_near, write_remeasure,
             read_remeasure

Inputs and calibration
----------------------

.. automodule:: rema.config
   :members: RemaConfig, CosmologyConfig, PhotozConfig, PscdConfig, NullConfig,
             parse_cosmology_overrides, apply_overrides

.. automodule:: rema.calibration
   :members: Calibration, ZlambdaCorrection

.. automodule:: rema.io.legacy
   :members: ingest, ingest_sweep, ingest_sweeps, read_galaxies, galaxy_counts, survey_hash

.. automodule:: rema.io.tables
   :members: read_table, write_table, table_hdu, write_fits, write_catalog, read_catalog

.. automodule:: rema.sky.maps
   :members: Footprint, build_footprint, read_randoms, index_randoms, read_randoms_index,
             randoms_index_files, sigma_flux_from_depth

.. automodule:: rema.sky.regions
   :members: Box, BoxUnion, sweep_box, sweep_union, sweeps_overlapping, bounding_box, intersect,
             sky_header, sky_from_header

Large runs
----------

.. automodule:: rema.pipeline
   :members: plan_regions, write_plan, read_plan, plan_boxes, required_buffer, uncovered_tiles,
             calib_suggest, region_status, merge_regions, close_pairs, edge_profile

.. automodule:: rema.calib.driver
   :members: calibrate_region, calibrate_wcen

Kernels
-------

.. automodule:: rema.core.richness
   :members: Neighbors, Stage, RadialQuad, Richness, richness

.. automodule:: rema.core.zlambda
   :members: ZLambda, zlambda

.. automodule:: rema.core.centering
   :members: Centering, WcenModel, center_bcg, center_wcen, lncglike

.. automodule:: rema.core.zred
   :members: compute_zred

.. automodule:: rema.model.redsequence
   :members: RSModel, ZredCorrection

.. automodule:: rema.validate.mocks
   :members: mock_field, mock_cluster, mock_template_cluster, GaussianPhotoz, concat

Photo-z cluster finding
-----------------------

See :doc:`photoz_finders`.

.. automodule:: rema.model.photoz
   :members: photoz_sigma, err_scale, photoz_valid

.. automodule:: rema.model.background
   :members: build_photoz_bkg, ZredBkg

.. automodule:: rema.validate.photoz
   :members: fit_err_scale, yaml_overlay

.. automodule:: rema.validate.null
   :members: null_shuffle

.. automodule:: rema.validate.compare
   :members: match_physical, null_threshold

.. automodule:: rema.pscd

.. automodule:: rema.pscd.model
   :members: Template, WidthTable, MagIntegrals

.. automodule:: rema.pscd.grid
   :members: Grid, gnomonic, inverse_gnomonic, paint, convolve

.. automodule:: rema.pscd.detect
   :members: build_cube, extract, amplitude, memberships, clean

.. automodule:: rema.pscd.run
   :members: run_pscd

Cluster abundance and cosmology
-------------------------------

See :doc:`cosmology_sensitivity`.

.. automodule:: rema.abundance.response
   :members: ResponseTable, delta_lnlam, delta_z, from_remeasure

.. automodule:: rema.abundance.area
   :members: ZvlimMap, zvlim_of

.. automodule:: rema.abundance.data
   :members: DataVector, build_data_vector

.. automodule:: rema.abundance.mor
   :members: MassRichness, mean_lnlam, var_lnlam, p_bins, mcclintock_lnm, mcclintock_cov

.. automodule:: rema.abundance.counts
   :members: CountsSetup, CountsModel, cosmology

.. automodule:: rema.abundance.covariance
   :members: sigma2_b_slabs, counts_covariance

.. automodule:: rema.abundance.likelihood
   :members: Likelihood, WLData

.. automodule:: rema.abundance.fisher
   :members: jacobians, fisher, constraints, shift

.. automodule:: rema.abundance.sampling
   :members: map_fit, laplace, nuts

Model tables
------------

.. automodule:: rema.model.cosmo
   :members: CosmoTable

.. automodule:: rema.model.sps
   :members: SSPGrid, Bandpass, exponential_weights, csp_fnu, ab_mag_10pc, lookback_time,
             passive_mags

.. automodule:: rema.data.build
   :members: build_all, build_mstar_redmapper, build_mstar_des_z03, build_bc03_tables,
             build_ezgal_tables, passive_population_mags, FilterSet, FILTER_SETS, fetch, compare
