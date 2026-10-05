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

Inputs and calibration
----------------------

.. automodule:: rema.config
   :members: RemaConfig

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
   :members: mock_field, mock_cluster, concat

Model tables
------------

.. automodule:: rema.model.sps
   :members: SSPGrid, Bandpass, exponential_weights, csp_fnu, ab_mag_10pc, lookback_time,
             passive_mags

.. automodule:: rema.data.build
   :members: build_all, build_mstar_redmapper, build_mstar_des_z03, build_bc03_tables,
             build_ezgal_tables, passive_population_mags, FilterSet, FILTER_SETS, fetch, compare
