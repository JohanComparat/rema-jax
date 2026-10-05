# Changelog

All notable changes to `rema` are documented here.

## [Unreleased]

### Model tables

- LSST and Euclid tables in `rema/data`, built by `python -m rema.data.build` like the DECam ones:
  redMaPPer's m*(z) in LSST i, r, z (`mstar_lsst_{i,r,z}03`); the BC03 passive population's m*(z)
  in every LSST band (`mstar_lsst_{u,g,r,i,z,y}_ezgal`, lsst/throughputs 1.9) and every Euclid band
  (`mstar_euclid_{vis,y,j,h}_ezgal`), and its adjacent colours (`colors_bc03_lsst_ugrizy`,
  `colors_bc03_euclid_visyjh`), at z = 0.01-2.50. The builder works per filter set
  (`FILTER_SETS`, `build_bc03_tables`, `build_mstar_redmapper`); the DECam tables are unchanged.
- `model.template` chooses the colour table that seeds the calibration (default
  `bc03_legacy_grizw1`; `bc03_lsst_ugrizy`, `bc03_euclid_visyjh` or a FITS file).
  `RSModel.from_template` reads the template's bands from its `BANDS` header and rejects bands
  that are not among them. Calibrations written with this option are not readable by rema 0.1.0.

### HPC

- CC-IN2P3: region tasks request their GPU with `--gpus 1`; the cluster rejects `--gres`.
  New driver variable `GPUS` (sends `--gpus=N` instead of `--gres=GPU_GRES`).
  `ccin2p3.env` sets `GPUS=1` with 5 CPUs and 45G per V100 task, the cluster's per-GPU CPU
  limit, so four tasks share a node. It also keeps JAX on the CPU on login and CPU nodes.
- `get_dr11_south_sweeps.sh`: `SUBDIRS=11.0-photo-z` gets only the photo-z files, in a second job
  next to the one getting the sweeps.
- `AREA_BOX` now also restricts the driver's bookkeeping (`rema todo --box`). `prepare` and `run`
  can then process part of the sky, for a test or while the download is incomplete.

## [0.1.0] - 2026-10-05

First public release, on PyPI as `rema-jax` (`pip install rema-jax`; the package is imported as
`rema` and installs the `rema` command). Versions below 1.0 may still change the interface.
Validated on a 75 deg² DR11 area: blind mode against redMaPPer DR10, scan mode at ACT DR5
clusters, z_lambda against spectroscopic cluster redshifts (`docs/design.md`).

### Model and algorithms

- Red-sequence model (`RSModel`) with unconstrained parameters (log scatter, partial
  correlations), a BC03 griz template and a reader for redMaPPer `*_pars.fit` files.
- Photometric likelihood in asinh magnitudes (default) or in magnitudes (redMaPPer).
- zred with redMaPPer's posterior, parabola refinement and corrections; chi^2 and zred
  backgrounds.
- Richness solve with exact implicit derivatives (`jax.lax.custom_root`), deterministic mask and
  depth completeness from a footprint map, z_lambda iteration with p(z).
- Centring on the brightest central candidate (BCG) or with redMaPPer's wcen model
  (`centering.method`: "auto" uses wcen when the calibration holds a wcen model). Up to 5 centre
  candidates per cluster (ID_CENT, P_CEN, ...), kept in the member table (CENT_RANK); the
  central-galaxy likelihood LNCGLIKE enters the likelihood pass.
- Blind mode: seeds, first pass, likelihood pass, percolation with redMaPPer's rejections
  (computed in batches of independent candidates, with the result of the sequential loop),
  consolidation, and checkpoints (`rema blind --checkpoint`).
- Scan mode at given positions (redMaPPer's zscan).
- Spectroscopic post-processing: Clerc et al. (2016) velocity clipping, bootstrap errors,
  BEST_Z.
- Calibration on DR11 (`rema calibrate`): initial model from spectroscopic galaxies, EM fit of
  the red-sequence nodes, zred and z_lambda corrections (with the z -> zred_uncorr mapping), wcen
  calibration (redMaPPer's WcenCalibrator), diagnostic plots. `rema calib-import` starts from a
  redMaPPer calibration instead.

### Data and production

- Legacy Surveys DR11 ingestion: sweeps plus row-matched photo-z sweeps reduced to per-sweep
  galaxy tables (dereddened griz fluxes, z-band S/N >= 5, `ZSPEC`), with explicit MASKBITS, TYPE,
  NOBS and E(B-V) cuts.
- Footprint and depth maps from the DR11 randoms, through a pixel-sorted randoms index read once
  (`rema randoms-index`, `rema maps`).
- Full-sky production: the region planner (`rema regions`), `rema status`, `rema todo` and
  `rema merge` with QA (duplicates, centres outside their box, close pairs across regions,
  provenance checks). Catalogues record their provenance (plan, calibration and configuration
  hashes, device, boxes) in the header.
- SLURM driver (`scripts/slurm/rema_dr11_blind.sh`: prepare, run, status, clean) with the stage
  script `rema_task.sh` and the production configuration `dr11_south.yaml`; only the region runs
  use the GPU. Settings for the CC-IN2P3 cluster in `scripts/slurm/ccin2p3.env`
  (`LEGACYSURVEY_DIR`, `CLUSTERS_DIR`, `MEMBERS_DIR`, partitions), and download scripts for the
  DR11 south sweeps and randoms.
- Model tables (`rema/data`): redMaPPer's m*(z), and the m*(z) and colours of a passive
  Bruzual & Charlot (2003) population computed in JAX (`rema.model.sps`). `python -m
  rema.data.build` rebuilds them from public inputs at fixed commits (EzGal's SSP grid,
  speclite filters, redMaPPer), checked by SHA-256.
- Configuration as nested frozen dataclasses with YAML input and output; defaults follow
  redMaPPer v0.7.7. The calibration and the catalogues store the configuration they used.

### Documentation, packaging and tests

- Sphinx documentation (Read the Docs): install, a tour on mock data, DR11 end to end, HPC
  production, reference, API, and three executed DR11 notebooks (blind run, pipeline stages,
  scan mode).
- Packaging for PyPI (`rema-jax`, extras `cuda`, `cuda13`, `plots`, `dev`, `docs`) and a
  release workflow that publishes on `v*` tags.
- Test suite on mock clusters, synthetic DR11 files and the command line (98% line coverage);
  CI on Python 3.11 to 3.13. `tests/test_jax_build.py` checks in seconds that the installed JAX
  computes richness reproducibly: the conda-forge `jaxlib 0.10.2` CPU build does not, the PyPI
  wheels do, so install JAX from PyPI.
