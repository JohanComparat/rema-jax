# Changelog

## [0.5.2] - 2026-10-13

### Documentation

- Second reading of every page against the house style, with the statements checked against the
  code and the generated tables:
  - the centring probability of the results page is the measured ⟨P_cen⟩ = 0.81 (0.82 is the
    prediction of Rykoff et al. 2016);
  - the tier-B numbers of the cosmology page match its values (4 223 clusters with λ ≥ 20,
    +0.8% for Ω_m = 0.35, counts within 1% and 3%), and the priors are those of
    `rema.abundance.likelihood`;
  - the photo-z pages match their tables (PSCD recovers 16–25% of the Wen & Han clusters,
    3–17% photo-z outliers fainter than z = 21);
  - a region is stale when it was made with another plan or calibration (the driver does not
    check the rema version);
  - the public-catalogue scan example uses the `inputs/full` and `out/full` layout that
    `scripts/scan/finalize_scan.sh` requires;
  - the checkpoint key, the `w` drop reason of percolation, the `--work` fallback of
    `rema scan-positions`, the `none` value of `BEST_Z_TYPE`, the MW_TRANSMISSION > 0 cut of
    ingest and the chunks of seven redshifts of the z_λ grid are documented.
- The randoms index takes 3.2 GB per DR11 randoms file, 12.7 GB for four (measured at
  CC-IN2P3; the pages said 8 GB).
- The generated tables and their scripts group thousands with a narrow no-break space.
- The public-catalogue scan ran on 2026-10-11 and 12.

## [0.5.1] - 2026-10-12

### Documentation

- Citations through sphinxcontrib-bibtex (new `docs` dependency): `docs/references.bib` holds
  the 60 cited papers, built from their Crossref and arXiv records, and the new Literature page
  lists them with a link to the published version (DOI), or to arXiv for a preprint.
  The pages cite them in author–year form with pinpoints (`:cite:t:`, `:cite:p:`); the README
  links to the DOIs.
- The README, the documentation pages, the CHANGELOG, the notebook markdown and the docstrings
  follow the author's house style: pages open on what they take and produce, one sentence per
  source line, British spelling, small counts in words, thousands grouped with a narrow
  no-break space, descriptive headings.
- References added for the methods named without one: asinh magnitudes, the gapper and biweight
  estimators, HEALPix, the Navarro–Frenk–White (NFW) profile, the Bruzual & Charlot (2003) models
  (BC03), EzGal, L-BFGS, the No-U-Turn Sampler (NUTS) and super-sample variance.
- Corrected statements about the cited papers:
  - Kluge et al. (2024) define λ_norm = 1.039 λ_griz, so rema's λ is 17% above their griz λ (the
    documentation said 9%);
  - the centring section is Sect. 8 of Rykoff et al. (2014) (the documentation attributed it to
    Rykoff et al. 2016);
  - the Atacama Cosmology Telescope (ACT) DR6 cluster catalogue is by the ACT-DES-HSC
    Collaboration (OJAp 9, 2026);
  - the Ω_b h² prior of the counts likelihood (0.02237 ± 0.00015) is the Planck 2018 value (the
    documentation called it a Big Bang nucleosynthesis prior);
  - the g − r scatter of Comparat et al. (2025) is 0.042–0.059;
  - Fig. 9 of Bleem et al. (2020) shows λ against the Sunyaev–Zel'dovich (SZ) significance;
  - journal references: Euclid Q1 clusters (A&A 711, A34), Myles et al. 2025 (MNRAS 544, 2080),
    Lee et al. 2025 (PRD 111, 063502).

### Fixed

- The figure of λ against velocity dispersion on the redMaPPer results page divides the
  Kluge et al. (2024) relation by 1.039 instead of multiplying it
  (`docs/figures/redmapper_dr11/fig_richness.py`); the figure and the notebook are updated.

## [0.5.0] - 2026-10-12

### Added

- The DR11 scan of the public cluster catalogues (documentation page "Public cluster catalogues
  in DR11"): λ, z_λ, the optical centre and the spectroscopic redshift at every cluster of ACT
  DR5/DR6, SPT-SZ 2500d, Planck PSZ2, MCXC, eRASS1, SDSS DR8 and DES Y1 redMaPPer and
  Wen & Han (2024), 1.64 M positions, on the data system of the IN2P3 computing centre (CC-IN2P3)
  (`dr11/south/rema/rema_dr11_v0.2.0_scan_public`).
  `scripts/scan/finalize_scan.sh` joins, checks and copies the products;
  `docs/figures/public_scan/make_figures.py` makes the page's figures.
- `rema scan-positions` (`rema.modes.positions`): scan mode and spectroscopic post-processing for a
  table of positions (FITS, CSV, ...), joined to it row for row (input columns as `INPUT_<NAME>`,
  `SCANNED`).
  It ingests the sweeps around the positions (kept in a work directory), indexes the randoms of
  their boxes and builds the footprints itself, with the DR11 south production calibration by
  default.
  `scripts/scan/scan_positions.sh` submits it as a SLURM job at CC-IN2P3.
  `--index` reuses an existing randoms index.
  `scripts/scan/public_catalogues.py` writes the public cluster catalogues (ACT DR5/DR6, SPT-SZ,
  PSZ2, MCXC, eRASS1, SDSS and DES redMaPPer, Wen & Han 2024) as inputs, large ones in right
  ascension (RA) slices that `scripts/scan/merge_parts.py` joins back.

### Fixed

- `Region.query` with a magnitude limit brighter than every galaxy of the region (a group of
  positions at a survey edge) returns no neighbours instead of crashing with an IndexError.
- `rema.modes.specpost.process` no longer fails on a catalogue without any member (an index array
  of float type), which crashed `rema scan --specpost` on positions without members.

## [0.4.0] - 2026-10-08

### Added

- Photometric-redshift (photo-z) cluster finding (documentation pages "Photo-z cluster finding on
  DR11", the method and the figures, and "Photo-z study: all the findings", every result, issue
  and open question):
  - a photo-z filter for `rema blind` and `rema scan` (`model.filter: photoz`): members from the
    DR11 photo-z (ZPHOT, ZPHOT_STD of the ingested tables) instead of the red-sequence colours,
    any colour; the field is the region's stacked photo-z distributions
    (`rema.model.background.build_photoz_bkg`); centring on the brightest cluster galaxy (BCG);
    seeds of any colour (`seeds.zphot_err_max`); an optional significance floor of the candidates
    (`richness.min_lnlamlike`);
  - photo-z widths calibrated against spectroscopic redshifts (`photoz.err_scale`, piecewise
    linear in magnitude; `rema.validate.photoz.fit_err_scale`, `scripts/photoz/fit_errscale.py`)
    and an extra magnitude limit of the members (`photoz.mag_max`, also in the completeness);
  - `rema pscd` (`rema.pscd`): Photo-z Space Cluster Detection (PSCD), an implementation of the
    Adaptive Matched Identifier of Clustered Objects (AMICO) matched filter (Bellagamba et al.
    2018; Maturi et al. 2019) with the Kilo-Degree Survey (KiDS) template, masked and
    depth-dependent normalisation, extraction with cleaning and membership probabilities;
  - null tests (`null.shuffle: photoz | colour`, `rema.validate.null`) and the comparison of
    finders at matched false-detection rates (`rema.validate.compare.match_physical`,
    `null_threshold`); drivers `scripts/photoz/photoz_strip.sh` (laptop) and
    `scripts/photoz/photoz_test.sh` (CC-IN2P3); figures in `docs/figures/photoz_finders/`;
  - `--set SECTION.KEY=VALUE` and `--set @overlay.yaml` override any configuration value
    (`rema.config.apply_overrides`); `SNR` = sqrt(2 LNLAMLIKE) in every catalogue; `FILTER` (and
    `NULL`) in the primary header; the members' and central's ZPHOT columns when the table has
    them; mocks with photo-z, true redshifts and blue members (`GaussianPhotoz`,
    `mock_template_cluster`).

### Changed

- The DR11 notebooks, the redMaPPer figure scripts and the cosmology sensitivity scripts use two
  directories, given by two environment variables.
  `REMA_DATA` is shared and read only: the DR11 south directory, with the production run and the
  external catalogues in `rema/`; it defaults to the data system at CC-IN2P3, else
  `~/data/legacysurvey/dr11/south`.
  `REMA_WORK` is writable: it defaults to `/sps/lsst/users/$USER/rema_work` when that user space
  exists, else `~/rema_work`, and the notebooks' JAX compilation cache defaults to the same space.
  They replace `REMA_DR11_DIR` (or `LEGACYSURVEY_DIR`), `REMA_PRODUCTS`, `REMA_CALIB`,
  `REMA_RANDOMS`, `REMA_EXTERNAL`, `REMA_KLUGE_PARS` and `REMA_COSMO` there; the command line, the
  SLURM scripts and the tests are unchanged.

### Fixed

- Without `REMA_WORK`, the notebooks no longer write next to the production run, which only its
  owner can write to (`Permission denied` for everyone else: scan notebook, ingest cell).
- `scripts/cosmo_sens/cosmo_sens.sh` puts the jobs' JAX compilation cache in `$OUTDIR/jax_cache`
  instead of `~/.cache/rema/jax` (4 GB in the 20 GB `$HOME` at CC-IN2P3).
- Backgrounds built from the galaxy table (no background in the calibration, or
  `rema background`) use the area of the footprint's pixels in the data box.
  They used the effective area of the whole footprint, also when the galaxies were read over a
  smaller data box, so the background density was too low by the ratio of the areas.
- Configuration values like `1e6` (a string in YAML 1.1) are read as numbers for float keys.

## [0.3.2] - 2026-10-08

### Added

- Cosmology sensitivity study (documentation page "Cosmology and the DR11 cluster counts"):
  - the cosmology of a run is configurable beyond Ω_m and h (`Omega_b`, `sum_mnu`, `w0`, `wa`; old
    configurations still load) and can be changed on the command line with
    `--cosmology KEY=VALUE` (blind, scan, background, zred, calibrate, remeasure);
  - `CosmoTable` is built from a ggah_mod cosmology inside a trace (`from_cosmology`), its
    parameters are leaves (another cosmology does not recompile the kernels), and
    `CosmoTable.jvp` gives its derivatives with respect to the cosmological parameters;
  - `Region.with_cosmology`: the same region in another cosmology (zred recomputed; optionally
    m* following the luminosity distance);
  - `rema remeasure` (`rema.modes.remeasure`): the clusters of a catalogue re-measured at fixed
    centres on a grid of cosmologies, with the catalogue's free fractions from its members table,
    and the derivatives d ln λ/dθ by autodiff and by finite differences;
  - `rema.abundance`: the finder's response table, the z_vlim area map, the data vector N(λ, z), a
    counts model (ggah_mod Tinker et al. 2008 mass function, log-normal mass–richness relation,
    photo-z kernel, the finder's response), Poisson and super-sample covariance, the likelihood
    with the DES Y1 weak-lensing calibration of McClintock et al. (2019), Fisher matrices, best fit
    and NUTS (optional extra `cosmo`: blackjax);
  - `scripts/cosmo_sens`: the study's scripts (DR11 fit and forecast, response table, SLURM driver
    for the re-measurements and re-runs on CC-IN2P3).
- `mocks.mock_cluster(..., extent=)`: members out to several r_λ.

### Changed

- Catalogues and calibrations carry the cosmology in their primary header (OMEGAM, HUBBLE, OMEGAB,
  MNU, W0, WA); `rema merge` keeps the regions' configuration (it wrote none before).
- The configuration has four more cosmology keys: rema 0.3.2 reads the configurations of earlier
  versions, but earlier versions reject those it writes (unknown keys).

### Fixed

- `scripts/dr11/combine_dr11.py`: the combined catalogue's NCLUSTER and NMEMBER are those of both
  parts (they were part 0's), and it carries the CONFIG table and OMEGAM, HUBBLE.
  The published DR11 south v0.2.0 combined catalogue was patched accordingly (headers only, data
  unchanged).

### Documentation

- "Cosmology and the DR11 cluster counts": the findings of the sensitivity study (tiers A, B and C
  on six DR11 regions, the red-sequence calibration in other cosmologies, the counts model, the
  DR11 fit and the forecast of the bias of ignoring the finder's response).

## [0.3.1] - 2026-10-07

### Fixed

- The `batch` stage of `scripts/slurm/rema_task.sh` (the region tasks grouped under a site's job
  limit, `MAX_ARRAY`) runs JAX on the GPU, like the `region` stage, instead of the CPU.
  In the DR11 south production, the batched regions took 4 h each on the five CPU cores of their
  V100 node, against about 1 h on the GPU; the catalogue does not depend on the device.
- `rema merge --glat-min`: the quality assurance (QA) counts the members after the cut.

### Added

- `scripts/dr11/combine_dr11.py` and `scripts/dr11/finalize_rema.sh`, which combined the two parts
  of the DR11 south production and moved them to the data system.

### Documentation

- The CC-IN2P3 page describes the finished DR11 south catalogue (3 481 608 clusters,
  18 500 deg²), how it was made and how to run the pipeline again.
  The high-performance computing (HPC) page and the design notes give the measured production
  costs, and the README and the front page lead with the catalogue.
- The setup of the notebooks on the CC-IN2P3 Jupyter platform (https://notebook.cc.in2p3.fr) is a
  section of the Installation page, linked from the front page.
- The redMaPPer figures page uses both parts of the run.

## [0.3.0] - 2026-10-07

### Added

- Documentation page "redMaPPer blind mode on DR11" (`docs/redmapper_dr11.rst`): the DR11 south
  production catalogue in 30 figures of the redMaPPer papers (Rykoff et al. 2014, 2016;
  Kluge et al. 2024) and of later validations (red sequence, z_λ against spectroscopic redshifts,
  richness against DES Y1, SDSS DR8 and eRASS1, depth and z_vlim maps, abundance, centring against
  X-ray and SZ centres, recovery of ACT, SPT, Planck, eRASS1, MCXC and Wen & Han clusters,
  membership against spectroscopy), with a table of numbers next to the published values and a
  literature list.
  The scripts are in `docs/figures/redmapper_dr11` (`make_all.py` runs them).
- Notebook `docs/notebooks/redmapper_dr11.ipynb`, assembled from those scripts by
  `build_notebooks.py redmapper`; `build_notebooks.py --kernel` chooses the Jupyter kernel.
- Documentation (CC-IN2P3): running the notebooks on the Jupyter platform
  https://notebook.cc.in2p3.fr, with a `rema` kernel registered once with the paths of the data
  system (`ipykernel install --env ...`), the session settings of each notebook and
  troubleshooting.
  The figure scripts write to `$REMA_WORK/redmapper` like the other notebooks, look for the
  external catalogues in `$REMA_PRODUCTS/external` (all of them in that one directory, at CC-IN2P3
  next to the products), stop with a clear message when the work directory is not writable, and
  run without `colossus` or `pandas` (their panels are left out).

### Fixed

- `rema_dr11_blind.sh status` no longer fails with a traceback during `prepare`, before the region
  plan exists: it reports the ingest (galaxy tables done in the area), the randoms indexes and the
  calibration.
  `rema todo` prints `ingest_done` and `randoms_done`, and `rema status` exits with a message when
  the plan is missing.
- Galactic latitude cut: `rema regions --glat-min B` leaves out the regions lying entirely at
  |b| < B, and `rema merge --glat-min B` keeps the clusters (and members) at |b| ≥ B, recorded as
  `GLATMIN` and in the QA (`n_clusters_low_glat`).
  The driver passes `GLAT_MIN` to both.
  Near the Galactic plane, stellar contamination lets most seeds survive the first pass (76%
  instead of about a third), and percolation ran out of memory there.
- The region plan no longer stops on tiles that have DR11 randoms but no DR11 sweep (edge slivers,
  e.g. eight tiles with 0.6 deg² of randoms between RA 0 and 240).
  `rema regions --sweeps DIR`, which the task script passes, treats tiles absent from the sweep
  directory and its checksum lists as outside the data and logs them; a listed sweep without a
  galaxy table is still an error.
  `ALLOW_UNCOVERED=1` passes `--allow-uncovered`.
- With `MAX_ARRAY` (90 in `ccin2p3.env`), the driver gives each array task several regions, run in
  turn by the new `batch` stage with a scaled time limit: the GPU quality of service at CC-IN2P3
  refuses more than 100 queued jobs per user, and it rejected the 197-task region array of the
  RA 0–240 run.
  The driver stops when an `sbatch` fails, instead of going on with an empty job id.
- `rema_dr11_blind.sh run` can be submitted while the calibration job of `prepare` is still queued
  or running (`CALIB=$OUTDIR/calib/calib.fits`): the regions and the merge wait for it
  (`afterok`).
  `status` reports that job instead of "calibration: none".

### Documentation

- The DR11 notebooks read the data system at CC-IN2P3 by default: the sweeps, every randoms file
  present (the footprints of the production run), and the calibration and merged catalogues of the
  DR11 south production run (`dr11/south/rema/`).
  The blind and scan notebooks compare their clusters with the production catalogue.
- New page "DR11 blind run at CC-IN2P3": install, data download and completeness check, a run on
  part of the sky (`AREA_BOX`), the full sky, the cluster's rules, monitoring, and the validation
  on the development area.

## [0.2.0] - 2026-10-05

Model tables for LSST and Euclid, the calibration's colour template in the configuration, and the
fixes found by the first runs at CC-IN2P3.

### Model tables

- LSST and Euclid tables in `rema/data`, built by `python -m rema.data.build` like the DECam ones:
  redMaPPer's m*(z) in LSST i, r, z (`mstar_lsst_{i,r,z}03`); the BC03 passive population's m*(z)
  in every LSST band (`mstar_lsst_{u,g,r,i,z,y}_ezgal`, lsst/throughputs 1.9) and every Euclid band
  (`mstar_euclid_{vis,y,j,h}_ezgal`), and its adjacent colours (`colors_bc03_lsst_ugrizy`,
  `colors_bc03_euclid_visyjh`), at z = 0.01–2.50.
  The builder works per filter set (`FILTER_SETS`, `build_bc03_tables`, `build_mstar_redmapper`);
  the DECam tables are unchanged.
- `model.template` chooses the colour table that seeds the calibration (default
  `bc03_legacy_grizw1`; `bc03_lsst_ugrizy`, `bc03_euclid_visyjh` or a FITS file).
  `RSModel.from_template` reads the template's bands from its `BANDS` header and rejects bands that
  are not among them.
  Calibrations written by 0.2.0 store `model.template` in their configuration, which rema 0.1.0
  cannot read.

### Documentation

- New page "Model tables": every packaged table with its bands, redshift range and source, figures
  and values, and how to calibrate with LSST or Euclid photometry or another band combination.

### HPC

- CC-IN2P3: region tasks request their GPU with `--gpus 1`, since the cluster rejects `--gres`.
  New driver variable `GPUS` (sends `--gpus=N` instead of `--gres=GPU_GRES`).
  `ccin2p3.env` sets `GPUS=1` with 5 CPUs and 45G per V100 task, the cluster's per-GPU CPU limit,
  so four tasks share a node.
  It also keeps JAX on the CPU on login and CPU nodes.
- `get_dr11_south_sweeps.sh`: `SUBDIRS=11.0-photo-z` gets only the photo-z files, in a second job
  next to the one getting the sweeps.
- `AREA_BOX` also restricts the driver's bookkeeping (`rema todo --box`), so `prepare` and `run`
  can process part of the sky, for a test or while the download is incomplete.

## [0.1.0] - 2026-10-05

First public release, on PyPI as `rema-jax` (`pip install rema-jax`; the package is imported as
`rema` and installs the `rema` command).
Versions below 1.0 may still change the interface.
Validated on a 75 deg² DR11 area: blind mode against redMaPPer DR10, scan mode at ACT DR5
clusters, z_λ against spectroscopic cluster redshifts (`docs/design.md`).

### Model and algorithms

- Red-sequence model (`RSModel`) with unconstrained parameters (log scatter, partial
  correlations), a BC03 griz template and a reader for redMaPPer `*_pars.fit` files.
- Photometric likelihood in asinh magnitudes (default) or in magnitudes (redMaPPer).
- zred with redMaPPer's posterior, parabola refinement and corrections; χ² and zred backgrounds.
- Richness solve with exact implicit derivatives (`jax.lax.custom_root`), deterministic mask and
  depth completeness from a footprint map, z_λ iteration with p(z).
- Centring on the brightest central candidate (BCG) or with redMaPPer's wcen model
  (`centering.method`: "auto" uses wcen when the calibration holds a wcen model).
  Up to five centre candidates per cluster (ID_CENT, P_CEN, ...), kept in the member table
  (CENT_RANK); the central-galaxy likelihood LNCGLIKE enters the likelihood pass.
- Blind mode: seeds, first pass, likelihood pass, percolation with redMaPPer's rejections
  (computed in batches of independent candidates, with the result of the sequential loop),
  consolidation, and checkpoints (`rema blind --checkpoint`).
- Scan mode at given positions (redMaPPer's zscan).
- Spectroscopic post-processing: Clerc et al. (2016) velocity clipping, bootstrap errors, BEST_Z.
- Calibration on DR11 (`rema calibrate`): initial model from spectroscopic galaxies,
  expectation–maximisation (EM) fit of the red-sequence nodes, zred and z_λ corrections (with the
  z → zred_uncorr mapping), wcen calibration (redMaPPer's WcenCalibrator), diagnostic plots.
  `rema calib-import` starts from a redMaPPer calibration instead.

### Data and production

- Legacy Surveys DR11 ingestion: sweeps plus row-matched photo-z sweeps reduced to per-sweep galaxy
  tables (dereddened griz fluxes, z-band S/N ≥ 5, `ZSPEC`), with explicit MASKBITS, TYPE, NOBS and
  E(B−V) cuts.
- Footprint and depth maps from the DR11 randoms, through a pixel-sorted randoms index read once
  (`rema randoms-index`, `rema maps`).
- Full-sky production: the region planner (`rema regions`), `rema status`, `rema todo` and
  `rema merge` with QA (duplicates, centres outside their box, close pairs across regions,
  provenance checks).
  Catalogues record their provenance (plan, calibration and configuration hashes, device, boxes)
  in the header.
- SLURM driver (`scripts/slurm/rema_dr11_blind.sh`: prepare, run, status, clean) with the stage
  script `rema_task.sh` and the production configuration `dr11_south.yaml`; only the region runs
  use the GPU.
  Settings for the CC-IN2P3 cluster in `scripts/slurm/ccin2p3.env` (`LEGACYSURVEY_DIR`,
  `CLUSTERS_DIR`, `MEMBERS_DIR`, partitions), and download scripts for the DR11 south sweeps and
  randoms.
- Model tables (`rema/data`): redMaPPer's m*(z), and the m*(z) and colours of a passive
  Bruzual & Charlot (2003) population computed in JAX (`rema.model.sps`).
  `python -m rema.data.build` rebuilds them from public inputs at fixed commits (EzGal's simple
  stellar population grid, speclite filters, redMaPPer), checked by SHA-256.
- Configuration as nested frozen dataclasses with YAML input and output; defaults follow redMaPPer
  v0.7.7.
  The calibration and the catalogues store the configuration they used.

### Documentation, packaging and tests

- Sphinx documentation (Read the Docs): install, a tour on mock data, DR11 end to end, HPC
  production, reference, API, and three executed DR11 notebooks (blind run, pipeline stages, scan
  mode).
- Packaging for PyPI (`rema-jax`, extras `cuda`, `cuda13`, `plots`, `dev`, `docs`) and a release
  workflow that publishes on `v*` tags.
- Test suite on mock clusters, synthetic DR11 files and the command line (98% line coverage);
  continuous integration on Python 3.11–3.13.
  `tests/test_jax_build.py` checks in seconds that the installed JAX computes richness
  reproducibly: the conda-forge `jaxlib 0.10.2` CPU build does not, the PyPI wheels do, so install
  JAX from PyPI.
