# rema

[![PyPI](https://img.shields.io/pypi/v/rema-jax)](https://pypi.org/project/rema-jax/)
[![Python](https://img.shields.io/pypi/pyversions/rema-jax)](https://pypi.org/project/rema-jax/)
[![tests](https://github.com/JohanComparat/rema-jax/actions/workflows/tests.yml/badge.svg)](https://github.com/JohanComparat/rema-jax/actions/workflows/tests.yml)
[![coverage](https://codecov.io/gh/JohanComparat/rema-jax/graph/badge.svg)](https://codecov.io/gh/JohanComparat/rema-jax)
[![docs](https://readthedocs.org/projects/rema-jax/badge/?version=latest)](https://rema-jax.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/JohanComparat/rema-jax/blob/main/LICENSE)

`rema` finds galaxy clusters in the DESI Legacy Imaging Surveys
([Dey et al. 2019](https://doi.org/10.3847/1538-3881/ab089d)) Data Release 11 (DR11) with a
red-sequence matched filter written in JAX.
From the DR11 sweeps and randoms, it produces catalogues of clusters and of their member galaxies,
with the richness λ, the redshift z_λ and the centre of each cluster, found blindly or measured at
given positions.

`rema` is a redesign of [redMaPPer](https://github.com/erykoff/redmapper) (Rykoff et al.
[2014](https://doi.org/10.1088/0004-637X/785/2/104), [2016](https://doi.org/10.3847/0067-0049/224/1/1)).
It keeps redMaPPer's red-sequence model, matched-filter richness λ, z_λ, centring, percolation and
redshift scans at given positions, and adds the velocity clipping of
[Clerc et al. (2016)](https://doi.org/10.1093/mnras/stw2214).
The code is vectorised JAX: it runs on CPU or GPU, λ has exact implicit derivatives, and the package
installs with `pip`, without a compiler, Spark, Java, GSL, esutil or healsparse.

Versions 0.x are early public releases: the interface may still change before 1.0.
`rema` implements the ingestion of the DR11 sweeps, footprint and depth maps from the randoms, the
DR11 griz red-sequence calibration (`rema calibrate`), zred, backgrounds, richness, z_λ, centring on
the brightest cluster galaxy (BCG) or with wcen, blind mode, scan mode, spectroscopic
post-processing, and SLURM drivers for high-performance computing (HPC) centres.
Cluster finding from the DR11 photometric redshifts (photo-z), whatever the galaxies' colours, is
experimental: a photo-z filter in blind and scan mode (`--set model.filter=photoz`), and `rema pscd`
(Photo-z Space Cluster Detection, PSCD), an implementation of the Adaptive Matched Identifier of
Clustered Objects (AMICO) matched filter
([Bellagamba et al. 2018](https://doi.org/10.1093/mnras/stx2701);
[photo-z finders page](https://rema-jax.readthedocs.io/en/latest/photoz_finders.html)).

The [documentation](https://rema-jax.readthedocs.io/en/latest/) covers the installation (including
the notebook platform of the IN2P3 computing centre, CC-IN2P3), a tour on mock data, DR11 end to
end, HPC production, the DR11 catalogue and its validation, the reference, the API and the DR11
notebooks.
The [design page](https://rema-jax.readthedocs.io/en/latest/design.html) holds the design and
validation notes.

## The DR11 south catalogue

We ran the blind mode of `rema` 0.2.0 over Legacy Surveys DR11 south, on 18 500 deg² with
E(B−V) < 0.2 and Galactic latitude |b| ≥ 15°.
It found 3 481 608 clusters with λ ≥ 3 (145 133 with λ ≥ 20) and 87.7 M members.
The catalogue is on the CC-IN2P3 data system in
`/sps/lsst/datasets/desi/legacysurveys/dr11/south/rema/rema_dr11_v0.2.0/`.
Its z_λ scatter against spectroscopic redshifts is 0.0067 (1+z) below z = 0.6, and it recovers
93–95% of the Atacama Cosmology Telescope (ACT) and South Pole Telescope (SPT) clusters in its
footprint.
The [results page](https://rema-jax.readthedocs.io/en/latest/redmapper_dr11.html) compares it with
the redMaPPer, eROMaPPer, Sunyaev–Zel'dovich (SZ) and X-ray catalogues in 30 figures.

With `rema scan-positions`, we measured λ, z_λ, the optical centre and the spectroscopic redshift
at the 1.64 M clusters of ACT DR5/DR6, SPT-SZ, Planck PSZ2, MCXC, eRASS1, SDSS and DES redMaPPer
and [Wen & Han (2024)](https://doi.org/10.3847/1538-4365/ad409d).
The products are in
`/sps/lsst/datasets/desi/legacysurveys/dr11/south/rema/rema_dr11_v0.2.0_scan_public/`
([public catalogues page](https://rema-jax.readthedocs.io/en/latest/public_catalogues_scan.html)).

## Installation

> **Maintainer setup.** On the development laptop, use the shared `dev` environment defined in
> `~/software/dev_env` (`conda activate dev`); this package is already installed there in editable
> mode.
> Do not create a separate environment for it: add missing dependencies to `~/software/dev_env`
> and rebuild.

`rema-jax` needs Python ≥ 3.11:

```bash
pip install rema-jax                # CPU
pip install "rema-jax[cuda]"        # NVIDIA GPU, CUDA 12 JAX wheels ([cuda13] for CUDA 13)
```

The package is imported as `rema` and installs the `rema` command.
With the CUDA wheels installed, JAX uses the GPU unless `JAX_PLATFORMS=cpu` is set;
`JAX_PLATFORMS=cuda` requires the GPU.

For development, from a clone:

```bash
git clone https://github.com/JohanComparat/rema-jax.git && cd rema-jax
mamba env create -f environment.yml      # Python 3.12, jax[cuda12], optax, ggah_mod
mamba activate rema
pip install -e ".[dev]"
pytest -m "not slow"                       # unit tests; `pytest -m data` needs the local DR11 files
```

## The DR11 pipeline

Run the commands in a working directory; they read the sweeps and the photo-z sweeps from the DR11
layout (`.../south/sweep/11.0/` and `.../south/sweep/11.0-photo-z/`).

```bash
export LEGACYSURVEY_DIR=/path/to/legacysurvey   # CC-IN2P3: /sps/lsst/datasets/desi/legacysurveys
D=$LEGACYSURVEY_DIR/dr11/south

# 1. galaxies: cleaned, dereddened griz fluxes, z-band S/N >= 5, Z_SPEC from the photo-z sweeps
rema ingest $D/sweep/11.0/sweep-000m005-005p000.fits $D/sweep/11.0/sweep-000m010-005m005.fits \
     --out galaxies.fits

# 2. footprint: fraction unmasked and depth per HEALPix pixel, from the DR11 randoms
rema maps $D/randoms/randoms-south-1-0.fits --box 0 5 -10 0 --out footprint.fits

# 3. calibration on DR11 griz (spectroscopic seeds from Z_SPEC; red sequence, background,
#    zred and z_lambda corrections), with diagnostic plots
rema calibrate --galaxies galaxies.fits --footprint footprint.fits --out calib.fits --plots calib_plots
#    (or start from an existing redMaPPer calibration:
#     rema calib-import legacy_dr10_south_v0.3_grz_z_cal_iter1_pars.fit --out calib.fits
#     rema background --galaxies galaxies.fits --calib calib.fits --footprint footprint.fits --out calib.fits)

# 4a. scan mode at given positions (redMaPPer zscan), with spectroscopic post-processing
rema scan --galaxies galaxies.fits --calib calib.fits --footprint footprint.fits \
     --positions act_dr5.fits --ra-col ra --dec-col dec --id-col name --specpost --out scan.fits

# 4b. blind cluster finding (keep clusters centred in the own box); with --checkpoint an
#     interrupted run resumes after its last completed stage
rema blind --galaxies galaxies.fits --calib calib.fits --footprint footprint.fits \
     --own 0 5 -10 0 --checkpoint blind_ckpt --specpost --out clusters.fits

# 4b'. a table of positions anywhere in DR11 south (ingests what it needs): input + results
rema scan-positions targets.csv --out targets_rema.fits

# 4c. the same from the photo-z (any colour), and the AMICO-like PSCD
rema blind ... --set @scripts/photoz/photoz_dr11.yaml --set @scripts/photoz/pz_filter.yaml --out pz.fits
rema pscd --galaxies galaxies.fits --calib calib.fits --footprint footprint.fits \
     --own 0 5 -10 0 --set @scripts/photoz/photoz_dr11.yaml --out pscd.fits
```

The outputs are FITS files with `CLUSTERS` and `MEMBERS` header-data units (HDUs), plus the
configuration (`CONFIG`).
Column names follow redMaPPer (`LAMBDA`, `Z_LAMBDA`, `SCALEVAL`, `MASKFRAC`, ...); the
spectroscopic post-processing adds `SPEC_Z_BOOT`, `VDISP`, `BEST_Z` and others.
All settings live in a YAML configuration (`rema.config.RemaConfig`); write the defaults with
`python -c "from rema.config import RemaConfig; print(RemaConfig().to_yaml())"`.

## Catalogue, notebooks and runs at CC-IN2P3

- **Catalogue:** `/sps/lsst/datasets/desi/legacysurveys/dr11/south/rema/` holds the combined
  catalogue `rema_dr11_v0.2.0/`, the two parts of the run with their calibration and logs, and the
  external catalogues of the results page, as described on the
  [CC-IN2P3 page](https://rema-jax.readthedocs.io/en/latest/ccin2p3.html).
- **Notebooks:** the DR11 notebooks run on the Jupyter platform https://notebook.cc.in2p3.fr with a
  `rema` kernel registered once from an environment on `/sps`
  (see [the notebooks at CC-IN2P3](https://rema-jax.readthedocs.io/en/latest/install.html#cc-notebooks)).
- **Running the pipeline again:** the driver `scripts/slurm/rema_dr11_blind.sh` (`prepare`, `run`,
  `status`, `clean`) runs with the settings in `scripts/slurm/ccin2p3.env`.
  The commands are on the CC-IN2P3 page, and the
  [HPC page](https://rema-jax.readthedocs.io/en/latest/hpc.html) describes the driver.

## Credits and licence

The algorithms, scan mode (redMaPPer's zscan) included, follow redMaPPer
([Rykoff et al. 2014, ApJ 785, 104](https://doi.org/10.1088/0004-637X/785/2/104);
[Rykoff et al. 2016, ApJS 224, 1](https://doi.org/10.3847/0067-0049/224/1/1)), and the
spectroscopic post-processing follows
[Clerc et al. (2016, MNRAS 463, 4490)](https://doi.org/10.1093/mnras/stw2214).
Distances come from [ggah_mod](https://pypi.org/project/ggah_mod/).
`python -m rema.data.build` builds the m*(z) and colour tables in `rema/data` from redMaPPer's
m*(z), the [Bruzual & Charlot (2003)](https://doi.org/10.1046/j.1365-8711.2003.06897.x) simple
stellar population (SSP) grid distributed with EzGal
([Mancone & Gonzalez 2012](https://doi.org/10.1086/666502)) and the speclite filter curves; see
[rema/data/README.md](https://github.com/JohanComparat/rema-jax/blob/main/rema/data/README.md).
The package is distributed under the MIT licence.
