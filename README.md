# rema

[![PyPI](https://img.shields.io/pypi/v/rema-jax)](https://pypi.org/project/rema-jax/)
[![Python](https://img.shields.io/pypi/pyversions/rema-jax)](https://pypi.org/project/rema-jax/)
[![tests](https://github.com/JohanComparat/rema-jax/actions/workflows/tests.yml/badge.svg)](https://github.com/JohanComparat/rema-jax/actions/workflows/tests.yml)
[![coverage](https://codecov.io/gh/JohanComparat/rema-jax/graph/badge.svg)](https://codecov.io/gh/JohanComparat/rema-jax)
[![docs](https://readthedocs.org/projects/rema-jax/badge/?version=latest)](https://rema-jax.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/JohanComparat/rema-jax/blob/main/LICENSE)

Red-sequence matched-filter galaxy cluster finding in JAX, for the DESI Legacy Imaging Surveys
DR11.

rema is a redesign of [redMaPPer](https://github.com/erykoff/redmapper) (Rykoff et al.
2014, 2016). It keeps the science: red-sequence model, matched-filter richness λ, z_λ, centring, percolation, redshift scans at
given positions, and Clerc et al. (2016) velocity clipping. The code is vectorised JAX. It
runs on CPU or GPU, it is differentiable (λ carries exact implicit derivatives), and it installs
with `pip`. It needs no compiler, Spark, Java, GSL, esutil or healsparse.

Status: an early public release (0.x); the interface may still change before 1.0. Implemented:
ingestion of DR11 sweeps, footprint and depth maps from the randoms, the DR11 griz red-sequence
calibration (`rema calibrate`), zred, backgrounds, richness, z_λ, BCG or wcen centring, blind
mode, scan mode and spectroscopic post-processing, and SLURM drivers for the HPC. Experimental:
cluster finding from the DR11 photo-z, whatever the galaxies' colours, with a photo-z filter in
blind and scan mode (`--set model.filter=photoz`) and `rema pscd`, an implementation of the
AMICO matched filter ([photo-z finders page](https://rema-jax.readthedocs.io/en/latest/photoz_finders.html)).

**The DR11 south catalogue is done.** The blind run of rema 0.2.0 over Legacy Surveys DR11 south
found 3,481,608 clusters with λ ≥ 3 (145,133 with λ ≥ 20) and 87.7 M members over 18,500 deg²
(E(B−V) < 0.2, |b| ≥ 15°). It is on the CC-IN2P3 data system in
`/sps/lsst/datasets/desi/legacysurveys/dr11/south/rema/rema_dr11_v0.2.0/`. Its z_λ scatter
against spectroscopic redshifts is 0.0067 (1+z) below z = 0.6, and it recovers 93–95% of the ACT
and SPT clusters in its footprint; the
[results page](https://rema-jax.readthedocs.io/en/latest/redmapper_dr11.html) compares it with
the redMaPPer, eROMaPPer, SZ and X-ray catalogues in 30 figures.

The documentation is at [rema-jax.readthedocs.io](https://rema-jax.readthedocs.io/en/latest/): install
(including the CC-IN2P3 notebook platform), a tour on mock data, DR11 end to end, HPC production,
the DR11 catalogue and its validation, reference, API and DR11 notebooks. Design and validation
notes are in [the design page](https://rema-jax.readthedocs.io/en/latest/design.html).

## Install

```bash
pip install rema-jax                # CPU
pip install "rema-jax[cuda]"        # NVIDIA GPU, CUDA 12 JAX wheels ([cuda13] for CUDA 13)
```

The package is imported as `rema` and installs the `rema` command. Python 3.11 or later.
When the CUDA wheels are installed, JAX uses the GPU unless `JAX_PLATFORMS=cpu` is set; set
`JAX_PLATFORMS=cuda` to require it.

For development:

> **Maintainer setup.** On the development laptop, use the shared `dev` environment defined in `~/software/dev_env` (`conda activate dev`); this package is already installed there in editable mode. Do not create a separate environment for it: add missing dependencies to `~/software/dev_env` and rebuild.

```bash
git clone https://github.com/JohanComparat/rema-jax.git && cd rema-jax
mamba env create -f environment.yml      # Python 3.12, jax[cuda12], optax, ggah_mod
mamba activate rema
pip install -e ".[dev]"
pytest -m "not slow"                       # unit tests; `pytest -m data` needs the local DR11 files
```

## End-to-end on DR11

Run in a working directory. Sweeps and photo-z sweeps are read from the DR11 layout
(`.../south/sweep/11.0/` and `.../south/sweep/11.0-photo-z/`).

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

# 4c. the same from the photo-z (any colour), and the AMICO-like PSCD
rema blind ... --set @scripts/photoz/photoz_dr11.yaml --set @scripts/photoz/pz_filter.yaml --out pz.fits
rema pscd --galaxies galaxies.fits --calib calib.fits --footprint footprint.fits \
     --own 0 5 -10 0 --set @scripts/photoz/photoz_dr11.yaml --out pscd.fits
```

Outputs are FITS files with `CLUSTERS` and `MEMBERS` HDUs, plus the configuration (`CONFIG`).
Column names follow redMaPPer (`LAMBDA`, `Z_LAMBDA`, `SCALEVAL`, `MASKFRAC`, ...); the
spectroscopic post-processing adds `SPEC_Z_BOOT`, `VDISP`, `BEST_Z`, .... All settings live in a YAML configuration
(`rema.config.RemaConfig`; write the defaults with
`python -c "from rema.config import RemaConfig; print(RemaConfig().to_yaml())"`).

## At CC-IN2P3

- **The catalogue:** `/sps/lsst/datasets/desi/legacysurveys/dr11/south/rema/` holds the combined
  catalogue `rema_dr11_v0.2.0/`, the two parts of the run with their calibration and logs, and
  the external catalogues of the results page. See the
  [CC-IN2P3 page](https://rema-jax.readthedocs.io/en/latest/ccin2p3.html).
- **Notebooks:** the DR11 notebooks run on the Jupyter platform https://notebook.cc.in2p3.fr with
  a `rema` kernel registered once from an environment on `/sps`; see
  [the notebooks at CC-IN2P3](https://rema-jax.readthedocs.io/en/latest/install.html#cc-notebooks).
- **Running it again:** `scripts/slurm/rema_dr11_blind.sh` with `scripts/slurm/ccin2p3.env`
  (`prepare`, `run`, `status`, `clean`); the commands are on the CC-IN2P3 page and the driver on
  the [HPC page](https://rema-jax.readthedocs.io/en/latest/hpc.html).

## Credits

The algorithms follow redMaPPer (Rykoff et al. 2014, ApJ 785, 104; Rykoff et al. 2016,
ApJS 224, 1), scan mode included (redMaPPer's zscan). The spectroscopic post-processing follows
Clerc et al. (2016, MNRAS 463, 4490). Distances come from
[ggah_mod](https://pypi.org/project/ggah_mod/). The m*(z) and colour tables in `rema/data` are built
from redMaPPer's m*(z), the Bruzual & Charlot (2003) SSP grid distributed with EzGal and the
speclite filter curves (`python -m rema.data.build`); see [rema/data/README.md](https://github.com/JohanComparat/rema-jax/blob/main/rema/data/README.md).

MIT licence.
