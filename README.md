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

- ingestion of DR11 sweeps, and footprint and depth maps from the randoms;
- zred, χ² and zred backgrounds, richness, z_λ, and BCG or wcen centring (redMaPPer's
  CenteringWcenZred, calibrated by `rema calibrate`);
- scan mode, blind mode and spectroscopic post-processing;
- the DR11 griz red-sequence calibration (`rema calibrate`).

On a 75 deg² development area (three DR11 sweeps, no footprint yet), blind mode takes 12 min
on a laptop GPU. Against the redMaPPer DR10 catalogue at 0.1 < z < 0.7 and λ ≥ 20, rema's λ
agrees to +4% ± 17% and z_λ to 0.001 ± 0.006, and 95% of rema's clusters are in DR10 within 3′.
z_λ matches the clusters' spectroscopic redshifts with NMAD 0.0065, and every ACT DR5 cluster
in the redshift range is detected. With wcen centring, the optical centres at the ACT clusters
match the redMaPPer DR10 ones for 80% of them. With the DR11 randoms footprint (mask and depth), z_λ matches
92 spectroscopic cluster redshifts with NMAD 0.0055. The full DR11 south run happens on an HPC, as
SLURM arrays of about 100 deg² regions driven by `scripts/slurm/rema_dr11_blind.sh` (see the
[HPC page](https://rema-jax.readthedocs.io/en/latest/hpc.html) of the documentation).

The documentation is at [rema-jax.readthedocs.io](https://rema-jax.readthedocs.io/en/latest/): install, a
tour on mock data, DR11 end to end, HPC production, reference, API and DR11 notebooks. Design
and validation notes are in [the design page](https://rema-jax.readthedocs.io/en/latest/design.html).

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
```

Outputs are FITS files with `CLUSTERS` and `MEMBERS` HDUs, plus the configuration (`CONFIG`).
Column names follow redMaPPer (`LAMBDA`, `Z_LAMBDA`, `SCALEVAL`, `MASKFRAC`, ...); the
spectroscopic post-processing adds `SPEC_Z_BOOT`, `VDISP`, `BEST_Z`, .... All settings live in a YAML configuration
(`rema.config.RemaConfig`; write the defaults with
`python -c "from rema.config import RemaConfig; print(RemaConfig().to_yaml())"`).

## Running at CC-IN2P3

The full DR11 south blind run on the [CC-IN2P3](https://doc.cc.in2p3.fr) SLURM cluster. The
inputs are read from `/sps/lsst/datasets/desi/legacysurveys/dr11/south`; the merged clusters
are written to `sweep/11.0-rm/` and the members to `sweep/11.0-rm-mem/`, next to `sweep/11.0`
and `sweep/11.0-photo-z`. `$HOME` is small: the Python environment, the caches and the work
directory all live on `/sps`.

**Once: install.**

```bash
ssh cca.in2p3.fr
SPS=/sps/lsst/users/$USER
git clone https://github.com/JohanComparat/rema-jax.git $HOME/software/rema-jax
# Miniforge and the environment on /sps (several GB with the CUDA libraries)
curl -L -o /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/mf.sh -b -p $SPS/miniforge3 && rm /tmp/mf.sh
source $SPS/miniforge3/etc/profile.d/conda.sh
conda create -y -p $SPS/envs/rema python=3.12 pip
conda activate $SPS/envs/rema
pip install --no-cache-dir -e "$HOME/software/rema-jax[cuda,plots]" ipykernel   # JAX from the PyPI wheels
# Jupyter kernel for the notebook platform, with the CC paths
python -m ipykernel install --user --name rema --display-name rema \
    --env LEGACYSURVEY_DIR /sps/lsst/datasets/desi/legacysurveys \
    --env JAX_COMPILATION_CACHE_DIR $SPS/.cache/rema/jax
```

Check that the GPU nodes see the GPU (one short job):

```bash
srun -p gpu_v100 --gpus 1 -t 0-00:10 -c 4 --mem 8G -L sps \
     python -c "import jax; print(jax.devices())"                  # [CudaDevice(id=0)]
```

**Each run.** In a fresh login shell:

```bash
source /sps/lsst/users/$USER/miniforge3/etc/profile.d/conda.sh
conda activate /sps/lsst/users/$USER/envs/rema       # the jobs inherit this environment
export OUTDIR=/sps/lsst/users/$USER/rema_dr11_v0.1.0  # one work directory per calibration and version
source $HOME/software/rema-jax/scripts/slurm/ccin2p3.env
D=$REMA/scripts/slurm

# 1. ingest every sweep and index the randoms (CPU arrays on htc)
$D/rema_dr11_blind.sh prepare
$D/rema_dr11_blind.sh status                          # or: squeue -u $USER
# 2. when the ingest is done, choose a calibration area among the suggested boxes ...
rema regions --galaxies $OUTDIR/galaxies --calib-suggest 400 --config $D/dr11_south.yaml
# ... and calibrate on it (one CPU job)
CALIB_BOX="150 170 -5 15" $D/rema_dr11_blind.sh prepare
# 3. check $OUTDIR/calib/plots and the header of $OUTDIR/calib/calib.fits, then run the
#    regions (one V100 job per region) and the merge
export CALIB=$OUTDIR/calib/calib.fits
$D/rema_dr11_blind.sh run
$D/rema_dr11_blind.sh status
# 4. once the merge job is done: delete the intermediate products (galaxy tables, randoms
#    index, JAX cache); the plan, calibration, logs and region catalogues are kept
$D/rema_dr11_blind.sh clean
```

- `prepare` and `run` submit only what is missing or stale: after failures or timeouts, call
  them again (in a shell set up as above, with the same `OUTDIR` and `CALIB`).
- Parallel jobs never write the same file: one galaxy table per sweep, one index per randoms
  file, one directory per region, one log per task. Each region task compiles into a private
  copy of the JAX cache and deletes its checkpoint once its catalogue is written.
- The merged products in `sweep/11.0-rm*/` have fixed names (`clusters_dr11.fits`,
  `clusters_dr11_members.fits`): a later run overwrites them. For a test, export
  `CLUSTERS_DIR=$OUTDIR MEMBERS_DIR=$OUTDIR` before sourcing `ccin2p3.env`.
- `ccin2p3.env` lists every setting (partitions, `--licenses=sps`, `NRAND=4` for the four
  randoms files on `/sps`); any of them can be exported before sourcing it. To use the H100
  nodes: `PART_GPU=gpu_h100 REGION_CPUS=12 REGION_MEM=96G`. See the HPC page of the documentation.

## Credits

The algorithms follow redMaPPer (Rykoff et al. 2014, ApJ 785, 104; Rykoff et al. 2016,
ApJS 224, 1), scan mode included (redMaPPer's zscan). The spectroscopic post-processing follows
Clerc et al. (2016, MNRAS 463, 4490). Distances come from
[ggah_mod](https://pypi.org/project/ggah_mod/). The m*(z) and colour tables in `rema/data` are built
from redMaPPer's m*(z), the Bruzual & Charlot (2003) SSP grid distributed with EzGal and the
speclite filter curves (`python -m rema.data.build`); see [rema/data/README.md](https://github.com/JohanComparat/rema-jax/blob/main/rema/data/README.md).

MIT licence.
