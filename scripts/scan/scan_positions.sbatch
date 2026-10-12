#!/bin/bash
# rema scan-positions as a SLURM batch job at CC-IN2P3: scan mode + spectroscopic post-processing
# at the positions of a table, on the DR11 south data, with the production calibration.
#
#   mkdir -p /sps/lsst/users/$USER/rema_scan/{targets,results,logs}
#   cp targets.fits /sps/lsst/users/$USER/rema_scan/targets/
#   sbatch $HOME/software/rema-jax/scripts/scan/scan_positions.sbatch targets.fits [--ra-col ra --dec-col dec ...]
#
# INPUT: a file name in $SCAN/targets (or any path); OUTPUT: $SCAN/results/<input stem>_rema.fits
# (set OUT=... to change it). The other arguments go to rema scan-positions (e.g. --ra-col ra
# --dec-col dec, --no-specpost, --set ...). SCAN = /sps/lsst/users/$USER/rema_scan; its galaxies/
# (the ingested sweeps, ~49 MB each) are kept and reused by later jobs. For more than a few thousand
# positions spread over the sky, raise --mem to 90G and --time (docs: ccin2p3, "Scan mode at given positions").
#SBATCH --job-name=scan-positions
#SBATCH --partition=gpu_v100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=12:00:00
#SBATCH --licenses=sps
#SBATCH --output=/sps/lsst/users/%u/rema_scan/logs/%x-%j.log
set -euo pipefail

SPS=/sps/lsst/users/$USER
SCAN=$SPS/rema_scan
DR11=/sps/lsst/datasets/desi/legacysurveys/dr11/south
CALIB=$DR11/rema/rema_dr11_v0.2.0_ra0-240/calib/calib.fits

IN=${1:?usage: sbatch scan_positions.sbatch INPUT [rema scan-positions options]}
shift
[[ -f $IN ]] || IN=$SCAN/targets/$IN
[[ -f $IN ]] || { echo "no input table $1 (nor in $SCAN/targets)"; exit 1; }
stem=$(basename "$IN"); stem=${stem%.*}
OUT=${OUT:-$SCAN/results/${stem}_rema.fits}
mkdir -p "$SCAN/logs" "$(dirname "$OUT")"

source "$SPS/miniforge3/etc/profile.d/conda.sh"
conda activate "$SPS/envs/rema"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$SCAN/jax_cache XDG_CACHE_HOME=$SPS/.cache

echo "$(date) $SLURM_JOB_ID on $(hostname): $IN -> $OUT"
rema scan-positions "$IN" --dr11 "$DR11" --calib "$CALIB" --work "$SCAN" --out "$OUT" "$@"
echo "$(date) done: $OUT"
