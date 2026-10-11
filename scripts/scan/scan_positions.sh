#!/usr/bin/env bash
# Scan mode + spectroscopic post-processing for a table of positions, on the DR11 south data at
# CC-IN2P3, as one SLURM job (rema scan-positions; docs/dr11.rst "Scan a list of positions").
#
#   rema-env                                          # the rema Python environment
#   $HOME/software/rema-jax/scripts/scan/scan_positions.sh INPUT OUTPUT [rema scan-positions options]
#
# INPUT: a table with RA and DEC (FITS, CSV, ECSV, ...; other names with --ra-col/--dec-col).
# OUTPUT: a FITS file with the input columns as INPUT_<NAME>, the scan and spectroscopic columns,
# SCANNED, and the members (MEMBERS HDU, keyed by the input row in MEM_MATCH_ID).
#
# Environment (defaults in brackets): WORK [/sps/lsst/users/$USER/rema_scan: galaxy tables of the
# sweeps, kept for later runs, and randoms indexes], CPU=1 (an htc job instead of a V100 GPU),
# TIME [12:00:00], MEM [45G GPU, 32G CPU], NOSUBMIT=1 (run here, without SLURM), DRY_RUN=1 (print
# the sbatch command only). Settings of scripts/slurm/ccin2p3.env (DR11, partitions, EXTRA_SBATCH).
set -euo pipefail

INPUT=$(realpath "${1:?input table}")
OUTPUT=$(realpath -m "${2:?output FITS}")
shift 2
REMA=${REMA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
# shellcheck disable=SC1091
source "$REMA/scripts/slurm/ccin2p3.env"
WORK=${WORK:-/sps/lsst/users/$USER/rema_scan}
mkdir -p "$WORK/logs" "$(dirname "$OUTPUT")"
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$WORK/jax_cache}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cmd=(rema scan-positions "$INPUT" --out "$OUTPUT" --dr11 "$DR11" --work "$WORK" "$@")
if [[ ${CPU:-0} == 1 ]]; then
    res=(--partition "$PART_CPU" --cpus-per-task 8 --mem "${MEM:-32G}")
    plat=cpu
else
    res=(--partition "$PART_GPU" --gpus "$GPUS" --cpus-per-task "$REGION_CPUS" --mem "${MEM:-45G}")
    plat=cuda
fi

if [[ ${NOSUBMIT:-0} == 1 ]]; then
    JAX_PLATFORMS=$plat "${cmd[@]}"
    exit
fi
name=scan-$(basename "$INPUT" | sed 's/\.[^.]*$//')
sb=(sbatch --parsable --job-name "$name" --output "$WORK/logs/$name-%j.log" --time "${TIME:-12:00:00}"
    ${EXTRA_SBATCH:-} "${res[@]}" --wrap "$(printf '%q ' env JAX_PLATFORMS=$plat "${cmd[@]}")")
if [[ ${DRY_RUN:-0} == 1 ]]; then
    echo "${sb[*]}"
    exit
fi
job=$("${sb[@]}")
echo "job $job: $name -> $OUTPUT"
echo "log: $WORK/logs/$name-$job.log"
