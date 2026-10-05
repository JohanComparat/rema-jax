#!/usr/bin/env bash
# Download the Legacy Survey DR11 south sweep catalogues and their photo-z
# files, resumable and checksum-verified. Both sets cover the same 1600 sky
# tiles: 11.0 is 1.80 TB (46 kB to 7 GB per file), 11.0-photo-z is 0.35 TB
# (12 kB to 1.4 GB per file), 2.15 TB in total.
#
# Usage: get_dr11_south_sweeps.sh [DEST_DIR] [NJOBS]
#   DEST_DIR  where to store the files      (default: $LEGACYSURVEY_DIR/dr11/south/sweep
#             when LEGACYSURVEY_DIR is set, else the current directory)
#   NJOBS     number of parallel downloads  (default: 4)
#
# LEGACYSURVEY_DIR is the Legacy Surveys root, the directory holding dr11/
# (CC-IN2P3: /sps/lsst/datasets/desi/legacysurveys).
#
# Each set goes into its own subdirectory, as on the server:
#   DEST_DIR/11.0/sweep-*.fits  and  DEST_DIR/11.0-photo-z/sweep-*-pz.fits
#
# Safe to interrupt (Ctrl-C, lost connection, killed job) and re-run. In each
# subdirectory:
#   - files whose sha256 matched are listed in .verified and skipped
#   - partial downloads live in .partial/ and are resumed (wget -c)
#   - a file is moved out of .partial/ only once its sha256 matches
#   - files already there (e.g. from a plain wget) are checked, and resumed
#     if incomplete
set -euo pipefail

BASE_URL=${BASE_URL:-https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/sweep}
SUBDIRS=(11.0 11.0-photo-z)
DEST=${1:-${LEGACYSURVEY_DIR:+$LEGACYSURVEY_DIR/dr11/south/sweep}}
DEST=${DEST:-.}
NJOBS=${2:-4}

fetch_one() {
    local sum=$1 f=$2
    # Left over from an earlier download: accept it if complete, else resume it.
    if [[ -f $f ]]; then
        if echo "$sum  $f" | sha256sum -c --status; then
            echo "$f" >> .verified
            echo "OK (already present) $f"
            return 0
        fi
        mv "$f" ".partial/$f"
    fi
    if ! wget -c -nv -P .partial --tries=20 --waitretry=30 --read-timeout=120 \
              --retry-connrefused "$DIR_URL/$f"; then
        echo "FAILED download $f" >&2
        return 1
    fi
    if echo "$sum  .partial/$f" | sha256sum -c --status; then
        mv ".partial/$f" "$f"
        echo "$f" >> .verified
        echo "OK $f"
    else
        rm -f ".partial/$f"
        echo "FAILED checksum $f (deleted, will be fetched again next run)" >&2
        return 1
    fi
}
export -f fetch_one

# Download $BASE_URL/<subdir> into ./<subdir>; fails if any file failed.
# Runs in a subshell so that cd and DIR_URL stay local to it.
get_subdir() (
    local sub=$1
    local sums=legacysurvey_dr11_south_sweep_$sub.sha256sum
    export DIR_URL=$BASE_URL/$sub
    mkdir -p "$sub/.partial" && cd "$sub" && touch .verified || return 1

    # The checksum file is also the list of files to get. Refresh it each run,
    # fall back to the cached copy if the server is unreachable.
    if wget -q -O "$sums.tmp" "$DIR_URL/$sums"; then
        mv "$sums.tmp" "$sums"
    else
        rm -f "$sums.tmp"
        [[ -s $sums ]] || { echo "Cannot fetch $DIR_URL/$sums" >&2; return 1; }
        echo "Server unreachable, using cached $sums" >&2
    fi

    # "sha256 filename" pairs not yet verified.
    local todo n_total n_todo
    todo=$(awk 'FILENAME == ARGV[1] { done[$1]; next }
                $2 ~ /\.fits$/ && !($2 in done)' .verified "$sums")
    n_total=$(awk '$2 ~ /\.fits$/' "$sums" | wc -l)
    n_todo=$(grep -c . <<< "$todo" || true)
    echo "$sub: $((n_total - n_todo)) / $n_total files already verified, $n_todo to go"

    if [[ $n_todo -gt 0 ]]; then
        xargs -n 2 -P "$NJOBS" bash -c 'fetch_one "$@"' _ <<< "$todo" || return 1
    fi
    echo "$sub: all $n_total files downloaded and verified."
)

mkdir -p "$DEST"
cd "$DEST"
failed=()
for sub in "${SUBDIRS[@]}"; do
    get_subdir "$sub" || failed+=("$sub")
done
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Some files failed in: ${failed[*]}; re-run the script to retry them." >&2
    exit 1
fi
