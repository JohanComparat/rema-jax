#!/usr/bin/env bash
# Download the Legacy Survey DR11 south random catalogues, resumable and
# checksum-verified. The server has 20 files, randoms-south-1-0.fits to
# randoms-south-1-19.fits, of 23 GB each (66 million randoms in 348-byte rows),
# 460 GB in total. The rows of every file are in random sky order, so a sky
# region cannot be fetched on its own; each file samples the whole footprint
# (2,500 randoms per deg², about 8 per HEALPix nside-1024 pixel), and one file
# is enough for footprint and depth maps.
#
# Usage: get_dr11_south_randoms.sh [DEST_DIR] [FILES] [NJOBS]
#   DEST_DIR  where to store the files  (default: $LEGACYSURVEY_DIR/dr11/south/randoms
#             when LEGACYSURVEY_DIR is set, else the current directory)
#   FILES     file indices: "0", "0-3", "0,5,9" or "all"   (default: 0)
#   NJOBS     number of parallel downloads                 (default: 1)
#
# LEGACYSURVEY_DIR is the Legacy Surveys root, the directory holding dr11/
# (CC-IN2P3: /sps/lsst/datasets/desi/legacysurveys).
#
# Safe to interrupt (Ctrl-C, lost connection, killed job) and re-run:
#   - files whose sha256 matched are listed in .verified and skipped
#   - partial downloads live in .partial/ and are resumed (wget -c)
#   - a file is moved out of .partial/ only once its sha256 matches
#   - files already there (e.g. from a plain wget) are checked, and resumed
#     if incomplete
# To continue a download started with another tool (e.g. a browser), stop that
# tool first and move its partial file to DEST_DIR/.partial/<file name>.
set -euo pipefail

BASE_URL=${BASE_URL:-https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/randoms}
SUMS=legacysurvey_dr11_south_randoms.sha256sum

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
              --retry-connrefused "$BASE_URL/$f"; then
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
export BASE_URL

# File names for a selection "0", "0-3", "0,5,9" or "all", one per line.
selected_files() {
    local spec=$1 part
    if [[ $spec == all ]]; then
        awk '$2 ~ /\.fits$/ {print $2}' "$SUMS"
        return
    fi
    IFS=, read -ra parts <<< "$spec"
    for part in "${parts[@]}"; do
        if [[ $part =~ ^([0-9]+)-([0-9]+)$ ]]; then
            seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
        elif [[ $part =~ ^[0-9]+$ ]]; then
            echo "$part"
        else
            echo "Bad file selection '$spec' (use e.g. 0, 0-3, 0,5,9 or all)" >&2
            return 1
        fi
    done | sed 's/.*/randoms-south-1-&.fits/'
}

main() {
    local dest=${1:-${LEGACYSURVEY_DIR:+$LEGACYSURVEY_DIR/dr11/south/randoms}}
    local files=${2:-0} njobs=${3:-1}
    dest=${dest:-.}
    mkdir -p "$dest/.partial"
    cd "$dest"
    touch .verified

    # The checksum file is also the list of files on the server. Refresh it each
    # run, fall back to the cached copy if the server is unreachable.
    if wget -q -O "$SUMS.tmp" "$BASE_URL/$SUMS"; then
        mv "$SUMS.tmp" "$SUMS"
    else
        rm -f "$SUMS.tmp"
        [[ -s $SUMS ]] || { echo "Cannot fetch $BASE_URL/$SUMS" >&2; return 1; }
        echo "Server unreachable, using cached $SUMS" >&2
    fi

    local wanted missing todo n_total n_todo
    wanted=$(selected_files "$files")
    missing=$(awk 'FILENAME == ARGV[1] { on_server[$2]; next } !($1 in on_server)' \
                  "$SUMS" <(printf '%s\n' "$wanted"))
    if [[ -n $missing ]]; then
        echo "Not on the server: $(tr '\n' ' ' <<< "$missing")" >&2
        return 1
    fi
    # "sha256 filename" pairs selected and not yet verified.
    todo=$(awk 'FILENAME == ARGV[1] { done[$1]; next }
                FILENAME == ARGV[2] { want[$1]; next }
                ($2 in want) && !($2 in done)' .verified <(printf '%s\n' "$wanted") "$SUMS")
    n_total=$(grep -c . <<< "$wanted" || true)
    n_todo=$(grep -c . <<< "$todo" || true)
    echo "randoms: $((n_total - n_todo)) / $n_total selected files already verified, $n_todo to go"
    [[ $n_todo -gt 0 ]] || return 0

    # Warn when the files to fetch (less what is already on disk) exceed the free space.
    local need=0 f p size free
    for f in $(awk '{print $2}' <<< "$todo"); do
        size=$(wget -q --spider -S "$BASE_URL/$f" 2>&1 \
               | awk 'tolower($1) == "content-length:" {print $2}' | tail -1) || size=0
        need=$((need + ${size:-0}))
        for p in "$f" ".partial/$f"; do
            if [[ -f $p ]]; then need=$((need - $(stat -c %s "$p"))); fi
        done
    done
    free=$(df -PB1 . | awk 'NR == 2 {print $4}')
    if [[ $need -gt $free ]]; then
        echo "WARNING: $((need / 1000000000)) GB to download but only $((free / 1000000000)) GB free here" >&2
    fi

    xargs -n 2 -P "$njobs" bash -c 'fetch_one "$@"' _ <<< "$todo" || {
        echo "Some files failed; re-run the script to retry them." >&2
        return 1
    }
    echo "randoms: all $n_total selected files downloaded and verified."
}

# Run unless sourced (the functions can be sourced for testing).
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    main "$@"
fi
