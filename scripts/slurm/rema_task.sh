#!/usr/bin/env bash
# One stage of the full DR11 blind run, for one array index. The driver (rema_dr11_blind.sh)
# submits these; they also run without SLURM: rema_task.sh STAGE [INDEX].
#
#   ingest  INDEX = chunk of CHUNK sweeps    -> $OUTDIR/galaxies/<sweep>.fits (one per sweep)
#   randoms INDEX = randoms file number      -> $OUTDIR/randoms_index/randoms-south-1-<k>.*
#   calib                                    -> $OUTDIR/calib/{footprint,calib}.fits, plots/
#   plan                                     -> $PLAN
#   region  INDEX = region of the plan       -> $OUTDIR/regions/<NNNN>/{footprint,clusters}.fits
#   batch   INDEX = line of $OUTDIR/jobs/batches (region ids), run one after the other
#                                               (its checkpoint/ is deleted once clusters.fits is written)
#   clean                                    deletes the intermediate products of a merged run
#   merge                                    -> $CLUSTERS_DIR/clusters_dr11{,_regions}.fits, _qa.json
#                                               and $MEMBERS_DIR/clusters_dr11_members.fits
#
# Environment: DR11 (the dr11/south directory; default $LEGACYSURVEY_DIR/dr11/south), OUTDIR, CONFIG (YAML for ingest, randoms, calib,
# plan and the region footprints; the region runs take the calibration's), CALIB (the calibration
# of the region runs), CALIB_BOX ("RA0 RA1 DEC0 DEC1[;RA0 RA1 DEC0 DEC1...]"), PLAN, NRAND,
# CHUNK, TARGET_AREA, BUFFER, BAND_HEIGHT, MAX_GAL, MAX_PAIRS, AREA_BOX ("RA0 RA1 DEC0 DEC1[;...]":
# run on this area only, e.g. a test on a few sweeps), ALLOW_MISSING=1 (merge), ALLOW_UNCOVERED=1
# (plan: accept tiles with randoms whose sweep is listed but not ingested), CLUSTERS_DIR and
# MEMBERS_DIR (merge: clusters [$OUTDIR] and members [$CLUSTERS_DIR]), DEVICE (gpu|cpu),
# JAX_COMPILATION_CACHE_DIR. Outputs are written atomically and existing ones are kept.
set -euo pipefail

STAGE=$1
INDEX=${2:-${SLURM_ARRAY_TASK_ID:-0}}
DR11=${DR11:-${LEGACYSURVEY_DIR:+$LEGACYSURVEY_DIR/dr11/south}}
: "${DR11:?set DR11 to the dr11/south directory, or LEGACYSURVEY_DIR}" "${OUTDIR:?set OUTDIR}"
PLAN=${PLAN:-$OUTDIR/regions.fits}
NRAND=${NRAND:-20}
CHUNK=${CHUNK:-20}
DEVICE=${DEVICE:-gpu}
[[ $STAGE == region ]] || DEVICE=cpu               # only the region runs use the GPU
CFG=()
[[ -n ${CONFIG:-} ]] && CFG=(--config "$CONFIG")
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$OUTDIR/jax_cache/$DEVICE}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ $DEVICE == gpu ]]; then export JAX_PLATFORMS=cuda; else export JAX_PLATFORMS=cpu; fi
mkdir -p "$OUTDIR" "$JAX_COMPILATION_CACHE_DIR"

# "RA0 RA1 DEC0 DEC1;..." -> --box RA0 RA1 DEC0 DEC1 --box ... (one word per line)
box_opts() {
    local b parts=()
    [[ -n $1 ]] || return 0
    IFS=';' read -ra parts <<< "$1"
    for b in "${parts[@]}"; do
        # shellcheck disable=SC2086
        printf '%s\n' --box $b
    done
}
mapfile -t area < <(box_opts "${AREA_BOX:-}")

# One region: footprint, blind run, then the checkpoint is deleted. Returns non-zero on failure
# (the checkpoint is then kept, so a rerun resumes).
run_region() {
    local id=$1 d
    d=$OUTDIR/regions/$(printf %04d "$id")
    mkdir -p "$d"
    if [[ ! -s $d/footprint.fits ]]; then
        rema maps --index "$OUTDIR/randoms_index" --nrand "$NRAND" --regions "$PLAN" --region-id "$id" \
            --out "$d/footprint.fits" "${CFG[@]}" || return 1
    fi
    rema blind --galaxies "$OUTDIR/galaxies" --regions "$PLAN" --region-id "$id" --calib "$CALIB" \
        --footprint "$d/footprint.fits" --checkpoint "$d/checkpoint" --specpost --out "$d/clusters.fits" \
        || return 1
    # clusters.fits is written: the checkpoint (resume data, the bulk of the region) is not needed.
    rm -rf "$d/checkpoint"
}

# JAX writes cache entries in place, without locking: once the priming region has filled the
# shared cache, each task compiles into a private copy in its job's TMPDIR, which concurrent
# tasks never see.
private_cache() {
    shared=$JAX_COMPILATION_CACHE_DIR
    if [[ -e $shared/.primed ]]; then
        private=$(mktemp -d "${TMPDIR:-/tmp}/rema_jax.XXXXXX")
        trap 'rm -rf "$private"' EXIT
        cp -a "$shared/." "$private/"
        export JAX_COMPILATION_CACHE_DIR=$private
    fi
}

case $STAGE in
  ingest)
    mapfile -t sweeps < <(find "$DR11/sweep/11.0" -maxdepth 1 -name 'sweep-*.fits' ! -name '*-pz.fits' | LC_ALL=C sort)
    chunk=("${sweeps[@]:$((INDEX * CHUNK)):$CHUNK}")
    (( ${#chunk[@]} )) || { echo "chunk $INDEX is empty"; exit 0; }
    rema ingest "${chunk[@]}" --outdir "$OUTDIR/galaxies" --require-pz "${area[@]}" "${CFG[@]}" ;;
  randoms)
    rema randoms-index "$DR11/randoms/randoms-south-1-$INDEX.fits" --outdir "$OUTDIR/randoms_index" \
        "${area[@]}" "${CFG[@]}" ;;
  calib)
    mapfile -t box < <(box_opts "${CALIB_BOX:?set CALIB_BOX (see rema regions --calib-suggest)}")
    d=$OUTDIR/calib
    mkdir -p "$d"
    [[ -s $d/footprint.fits ]] || rema maps --index "$OUTDIR/randoms_index" --nrand "$NRAND" "${box[@]}" \
        --out "$d/footprint.fits" "${CFG[@]}"
    [[ -s $d/calib.fits ]] || rema calibrate --galaxies "$OUTDIR/galaxies" "${box[@]}" \
        --footprint "$d/footprint.fits" --out "$d/calib.fits" --plots "$d/plots" "${CFG[@]}" ;;
  plan)
    opts=(--target-area "${TARGET_AREA:-100}" --buffer "${BUFFER:-2}" --band-height "${BAND_HEIGHT:-10}")
    [[ -n ${MAX_GAL:-} ]] && opts+=(--max-gal "$MAX_GAL")
    [[ -n ${MAX_PAIRS:-} ]] && opts+=(--max-pairs "$MAX_PAIRS")
    [[ ${ALLOW_UNCOVERED:-0} == 1 ]] && opts+=(--allow-uncovered)
    opts+=("${area[@]}")
    rema regions --galaxies "$OUTDIR/galaxies" --index "$OUTDIR/randoms_index" --sweeps "$DR11/sweep/11.0" \
        --out "$PLAN" "${opts[@]}" "${CFG[@]}" ;;
  region)
    : "${CALIB:?set CALIB to the calibration of the run}"
    private_cache
    run_region "$INDEX"
    touch "$shared/.primed" ;;
  batch)
    : "${CALIB:?set CALIB to the calibration of the run}"
    ids=$(sed -n "$((INDEX + 1))p" "$OUTDIR/jobs/batches")
    [[ -n $ids ]] || { echo "batch $INDEX is empty"; exit 0; }
    private_cache
    failed=()
    for id in $ids; do
        echo "region $id"
        run_region "$id" || failed+=("$id")
    done
    (( ${#failed[@]} == 0 )) || { echo "regions failed: ${failed[*]}" >&2; exit 1; } ;;
  merge)
    : "${CALIB:?set CALIB to the calibration of the run}"
    opts=()
    [[ ${ALLOW_MISSING:-0} == 1 ]] && opts+=(--allow-missing)
    cdir=${CLUSTERS_DIR:-$OUTDIR}
    mdir=${MEMBERS_DIR:-$cdir}
    mkdir -p "$cdir" "$mdir"
    rema merge --plan "$PLAN" --runs "$OUTDIR/regions" --calib "$CALIB" \
        --out "$cdir/clusters_dr11.fits" --members-out "$mdir/clusters_dr11_members.fits" "${opts[@]}" ;;
  clean)
    # Only after a complete merge: the merged products must exist and hold every region.
    cdir=${CLUSTERS_DIR:-$OUTDIR}
    mdir=${MEMBERS_DIR:-$cdir}
    [[ -s $cdir/clusters_dr11.fits && -s $mdir/clusters_dr11_members.fits ]] || \
        { echo "no merged products in $cdir and $mdir: run the merge first" >&2; exit 1; }
    st=$(rema status --plan "$PLAN" --runs "$OUTDIR/regions" ${CALIB:+--calib "$CALIB"})
    [[ -z $(sed -n 's/^todo_array=//p' <<< "$st") ]] || \
        { echo "regions still missing or stale: not cleaning" >&2; echo "$st" | head -1 >&2; exit 1; }
    [[ $cdir/clusters_dr11.fits -nt $OUTDIR/regions ]] || \
        { echo "regions changed after the merge: merge again before cleaning" >&2; exit 1; }
    du -sh "$OUTDIR"/{galaxies,randoms_index,jax_cache} 2>/dev/null || true
    rm -rf "$OUTDIR/galaxies" "$OUTDIR/randoms_index" "$OUTDIR/jax_cache" "$OUTDIR"/regions/*/checkpoint
    echo "kept: the plan, calib/, logs/, regions/*/{footprint,clusters}.fits and the merged products" ;;
  *) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac
