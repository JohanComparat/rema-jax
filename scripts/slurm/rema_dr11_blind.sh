#!/usr/bin/env bash
# Driver of the full DR11 south blind run on a SLURM cluster, in two phases with a check of the
# calibration in between:
#
#   rema_dr11_blind.sh prepare   ingest every sweep once (one table per sweep), index every
#                                randoms file once, and, when CALIB_BOX is set, calibrate
#   (check $OUTDIR/calib/plots and the calibration header, then set CALIB explicitly)
#   rema_dr11_blind.sh run       plan the regions, run them (one array task per region, the
#                                first one alone to fill the JAX cache), then merge; while the
#                                calibration job of 'prepare' is still queued or running (and
#                                CALIB is its $OUTDIR/calib/calib.fits), the regions wait for it
#   rema_dr11_blind.sh status    progress of the run
#   rema_dr11_blind.sh clean     once the merge is complete, delete the intermediate products
#                                (per-sweep galaxy tables, randoms index, JAX cache, checkpoints)
#
# Both phases submit only what is missing (or stale: made with another plan or calibration), so
# after failures or timeouts simply call them again. They refuse to submit while jobs they
# submitted before are still queued or running.
#
# Environment (defaults in brackets):
#   DR11 [$LEGACYSURVEY_DIR/dr11/south]  dr11/south directory (sweep/11.0, sweep/11.0-photo-z,
#                         randoms/); one of DR11 or LEGACYSURVEY_DIR is required
#   LEGACYSURVEY_DIR      Legacy Surveys root holding dr11/ (CC-IN2P3: /sps/lsst/datasets/desi/legacysurveys)
#   OUTDIR [required]     run directory; one per calibration and rema version
#   CONFIG [scripts/slurm/dr11_south.yaml]  configuration (E(B-V) < 0.2 cut)
#   CALIB                 calibration of the region runs (phase 2; e.g. $OUTDIR/calib/calib.fits)
#   CLUSTERS_DIR [$OUTDIR]  MEMBERS_DIR [$CLUSTERS_DIR]   where the merge writes the clusters
#                         (with the _regions and _qa files) and the members
#   CALIB_BOX             calibration area(s) "RA0 RA1 DEC0 DEC1[;...]"; without it, phase 1 prints
#                         suggestions once the galaxies are ingested
#   NRAND [20]  CHUNK [20]  TARGET_AREA [100]  BUFFER [2]  MAX_GAL  MAX_PAIRS
#   AREA_BOX              "RA0 RA1 DEC0 DEC1[;...]": run on this area only (a test, or the part of
#                         the sky already downloaded); every stage and the bookkeeping use it
#   DEVICE [gpu]          gpu or cpu, for the region tasks (ingest, randoms, calib, merge: CPU)
#   ACCOUNT  PART_GPU  PART_CPU  GPU_GRES [gpu:1]  GPU_CONSTRAINT
#   GPUS                  GPUs per region task as --gpus=N instead of --gres=GPU_GRES (sites
#                         that take only --gpus, e.g. CC-IN2P3)
#   REGION_CPUS / REGION_MEM / REGION_TIME   [gpu: 8, 96G, 8:00:00; cpu: 16, 64G, 24:00:00]
#   GPAR [20]  RPAR [20]  BPAR [50]          concurrent tasks of the ingest/randoms/region arrays
#   MAX_ARRAY             most tasks in the region array (a site's limit of queued jobs, e.g. 100
#                         GPU jobs per user at CC-IN2P3); above it each task runs several regions in
#                         turn ($OUTDIR/jobs/batches), with REGION_TIME (H:MM:SS) times that number
#   ARRAY                 override the region array (e.g. ARRAY=17,42 to rerun two regions)
#   EXTRA_SBATCH          extra sbatch options for every job (e.g. --licenses=sps at CC-IN2P3)
#
# Site settings: source scripts/slurm/ccin2p3.env first on the CC-IN2P3 cluster.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:?usage: rema_dr11_blind.sh prepare|run|status|clean}
DR11=${DR11:-${LEGACYSURVEY_DIR:+$LEGACYSURVEY_DIR/dr11/south}}
: "${DR11:?set DR11 or LEGACYSURVEY_DIR}" "${OUTDIR:?set OUTDIR}"
export DR11 OUTDIR
export CONFIG=${CONFIG:-$HERE/dr11_south.yaml}
export PLAN=${PLAN:-$OUTDIR/regions.fits}
export NRAND=${NRAND:-20} CHUNK=${CHUNK:-20} TARGET_AREA=${TARGET_AREA:-100} BUFFER=${BUFFER:-2}
export DEVICE=${DEVICE:-gpu}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$OUTDIR/jax_cache/$DEVICE}
GPAR=${GPAR:-20} RPAR=${RPAR:-20} BPAR=${BPAR:-50}
TASK=$HERE/rema_task.sh
JOBS=$OUTDIR/jobs
mkdir -p "$OUTDIR/logs" "$JOBS"

common=(--parsable --export=ALL)
[[ -n ${ACCOUNT:-} ]] && common+=(--account="$ACCOUNT")
# shellcheck disable=SC2206
[[ -n ${EXTRA_SBATCH:-} ]] && common+=($EXTRA_SBATCH)
cpu=("${common[@]}")
[[ -n ${PART_CPU:-} ]] && cpu+=(--partition="$PART_CPU")
if [[ $DEVICE == gpu ]]; then
    if [[ -n ${GPUS:-} ]]; then gpu=(--gpus="$GPUS"); else gpu=(--gres="${GPU_GRES:-gpu:1}"); fi
    region=("${common[@]}" "${gpu[@]}" --cpus-per-task="${REGION_CPUS:-8}"
            --mem="${REGION_MEM:-96G}" --time="${REGION_TIME:-8:00:00}")
    [[ -n ${PART_GPU:-} ]] && region+=(--partition="$PART_GPU")
    [[ -n ${GPU_CONSTRAINT:-} ]] && region+=(--constraint="$GPU_CONSTRAINT")
else
    region=("${cpu[@]}" --cpus-per-task="${REGION_CPUS:-16}" --mem="${REGION_MEM:-64G}"
            --time="${REGION_TIME:-24:00:00}")
fi

kv() { sed -n "s/^$1=//p"; }                       # value of key=value lines on stdin

# SLURM array specification -> ids, one per line ("0-2,5%50" -> 0 1 2 5).
expand() {
    local part
    for part in ${1//,/ }; do
        part=${part%%%*}
        if [[ $part == *-* ]]; then seq "${part%-*}" "${part#*-}"; else echo "$part"; fi
    done
}

# Job id of the calibration submitted by 'prepare', while it is queued or running.
calib_job() {
    [[ -s $JOBS/prepare ]] || return 0
    squeue -h -j "$(paste -sd, "$JOBS/prepare")" -n rema-calib -o %i 2>/dev/null | head -1
}

# AREA_BOX "RA0 RA1 DEC0 DEC1;..." -> --box RA0 RA1 DEC0 DEC1 --box ... (the task script does the same)
area=()
if [[ -n ${AREA_BOX:-} ]]; then
    IFS=';' read -ra _boxes <<< "$AREA_BOX"
    # shellcheck disable=SC2206
    for b in "${_boxes[@]}"; do area+=(--box $b); done
fi

# Refuse to submit while jobs recorded in $JOBS/$1 are still queued or running.
check_idle() {
    local f=$JOBS/$1 ids
    [[ -s $f ]] || return 0
    ids=$(paste -sd, "$f")
    if [[ -n $(squeue -h -j "$ids" 2>/dev/null) ]]; then
        echo "jobs $ids of a previous '$1' are still queued or running; wait or scancel them" >&2
        exit 1
    fi
}

submit() {                                         # submit RECORD sbatch-args... ; prints the job id
    local rec=$1; shift
    local id
    id=$(sbatch "$@") || { echo "sbatch failed: $*" >&2; return 1; }
    id=${id%%;*}
    echo "$id" >> "$JOBS/$rec"
    echo "$id"
}

case $MODE in
  prepare)
    check_idle prepare
    : > "$JOBS/prepare"
    todo=$(rema todo --sweeps "$DR11/sweep/11.0" --galaxies "$OUTDIR/galaxies" \
                     --index "$OUTDIR/randoms_index" --chunk "$CHUNK" --nrand "$NRAND" "${area[@]}")
    garr=$(kv ingest_array <<< "$todo"); rarr=$(kv randoms_array <<< "$todo")
    deps=()
    if [[ -n $garr ]]; then
        g=$(submit prepare "${cpu[@]}" --job-name=rema-ingest --array="$garr%$GPAR" --cpus-per-task=4 \
                --mem=16G --time=6:00:00 --output="$OUTDIR/logs/ingest_%a.log" "$TASK" ingest)
        deps+=("$g"); echo "ingest: job $g, chunks $garr"
    fi
    if [[ -n $rarr ]]; then
        r=$(submit prepare "${cpu[@]}" --job-name=rema-randoms --array="$rarr%$RPAR" --cpus-per-task=2 \
                --mem=24G --time=4:00:00 --output="$OUTDIR/logs/randoms_%a.log" "$TASK" randoms)
        deps+=("$r"); echo "randoms index: job $r, files $rarr"
    fi
    if [[ -n ${CALIB_BOX:-} && ! -s $OUTDIR/calib/calib.fits ]]; then
        dep=(); (( ${#deps[@]} )) && dep=(--dependency="afterok:$(IFS=:; echo "${deps[*]}")")
        c=$(submit prepare "${cpu[@]}" "${dep[@]}" --job-name=rema-calib --cpus-per-task=32 --mem=128G \
                --time=24:00:00 --output="$OUTDIR/logs/calib.log" "$TASK" calib)
        echo "calibration: job $c, area $CALIB_BOX"
    elif [[ -z ${CALIB_BOX:-} ]]; then
        echo "no CALIB_BOX: once the galaxies are ingested, list candidate areas with"
        echo "  rema regions --galaxies $OUTDIR/galaxies --calib-suggest 400 --config $CONFIG"
        echo "then rerun 'prepare' with CALIB_BOX=\"RA0 RA1 DEC0 DEC1[;...]\""
    fi ;;
  run)
    : "${CALIB:?set CALIB to the checked calibration (e.g. $OUTDIR/calib/calib.fits)}"
    calwait=()
    if [[ ! -s $CALIB ]]; then
        cj=$(calib_job)
        [[ -n $cj && $CALIB == "$OUTDIR/calib/calib.fits" ]] || { echo "no calibration at $CALIB" >&2; exit 1; }
        calwait=(--dependency="afterok:$cj")
        echo "calibration job $cj is not finished: the regions start once it succeeds (unchecked calibration)"
    fi
    check_idle run
    for d in "${CLUSTERS_DIR:-$OUTDIR}" "${MEMBERS_DIR:-${CLUSTERS_DIR:-$OUTDIR}}"; do
        mkdir -p "$d" 2>/dev/null && [[ -w $d ]] || { echo "cannot write the merged products to $d" >&2; exit 1; }
    done
    todo=$(rema todo --sweeps "$DR11/sweep/11.0" --galaxies "$OUTDIR/galaxies" \
                     --index "$OUTDIR/randoms_index" --chunk "$CHUNK" --nrand "$NRAND" "${area[@]}")
    if [[ -n $(kv ingest_array <<< "$todo") || -n $(kv randoms_array <<< "$todo") ]]; then
        echo "ingest or randoms index incomplete: run 'prepare' first" >&2; exit 1
    fi
    [[ -s $PLAN ]] || "$TASK" plan
    calopt=(); [[ -s $CALIB ]] && calopt=(--calib "$CALIB")
    st=$(rema status --plan "$PLAN" --runs "$OUTDIR/regions" "${calopt[@]}")
    echo "$st" | head -1
    if [[ -n ${ARRAY:-} ]]; then arr=$ARRAY; prime=""; else arr=$(kv todo_array <<< "$st"); prime=$(kv prime <<< "$st"); fi
    : > "$JOBS/run"
    deps=()
    if [[ -n $arr ]]; then
        if [[ -n $prime && ! -e $JAX_COMPILATION_CACHE_DIR/.primed ]]; then
            p=$(submit run "${region[@]}" "${calwait[@]}" --job-name=rema-prime --array="$prime" \
                    --output="$OUTDIR/logs/region_%a.log" "$TASK" region)
            deps+=("$p"); echo "priming region $prime: job $p"
            arr=$(kv todo_array_noprime <<< "$st")
        fi
        if [[ -n $arr ]]; then
            dep=("${calwait[@]}"); (( ${#deps[@]} )) && dep=(--dependency="afterany:${deps[0]}")
            mapfile -t ids < <(expand "$arr")
            if (( ${MAX_ARRAY:-0} > 0 && ${#ids[@]} > MAX_ARRAY )); then
                # Too many tasks for the site: each task runs up to k regions in turn, k times as
                # long. Regions are numbered by decreasing cost: deal them out in turn, so that
                # every task gets a mix of expensive and cheap ones.
                k=$(( (${#ids[@]} + MAX_ARRAY - 1) / MAX_ARRAY ))
                nb=$(( (${#ids[@]} + k - 1) / k ))
                : > "$JOBS/batches"
                for ((j = 0; j < nb; j++)); do
                    line=(); for ((i = j; i < ${#ids[@]}; i += nb)); do line+=("${ids[i]}"); done
                    echo "${line[*]}" >> "$JOBS/batches"
                done
                rt=${REGION_TIME:-$([[ $DEVICE == gpu ]] && echo 8:00:00 || echo 24:00:00)}
                IFS=: read -r hh mm ss <<< "$rt"
                t=$(( (10#$hh * 3600 + 10#$mm * 60 + 10#$ss) * k ))
                tt="$((t / 3600)):$(printf %02d:%02d $((t % 3600 / 60)) $((t % 60)))"
                b=$(submit run "${region[@]/#--time=*/--time=$tt}" "${dep[@]}" \
                        --job-name=rema-region --array="0-$((nb - 1))%$BPAR" \
                        --output="$OUTDIR/logs/batch_%a.log" "$TASK" batch)
                deps+=("$b"); echo "regions: job $b, $nb tasks of up to $k regions (${#ids[@]} regions, $DEVICE)"
            else
                b=$(submit run "${region[@]}" "${dep[@]}" --job-name=rema-region --array="$arr%$BPAR" \
                        --output="$OUTDIR/logs/region_%a.log" "$TASK" region)
                deps+=("$b"); echo "regions: job $b, array $arr ($DEVICE)"
            fi
        fi
    fi
    dep=("${calwait[@]}"); (( ${#deps[@]} )) && dep=(--dependency="afterany:$(IFS=:; echo "${deps[*]}")")
    m=$(submit run "${cpu[@]}" "${dep[@]}" --job-name=rema-merge --cpus-per-task=8 --mem=128G \
            --time=4:00:00 --output="$OUTDIR/logs/merge.log" "$TASK" merge)
    echo "merge: job $m" ;;
  status)
    if [[ -s $PLAN ]]; then
        calopt=(); [[ -n ${CALIB:-} && -s $CALIB ]] && calopt=(--calib "$CALIB")
        rema status --plan "$PLAN" --runs "$OUTDIR/regions" "${calopt[@]}" | head -1
    else                                           # phase 1: the plan is made by the first 'run'
        todo=$(rema todo --sweeps "$DR11/sweep/11.0" --galaxies "$OUTDIR/galaxies" \
                         --index "$OUTDIR/randoms_index" --chunk "$CHUNK" --nrand "$NRAND" "${area[@]}")
        cal="none (rerun 'prepare' with CALIB_BOX)"
        cj=$(calib_job)
        [[ -n $cj ]] && cal="job $cj queued or running"
        [[ -s $OUTDIR/calib/calib.fits ]] && cal=$OUTDIR/calib/calib.fits
        echo "prepare: galaxy tables $(kv ingest_done <<< "$todo") sweeps," \
             "randoms indexes $(kv randoms_done <<< "$todo"); calibration: $cal"
        echo "no region plan yet ($PLAN): 'run' makes it"
    fi
    for f in prepare run; do
        [[ -s $JOBS/$f ]] && squeue -h -j "$(paste -sd, "$JOBS/$f")" -o "%i %j %T %M" 2>/dev/null || true
    done ;;
  clean)
    check_idle run
    "$TASK" clean ;;
  *) echo "usage: rema_dr11_blind.sh prepare|run|status|clean" >&2; exit 1 ;;
esac
