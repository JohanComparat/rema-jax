#!/usr/bin/env bash
# Photo-z cluster finding on DR11 regions at CC-IN2P3 (docs/photoz_finders.rst): the photo-z
# filter of rema blind and PSCD (rema pscd), with their null tests, on the regions of the
# production plan; the red-sequence baseline is the cosmology study's fiducial rerun.
#
#   source $REMA/scripts/slurm/ccin2p3.env           # and activate the Python environment
#   $REMA/scripts/photoz/photoz_test.sh STAGE
#
# Stages (each submits jobs and returns; run them in this order, each once the previous jobs are
# done; `status` tells what is there):
#   prepare  galaxy tables (with the photo-z) of the regions' data boxes; their footprints, copied
#            from the cosmology study when there, else built from a randoms index
#   run      per region and variant (VARIANTS): rema blind or rema pscd, GPU or CPU jobs; and
#            the photo-z width fit of every region (errscale_<region>.txt)
#   strip    the strip variants STRIP_VARIANTS [rs_bcg null_rs null_pz1], one GPU job each, with
#            scripts/photoz/photoz_strip.sh on the inputs in STRIP_INPUTS (the laptop's strip
#            galaxy table galaxies_3sweeps_v2.fits and footprint_strip.fits, copied there) into
#            $OUTDIR/strip
#   status   what is done
#
# Variants (as scripts/photoz/photoz_strip.sh): pz_v0 pz_v1 pscd_0 pscd_1 null_pz null_pz1
# null_pscd null_pscd1 null_rs rs_bcg. rs_wcen is copied from the cosmology study (tier B,
# fiducial), which ran the production code, calibration and configuration.
#
# Environment (defaults in brackets): PRODUCTS [the DR11 rema products on the data system], RUN
# [$PRODUCTS/rema_dr11_v0.2.0_ra0-240: the plan and the calibration], OUTDIR
# [/sps/lsst/users/$USER/rema_photoz], COSMO_SENS [/sps/lsst/users/$USER/rema_cosmo_sens],
# REGIONS [9 82], VARIANTS [all of the above], PSCD_SNR [4], DRY_RUN=1 (print the sbatch commands
# only). Partitions, GPUs, memory, NRAND, DR11 and EXTRA_SBATCH as for
# scripts/slurm/rema_dr11_blind.sh.
set -euo pipefail

STAGE=${1:?stage: prepare, run, strip or status}
REMA=${REMA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PRODUCTS=${PRODUCTS:-${DR11:?set DR11}/rema}
RUN=${RUN:-$PRODUCTS/rema_dr11_v0.2.0_ra0-240}
PLAN=${PLAN:-$RUN/regions.fits}
CALIB=${CALIB:-$RUN/calib/calib.fits}
OUTDIR=${OUTDIR:-/sps/lsst/users/$USER/rema_photoz}
COSMO_SENS=${COSMO_SENS:-/sps/lsst/users/$USER/rema_cosmo_sens}
CONFIG=${CONFIG:-$REMA/scripts/slurm/dr11_south.yaml}
REGIONS=${REGIONS:-9 82}
VARIANTS=${VARIANTS:-pz_v0 pz_v1 pscd_0 pscd_1 null_pz null_pz1 null_pscd null_pscd1 null_rs rs_bcg}
PSCD_SNR=${PSCD_SNR:-4}
PART_CPU=${PART_CPU:-htc}
PART_GPU=${PART_GPU:-gpu_v100}
GPUS=${GPUS:-1}
REGION_CPUS=${REGION_CPUS:-5}
REGION_MEM=${REGION_MEM:-90G}
NRAND=${NRAND:-4}
STRIP_INPUTS=${STRIP_INPUTS:-$OUTDIR/strip_inputs}
STRIP_VARIANTS=${STRIP_VARIANTS:-rs_bcg null_rs null_pz1}
export OUTDIR PLAN CALIB NRAND
# compilation cache on /sps ($HOME is small), no GPU memory preallocation
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$OUTDIR/jax_cache}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
LOGS=$OUTDIR/logs
mkdir -p "$LOGS" "$OUTDIR/runs"
TASK=$REMA/scripts/slurm/rema_task.sh
PZ="--set @$REMA/scripts/photoz/photoz_dr11.yaml"
FILTER="--set @$REMA/scripts/photoz/pz_filter.yaml"
PSCD="--set pscd.snr_kind=background --set pscd.snr_min=$PSCD_SNR"
CUT="--set photoz.mag_max=22"

submit() {   # submit NAME [sbatch options...] -- COMMAND...; prints the job id
    local name=$1; shift
    local opts=()
    while [[ $1 != -- ]]; do opts+=("$1"); shift; done
    shift
    local cmd=(sbatch --parsable --job-name "$name" --output "$LOGS/$name-%A_%a.log" ${EXTRA_SBATCH:-} "${opts[@]}"
               --wrap "$(printf '%q ' "$@")")
    if [[ ${DRY_RUN:-0} == 1 ]]; then
        echo "${cmd[*]}" >&2
        echo 0
    else
        "${cmd[@]}"
    fi
}

# The union of the data boxes of the regions: "RA0 RA1 DEC0 DEC1;..."
area_box() {
    python - "$PLAN" $REGIONS <<'EOF'
import sys
from rema.pipeline import plan_boxes, read_plan
plan, meta = read_plan(sys.argv[1])
out = []
for rid in map(int, sys.argv[2:]):
    _, data = plan_boxes(plan, rid, meta)
    for b in getattr(data, "boxes", [data]):
        out.append(f"{b.ra_min:g} {b.ra_max:g} {b.dec_min:g} {b.dec_max:g}")
print(";".join(out))
EOF
}

fp_path() { echo "$OUTDIR/regions/$(printf %04d "$1")/footprint.fits"; }

# The command of one variant on one region (rema blind or rema pscd), and its kind (gpu or cpu).
variant_cmd() {
    local v=$1 rid=$2
    local common="--galaxies $OUTDIR/galaxies --regions $PLAN --region-id $rid --calib $CALIB --footprint $(fp_path "$rid") --specpost"
    case $v in
        pz_v0)      echo "gpu blind $common $PZ $FILTER" ;;
        pz_v1)      echo "gpu blind $common $PZ $FILTER $CUT" ;;
        null_pz)    echo "gpu blind $common $PZ $FILTER --set null.shuffle=photoz" ;;
        null_pz1)   echo "gpu blind $common $PZ $FILTER $CUT --set null.shuffle=photoz" ;;
        null_rs)    echo "gpu blind $common --set null.shuffle=colour" ;;
        rs_bcg)     echo "gpu blind $common --centering bcg" ;;
        pscd_0)     echo "cpu pscd $common $PZ $PSCD" ;;
        pscd_1)     echo "cpu pscd $common $PZ $PSCD $CUT" ;;
        null_pscd)  echo "cpu pscd $common $PZ $PSCD --set null.shuffle=photoz" ;;
        null_pscd1) echo "cpu pscd $common $PZ $PSCD $CUT --set null.shuffle=photoz" ;;
        *) echo "unknown variant $v" >&2; return 2 ;;
    esac
}

gpu=(--partition "$PART_GPU" --gpus "$GPUS" --cpus-per-task "$REGION_CPUS" --mem "$REGION_MEM" --time 2-00:00:00)
cpu=(--partition "$PART_CPU" --cpus-per-task 8 --mem 64G --time 2-00:00:00)
cpu_small=(--partition "$PART_CPU" --cpus-per-task 4 --mem 24G --time 1-00:00:00)

case $STAGE in
  prepare)
    AREA_BOX="$(area_box)"
    export AREA_BOX
    echo "area: $AREA_BOX"
    missing_fp=0
    for rid in $REGIONS; do
        src=$COSMO_SENS/regions/$(printf %04d "$rid")/footprint.fits
        dst=$(fp_path "$rid")
        mkdir -p "$(dirname "$dst")"
        if [[ ! -s $dst && -s $src ]]; then cp "$src" "$dst"; fi
        [[ -s $dst ]] || missing_fp=1
    done
    # (the randoms are only needed, and submitted, when a footprint could not be copied)
    spec=$(rema todo --sweeps "$DR11/sweep/11.0" --galaxies "$OUTDIR/galaxies" --index "$OUTDIR/randoms_index" --nrand "$NRAND" \
               $(python -c "import sys; [print('--box', *b.split()) for b in sys.argv[1].split(';')]" "$AREA_BOX"))
    echo "$spec"
    ing=$(sed -n 's/^ingest_array=//p' <<< "$spec")
    rnd=$(sed -n 's/^randoms_array=//p' <<< "$spec")
    [[ -n $ing ]] && submit pz-ingest "${cpu_small[@]}" --array "$ing" -- env AREA_BOX="$AREA_BOX" CONFIG="$CONFIG" "$TASK" ingest
    [[ $missing_fp == 1 && -n $rnd ]] && submit pz-randoms "${cpu[@]}" --array "$rnd" -- env AREA_BOX="$AREA_BOX" CONFIG="$CONFIG" "$TASK" randoms
    ;;
  run)
    for rid in $REGIONS; do
        r=$(printf %04d "$rid")
        f=$(fp_path "$rid")
        fpcmd="[[ -s $f ]] || rema maps --index $OUTDIR/randoms_index --nrand $NRAND --regions $PLAN --region-id $rid --out $f"
        base=$OUTDIR/runs/rs_wcen/$r
        if [[ ! -s $base/clusters.fits && -s $COSMO_SENS/tierB/fiducial/$r/clusters.fits ]]; then
            mkdir -p "$base" && cp "$COSMO_SENS/tierB/fiducial/$r/clusters.fits" "$base/clusters.fits"
        fi
        if [[ ! -s $OUTDIR/runs/errscale_$r.txt ]]; then
            submit "pz-errscale-$rid" "${cpu_small[@]}" -- bash -c "python $REMA/scripts/photoz/fit_errscale.py $OUTDIR/galaxies \
$(python -c "
import sys
from rema.pipeline import plan_boxes, read_plan
plan, meta = read_plan(sys.argv[1]); own, _ = plan_boxes(plan, int(sys.argv[2]), meta)
print('--box', own.ra_min, own.ra_max, own.dec_min, own.dec_max)" "$PLAN" "$rid") \
--json $OUTDIR/runs/errscale_$r.json > $OUTDIR/runs/errscale_$r.txt"
        fi
        for v in $VARIANTS; do
            d=$OUTDIR/runs/$v/$r
            [[ -s $d/clusters.fits ]] && continue
            read -r kind cmd <<< "$(variant_cmd "$v" "$rid")"
            if [[ $kind == gpu ]]; then
                opts=("${gpu[@]}"); plat=cuda; ck="--checkpoint $d/checkpoint"
            else
                opts=("${cpu[@]}"); plat=cpu; ck=""
            fi
            # shellcheck disable=SC2086
            submit "pz-$v-$rid" "${opts[@]}" -- env JAX_PLATFORMS=$plat bash -c "$fpcmd && mkdir -p $d && rema $cmd $ck \
--out $d/clusters.tmp.fits && mv $d/clusters.tmp.fits $d/clusters.fits && rm -rf $d/checkpoint"
        done
    done
    ;;
  strip)
    for v in $STRIP_VARIANTS; do
        [[ -s $OUTDIR/strip/$v/clusters.fits ]] && continue
        submit "pz-strip-$v" "${gpu[@]}" -- env JAX_PLATFORMS=cuda PY=python PZ_GAL="$STRIP_INPUTS/galaxies_3sweeps_v2.fits" \
            PZ_FP="$STRIP_INPUTS/footprint_strip.fits" PZ_CALIB="$CALIB" PZ_OUT="$OUTDIR/strip" \
            bash "$REMA/scripts/photoz/photoz_strip.sh" "$v"
    done
    ;;
  status)
    echo "galaxy tables: $(find "$OUTDIR/galaxies" -name 'sweep-*.fits' 2>/dev/null | wc -l)"
    for rid in $REGIONS; do
        r=$(printf %04d "$rid")
        printf 'region %4s: footprint %s, errscale %s\n' "$rid" "$([[ -s $(fp_path "$rid") ]] && echo yes || echo no)" \
            "$([[ -s $OUTDIR/runs/errscale_$r.txt ]] && echo yes || echo no)"
        for v in rs_wcen $VARIANTS; do
            printf '   %-11s %s\n' "$v" "$([[ -s $OUTDIR/runs/$v/$r/clusters.fits ]] && echo done || echo -)"
        done
    done
    for v in $STRIP_VARIANTS; do
        printf 'strip %-11s %s\n' "$v" "$([[ -s $OUTDIR/strip/$v/clusters.fits ]] && echo done || echo -)"
    done
    du -sh "$OUTDIR" 2>/dev/null || true
    ;;
  *)
    echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
