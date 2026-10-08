#!/usr/bin/env bash
# Cosmology sensitivity of the DR11 south catalogue (docs/cosmology_sensitivity.rst), on SLURM.
#
#   source $REMA/scripts/slurm/ccin2p3.env           # and activate the Python environment
#   $REMA/scripts/cosmo_sens/cosmo_sens.sh STAGE
#
# Stages (each submits jobs and returns; run them in this order, each once the previous jobs are
# done; `status` tells what is there):
#   prepare  galaxy tables and randoms index of the study regions and the calibration area, and a
#            small copy of the catalogue and its members there (catalog.fits, members.fits)
#   tierA    per region: footprint, then `rema remeasure` of the catalogue's clusters at fixed
#            centres on the cosmology grid (members' free fractions, autodiff and finite
#            differences), and with m* following the luminosity distance
#   tierB    per region and cosmology: `rema blind` with --cosmology (fixed calibration)
#   tierC    per cosmology: `rema calibrate` with --cosmology, then `rema blind` on TIERC_REGIONS
#   status   what is done
#
# Environment (defaults in brackets): PRODUCTS [the DR11 rema products on the data system], RUN
# [$PRODUCTS/rema_dr11_v0.2.0_ra0-240: the plan, the calibration and its footprint], CATALOG and
# MEMBERS [the combined catalogue], OUTDIR [/sps/lsst/users/$USER/rema_cosmo_sens], REGIONS
# [15 9 55 82 91 113], TIERC_REGIONS [15 82], GRID_A, GRID_B, GRID_C (cosmologies, see below),
# LAMBDA_MIN [5], JAX_COMPILATION_CACHE_DIR [$OUTDIR/jax_cache], DRY_RUN=1 (print the sbatch
# commands only). Partitions, GPUs, memory, NRAND, DR11 and EXTRA_SBATCH as for
# scripts/slurm/rema_dr11_blind.sh.
set -euo pipefail

STAGE=${1:?stage: prepare, tierA, tierB, tierC or status}
REMA=${REMA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PRODUCTS=${PRODUCTS:-${DR11:?set DR11}/rema}
RUN=${RUN:-$PRODUCTS/rema_dr11_v0.2.0_ra0-240}
PLAN=${PLAN:-$RUN/regions.fits}
CALIB=${CALIB:-$RUN/calib/calib.fits}
CALIB_FOOTPRINT=${CALIB_FOOTPRINT:-$RUN/calib/footprint.fits}
OUTDIR=${OUTDIR:-/sps/lsst/users/$USER/rema_cosmo_sens}
CATALOG=${CATALOG:-$PRODUCTS/rema_dr11_v0.2.0/clusters_dr11.fits}
MEMBERS=${MEMBERS:-$PRODUCTS/rema_dr11_v0.2.0/clusters_dr11_members.fits}
CONFIG=${CONFIG:-$REMA/scripts/slurm/dr11_south.yaml}
# the copies made by `prepare`
SLIM_CAT=$OUTDIR/catalog.fits
SLIM_MEM=$OUTDIR/members.fits
REGIONS=${REGIONS:-15 9 55 82 91 113}
TIERC_REGIONS=${TIERC_REGIONS:-15 82}
CALIB_BOX=${CALIB_BOX:-160 180 -10 10;190 210 -10 10}
LAMBDA_MIN=${LAMBDA_MIN:-5}
# One-at-a-time variations (tier A: rema remeasure --vary; tiers B and C: one run per entry).
GRID_A=${GRID_A:-Omega_m=0.25,0.35 w0=-0.8,-1.2 h=0.65,0.75 sum_mnu=0.12,0.24 wa=-0.3,0.3 Omega_b=0.04,0.06}
GRID_B=${GRID_B:-fiducial Omega_m=0.25 Omega_m=0.35 w0=-0.8 w0=-1.2}
GRID_C=${GRID_C:-fiducial Omega_m=0.25 Omega_m=0.35}
PART_CPU=${PART_CPU:-htc}
PART_GPU=${PART_GPU:-gpu_v100}
GPUS=${GPUS:-1}
REGION_CPUS=${REGION_CPUS:-5}
REGION_MEM=${REGION_MEM:-45G}
NRAND=${NRAND:-4}
# The jobs' JAX compilation cache next to the results (rema's default, ~/.cache/rema/jax, would
# fill $HOME).
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$OUTDIR/jax_cache}
export OUTDIR PLAN CALIB NRAND
LOGS=$OUTDIR/logs
mkdir -p "$LOGS" "$OUTDIR/tierA" "$OUTDIR/tierB" "$OUTDIR/tierC" "$JAX_COMPILATION_CACHE_DIR"
TASK=$REMA/scripts/slurm/rema_task.sh

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

# "Omega_m=0.25" -> "--cosmology Omega_m=0.25"; "fiducial" -> nothing
cosmo_opt() { [[ $1 == fiducial ]] || printf -- '--cosmology %s' "$1"; }

# The union of the data boxes of the study regions, and the calibration boxes: "RA0 RA1 DEC0 DEC1;..."
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

# The footprint of a region (built by the first job that needs it; written atomically).
fp_cmd() {
    local d=$OUTDIR/regions/$(printf %04d "$1")
    printf 'mkdir -p %s && { [[ -s %s/footprint.fits ]] || rema maps --index %s/randoms_index --nrand %s --regions %s --region-id %s --out %s/footprint.fits; }' \
        "$d" "$d" "$OUTDIR" "$NRAND" "$PLAN" "$1" "$d"
}

gpu=(--partition "$PART_GPU" --gpus "$GPUS" --cpus-per-task "$REGION_CPUS" --mem "$REGION_MEM" --time 1-00:00:00)
cpu=(--partition "$PART_CPU" --cpus-per-task 8 --mem 64G --time 2-00:00:00)
cpu_small=(--partition "$PART_CPU" --cpus-per-task 4 --mem 24G --time 1-00:00:00)

case $STAGE in
  prepare)
    AREA_BOX="$(area_box);$CALIB_BOX"
    export AREA_BOX
    echo "area: $AREA_BOX"
    spec=$(rema todo --sweeps "$DR11/sweep/11.0" --galaxies "$OUTDIR/galaxies" --index "$OUTDIR/randoms_index" \
               --nrand "$NRAND" $(python -c "import sys; [print('--box', *b.split()) for b in sys.argv[1].split(';')]" "$AREA_BOX"))
    echo "$spec"
    ing=$(sed -n 's/^ingest_array=//p' <<< "$spec")
    rnd=$(sed -n 's/^randoms_array=//p' <<< "$spec")
    [[ -n $ing ]] && submit cs-ingest "${cpu_small[@]}" --array "$ing" -- env AREA_BOX="$AREA_BOX" CONFIG="$CONFIG" "$TASK" ingest
    [[ -n $rnd ]] && submit cs-randoms "${cpu[@]}" --array "$rnd" -- env AREA_BOX="$AREA_BOX" CONFIG="$CONFIG" "$TASK" randoms
    if [[ ! -s $SLIM_MEM ]]; then
        boxes=$(python -c "import sys; print(' '.join('--box ' + b for b in sys.argv[1].split(';')))" "$(area_box)")
        # shellcheck disable=SC2086
        submit cs-slim "${cpu[@]}" -- python "$REMA/scripts/cosmo_sens/slim_catalog.py" "$CATALOG" "$MEMBERS" \
            $boxes --out-clusters "$SLIM_CAT" --out-members "$SLIM_MEM"
    fi
    ;;
  tierA)
    vary=()
    for v in $GRID_A; do vary+=(--vary "$v"); done
    for rid in $REGIONS; do
        d=$OUTDIR/regions/$(printf %04d "$rid")
        script="set -euo pipefail; mkdir -p $d
[[ -s $d/footprint.fits ]] || rema maps --index $OUTDIR/randoms_index --nrand $NRAND --regions $PLAN --region-id $rid --out $d/footprint.fits
common=(--galaxies $OUTDIR/galaxies --calib $CALIB --footprint $d/footprint.fits --regions $PLAN --region-id $rid --catalog $SLIM_CAT --lambda-min $LAMBDA_MIN)
[[ -s $OUTDIR/tierA/$rid.fits ]] || rema remeasure \"\${common[@]}\" --members $SLIM_MEM ${vary[*]} --jvp Omega_m,w0 --fd --out $OUTDIR/tierA/$rid.fits
[[ -s $OUTDIR/tierA/${rid}_mstar.fits ]] || rema remeasure \"\${common[@]}\" --members $SLIM_MEM --vary Omega_m=0.25,0.35 --vary w0=-0.8,-1.2 --mstar-follows-cosmology --out $OUTDIR/tierA/${rid}_mstar.fits
[[ -s $OUTDIR/tierA/${rid}_one.fits ]] || rema remeasure \"\${common[@]}\" --vary Omega_m=0.25,0.35 --vary w0=-0.8,-1.2 --out $OUTDIR/tierA/${rid}_one.fits"
        submit "cs-tierA-$rid" "${gpu[@]}" -- env JAX_PLATFORMS=cuda bash -c "$script"
    done
    ;;
  tierB)
    for c in $GRID_B; do
        for rid in $REGIONS; do
            d=$OUTDIR/tierB/$c/$(printf %04d "$rid")
            f=$OUTDIR/regions/$(printf %04d "$rid")/footprint.fits
            [[ -s $d/clusters.fits ]] && continue
            submit "cs-tierB-$rid" "${gpu[@]}" -- env JAX_PLATFORMS=cuda bash -c "$(fp_cmd "$rid") && mkdir -p $d && rema blind \
--galaxies $OUTDIR/galaxies --regions $PLAN --region-id $rid --calib $CALIB --footprint $f $(cosmo_opt "$c") \
--checkpoint $d/checkpoint --out $d/clusters.fits && rm -rf $d/checkpoint"
        done
    done
    ;;
  tierC)
    mapfile -t cbox < <(python -c "import sys; [print('--box', *b.split(), sep=chr(10)) for b in sys.argv[1].split(';')]" "$CALIB_BOX")
    for c in $GRID_C; do
        cal=$OUTDIR/tierC/$c/calib.fits
        dep=()
        if [[ ! -s $cal ]]; then
            jid=$(submit "cs-calib" "${cpu[@]}" -- bash -c "mkdir -p $OUTDIR/tierC/$c && rema calibrate --galaxies \
$OUTDIR/galaxies ${cbox[*]} --footprint $CALIB_FOOTPRINT --config $CONFIG $(cosmo_opt "$c") \
--out $cal --plots $OUTDIR/tierC/$c/plots")
            dep=(--dependency "afterok:$jid")
        fi
        for rid in $TIERC_REGIONS; do
            d=$OUTDIR/tierC/$c/$(printf %04d "$rid")
            f=$OUTDIR/regions/$(printf %04d "$rid")/footprint.fits
            [[ -s $d/clusters.fits ]] && continue
            submit "cs-tierC-$rid" "${gpu[@]}" "${dep[@]}" -- env JAX_PLATFORMS=cuda bash -c "$(fp_cmd "$rid") && mkdir -p $d && rema blind \
--galaxies $OUTDIR/galaxies --regions $PLAN --region-id $rid --calib $cal --footprint $f \
--checkpoint $d/checkpoint --out $d/clusters.fits && rm -rf $d/checkpoint"
        done
    done
    ;;
  status)
    echo "galaxy tables: $(find "$OUTDIR/galaxies" -name 'sweep-*.fits' 2>/dev/null | wc -l)"
    echo "randoms index: $(find "$OUTDIR/randoms_index" -name '*.json' 2>/dev/null | wc -l)"
    for rid in $REGIONS; do
        printf 'region %4s: footprint %s, tier A %s\n' "$rid" \
            "$([[ -s $OUTDIR/regions/$(printf %04d "$rid")/footprint.fits ]] && echo yes || echo no)" \
            "$(ls "$OUTDIR/tierA/$rid"*.fits 2>/dev/null | wc -l)/3"
    done
    for c in $GRID_B; do echo "tier B $c: $(ls "$OUTDIR"/tierB/"$c"/*/clusters.fits 2>/dev/null | wc -l)/$(wc -w <<< "$REGIONS")"; done
    for c in $GRID_C; do echo "tier C $c: calib $([[ -s $OUTDIR/tierC/$c/calib.fits ]] && echo yes || echo no), $(ls "$OUTDIR"/tierC/"$c"/*/clusters.fits 2>/dev/null | wc -l)/$(wc -w <<< "$TIERC_REGIONS")"; done
    du -sh "$OUTDIR" 2>/dev/null || true
    ;;
  *)
    echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
