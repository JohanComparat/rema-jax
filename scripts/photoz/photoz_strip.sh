#!/usr/bin/env bash
# Photo-z cluster finding on the DR11 strip (RA 0-5, Dec -15..0): the red-sequence finder, the
# photo-z filter and PSCD on the same galaxies, with null tests (docs/photoz_finders.rst).
#
#   scripts/photoz/photoz_strip.sh [VARIANT ...]      (default: all, in the order below)
#
# Variants (each writes $OUT/<variant>/clusters.fits and a log):
#   rs_wcen   red-sequence blind run, as the production (wcen centring)
#   rs_bcg    the same with BCG centring (the photo-z finders centre on the brightest member)
#   pz_v0     photo-z filter, luminosity limit m* + 1.75
#   pz_v1     photo-z filter, and members brighter than z = 22 (photoz.mag_max)
#   pscd_0    PSCD down to the catalogue limit
#   pscd_1    PSCD, galaxies brighter than z = 22
#   null_rs   rs_wcen with the colours shuffled (null.shuffle: colour)
#   null_pz   pz_v0 with the photo-z shuffled (null.shuffle: photoz)
#   null_pscd pscd_0 with the photo-z shuffled
#   null_pz1, null_pscd1  the same for pz_v1 and pscd_1
#
# Environment: PY (python with rema importable and a working jaxlib), PZ_DATA [the local rema
# products], PZ_GAL and PZ_FP [the strip's galaxy table and footprint], PZ_CALIB [the production
# calibration], PZ_OUT [notebooks/photoz/strip], PSCD_SNR [4: background S/N of the PSCD
# extraction]. An existing clusters.fits is kept.
set -euo pipefail

REMA=${REMA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PY:-python}
PZ_DATA=${PZ_DATA:-$HOME/data/legacysurvey/dr11/south/rema}
GAL=${PZ_GAL:-$PZ_DATA/strip/galaxies_3sweeps_v2.fits}
FP=${PZ_FP:-$PZ_DATA/strip/footprint_strip.fits}
CALIB=${PZ_CALIB:-$PZ_DATA/rema_dr11_v0.2.0_ra0-240/calib/calib.fits}
OUT=${PZ_OUT:-$PZ_DATA/notebooks/photoz/strip}
PSCD_SNR=${PSCD_SNR:-4}
BOX=(--box 0 5 -15 0 --own 1 4 -14 -1)
PZ=(--set "@$REMA/scripts/photoz/photoz_dr11.yaml")
FILTER=(--set "@$REMA/scripts/photoz/pz_filter.yaml")
PSCD=(--set pscd.snr_kind=background --set "pscd.snr_min=$PSCD_SNR")
CUT=(--set photoz.mag_max=22)
export PYTHONPATH=$REMA${PYTHONPATH:+:$PYTHONPATH}

rema() { "$PY" -m rema.cli "$@"; }

run() {   # run VARIANT COMMAND ARGS...: rema COMMAND with the common inputs, into $OUT/VARIANT
    local v=$1 cmd=$2
    shift 2
    local d=$OUT/$v
    mkdir -p "$d"
    if [[ -s $d/clusters.fits ]]; then
        echo "$v: done"
        return
    fi
    echo "$v: rema $cmd $*"
    local common=(--galaxies "$GAL" --calib "$CALIB" --footprint "$FP" "${BOX[@]}" --specpost)
    rema "$cmd" "${common[@]}" "$@" --out "$d/clusters.tmp.fits" > "$d/run.log" 2>&1
    mv "$d/clusters.tmp.fits" "$d/clusters.fits"
}

variants=("$@")
[[ ${#variants[@]} -gt 0 ]] || variants=(rs_wcen rs_bcg pz_v0 pz_v1 pscd_0 pscd_1 null_rs null_pz null_pscd
                                        null_pz1 null_pscd1)
for v in "${variants[@]}"; do
    case $v in
        rs_wcen)   run "$v" blind ;;
        rs_bcg)    run "$v" blind --centering bcg ;;
        pz_v0)     run "$v" blind "${PZ[@]}" "${FILTER[@]}" ;;
        pz_v1)     run "$v" blind "${PZ[@]}" "${FILTER[@]}" "${CUT[@]}" ;;
        pscd_0)    run "$v" pscd "${PZ[@]}" "${PSCD[@]}" ;;
        pscd_1)    run "$v" pscd "${PZ[@]}" "${PSCD[@]}" "${CUT[@]}" ;;
        null_rs)   run "$v" blind --set null.shuffle=colour ;;
        null_pz)   run "$v" blind "${PZ[@]}" "${FILTER[@]}" --set null.shuffle=photoz ;;
        null_pscd) run "$v" pscd "${PZ[@]}" "${PSCD[@]}" --set null.shuffle=photoz ;;
        null_pz1)  run "$v" blind "${PZ[@]}" "${FILTER[@]}" "${CUT[@]}" --set null.shuffle=photoz ;;
        null_pscd1) run "$v" pscd "${PZ[@]}" "${PSCD[@]}" "${CUT[@]}" --set null.shuffle=photoz ;;
        *) echo "unknown variant $v" >&2; exit 2 ;;
    esac
done
