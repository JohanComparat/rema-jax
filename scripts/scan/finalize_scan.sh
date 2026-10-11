#!/bin/bash
# finalize_scan.sh WORK DEST: after the rema scan-positions jobs of the public cluster catalogues
# (public_catalogues.py inputs in WORK/inputs/full, outputs in WORK/out/full and, for the RA
# slices, WORK/out/full/parts), join the slices, check every output against its input, copy the
# outputs, inputs and logs to DEST, check the copies byte for byte, then delete WORK (the ingested
# sweeps, randoms index and JAX cache included). Submitted with --dependency=afterok on the scans,
# e.g. finalize_scan.sh /sps/lsst/users/$USER/rema_scan \
#          /sps/lsst/datasets/desi/legacysurveys/dr11/south/rema/rema_dr11_v0.2.0_scan_public
# KEEP_WORK=1 copies without deleting WORK.
set -euo pipefail
W=$1 D=$2
U=/sps/lsst/users/$USER
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $W == "$U"/rema_scan* && -d $W/inputs/full && -d $W/out/full ]] || { echo "unexpected WORK $W"; exit 1; }
for c in $U/miniforge3 $HOME/miniforge3; do [[ -f $c/etc/profile.d/conda.sh ]] && { source $c/etc/profile.d/conda.sh; break; }; done
conda activate $U/envs/rema
export JAX_PLATFORMS=cpu

echo "1. join the slices"
mapfile -t sliced < <(find "$W/inputs/full" -name '*_ra[0-9]*.fits' -printf '%f\n' | sed 's/_ra[0-9.]*\.fits$//' | sort -u)
for n in "${sliced[@]}"; do
    ni=$(find "$W/inputs/full" -name "${n}_ra[0-9]*.fits" | wc -l)
    no=$(find "$W/out/full/parts" -name "${n}_ra[0-9]*.fits" | wc -l)
    (( ni == no )) || { echo "$n: $no outputs for $ni slices: nothing moved or deleted"; exit 1; }
    [[ -s $W/out/full/$n.fits ]] || python "$HERE/merge_parts.py" "$W/out/full/$n.fits" "$W/out/full/parts/${n}_ra"*.fits
done

echo "2. checks"
python - "$W" "${sliced[@]}" <<'PY'
# Every input has an output with one row per input row, in order.
import json, sys
from pathlib import Path
import numpy as np
from astropy.io import fits
w, sliced = Path(sys.argv[1]), set(sys.argv[2:])
names = sorted({p.name.split("_ra")[0] if p.name.split("_ra")[0] in sliced else p.stem
                for p in (w / "inputs/full").glob("*.fits")})
summary = {}
for n in names:
    ins = sorted((w / "inputs/full").glob(f"{n}_ra*.fits")) if n in sliced else [w / "inputs/full" / f"{n}.fits"]
    row = np.concatenate([fits.getdata(f, 1)["ROW"] for f in ins])
    out = w / "out/full" / f"{n}.fits"
    if not out.exists():
        sys.exit(f"missing {out}: nothing moved or deleted")
    with fits.open(out, memmap=True) as h:
        c = h["CLUSTERS"].data
        if not np.array_equal(np.sort(row), np.asarray(c["INPUT_ROW"])):
            sys.exit(f"{out}: rows do not match its input: nothing moved or deleted")
        nm = h["MEMBERS"].header.get("NAXIS2", 0) if "MEMBERS" in h else 0
        summary[n] = {"n_input": int(row.size), "n_scanned": int(np.sum(c["SCANNED"])),
                      "n_lambda_pos": int(np.sum(c["LAMBDA"] > 0)), "n_lambda_20": int(np.sum(c["LAMBDA"] > 20)),
                      "n_members": int(nm)}
    print(n, summary[n])
(w / "out/full/summary.json").write_text(json.dumps(summary, indent=1) + "\n")
PY

echo "3. copy to $D"
mkdir -p "$D/inputs" "$D/logs"
for f in "$W"/out/full/*.fits "$W"/out/full/summary.json; do cp -a "$f" "$D/"; done
cp -a "$W"/inputs/full/. "$D/inputs/"
cp -a "$W"/logs/. "$D/logs/"
cat > "$D/README.txt" <<EOF
rema scan-positions on public cluster catalogues, LS DR11 south ($(date +%F))

One FITS file per catalogue (the inputs of scripts/scan/public_catalogues.py, in inputs/):
CLUSTERS has one row per input cluster, in input order: the input columns as INPUT_<NAME>
(INPUT_ROW = row in the original catalogue file in ../external), the scan-mode and spectroscopic
post-processing columns of rema (LAMBDA, Z_LAMBDA, RA/DEC of the centre, BEST_Z, NSPEC, ...) and
SCANNED (false outside the DR11 south sweeps; pipeline columns are -1 there). MEMBERS: the members,
keyed by MEM_MATCH_ID = the input row. CONFIG: the rema configuration.
Calibration: rema_dr11_v0.2.0_ra0-240/calib/calib.fits (the production one). No |b| or E(B-V) cut
on the positions: cut on the input coordinates as needed (the production catalogue uses |b| > 15).
Large catalogues ran in 15 deg RA slices (inputs/<name>_ra*.fits) joined by scripts/scan/merge_parts.py.
summary.json: rows, scanned, lambda > 0, lambda > 20, members per catalogue. Code: rema-jax $(git -C "$HERE" rev-parse --short HEAD).
EOF
for f in "$W"/out/full/*.fits "$W"/out/full/summary.json; do cmp "$f" "$D/$(basename "$f")"; done
diff -rq "$W/inputs/full" "$D/inputs"
diff -rq "$W/logs" "$D/logs"
echo "copies identical"; ls -la "$D"

if [[ ${KEEP_WORK:-0} == 1 ]]; then echo "KEEP_WORK=1: $W kept"; exit 0; fi
echo "4. delete $W"
du -sh "$W"
rm -rf "$W"
du -sh "$U" "$D"
echo "done"
