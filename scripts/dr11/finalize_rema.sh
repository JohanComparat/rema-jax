#!/bin/bash
# finalize_rema.sh OUTDIR DEST: after a merge, copy a rema run's final products to DEST, check
# the copies byte for byte, then delete OUTDIR. Submitted with --dependency=afterok on the merge.
# Used for the DR11 south production of rema 0.2.0 at CC-IN2P3 (docs: "The DR11 south catalogue at
# CC-IN2P3"), e.g. finalize_rema.sh $OUTDIR /sps/lsst/datasets/desi/legacysurveys/dr11/south/rema/<run>.
set -euo pipefail
O=$1 D=$2
U=/sps/lsst/users/$USER
[[ $O == $U/rema_dr11_* && -d $O ]] || { echo "unexpected OUTDIR $O"; exit 1; }
for c in $U/miniforge3 $HOME/miniforge3; do [[ -f $c/etc/profile.d/conda.sh ]] && { source $c/etc/profile.d/conda.sh; break; }; done
conda activate $U/envs/rema
export JAX_PLATFORMS=cpu
echo "1. checks"
for f in clusters_dr11.fits clusters_dr11_members.fits clusters_dr11_regions.fits clusters_dr11_qa.json regions.fits calib/calib.fits; do
    [[ -s $O/$f ]] || { echo "missing $O/$f: nothing moved or deleted"; exit 1; }
done
python - "$O/clusters_dr11_qa.json" "$O/regions.fits" <<'PY'
# Complete, or missing only regions that reach below |b| = 15 deg (failures there are not rerun).
import json, os, sys
from rema.pipeline import box_abs_glat, plan_boxes, read_plan
q = json.load(open(sys.argv[1]))
plan, meta = read_plan(sys.argv[2])
print({k: q.get(k) for k in ("n_regions", "n_done", "n_clusters", "n_clusters_low_glat", "glat_min",
                              "missing", "duplicate_ids", "centres_outside_own", "members_without_cluster")})
missing = [int(r) for r in (q.get("missing") or [])]
accepted = {int(r) for r in os.environ.get("ACCEPT_MISSING", "").split()}   # dropped on purpose
high = []
for rid in [r for r in missing if r not in accepted]:
    own, _ = plan_boxes(plan, rid, meta)
    if box_abs_glat(own.bounding() if hasattr(own, "boxes") else own)[0] >= 15:
        high.append(rid)
ok = (q.get("glat_min") == 15 and q["n_done"] + len(missing) == q["n_regions"] and not high
      and q.get("duplicate_ids") == 0)
print(f"missing regions {missing}; accepted on purpose {sorted(accepted)}; others entirely at |b| >= 15: {high}")
sys.exit(0 if ok else "merge QA not acceptable: nothing moved or deleted")
PY
echo "2. copy to $D"
mkdir -p "$D"
cp -a "$O"/clusters_dr11.fits "$O"/clusters_dr11_members.fits "$O"/clusters_dr11_regions.fits \
      "$O"/clusters_dr11_qa.json "$O"/regions.fits "$O"/calib "$O"/logs "$D"/
for f in clusters_dr11.fits clusters_dr11_members.fits clusters_dr11_regions.fits clusters_dr11_qa.json regions.fits; do
    cmp "$O/$f" "$D/$f"
done
diff -rq "$O/calib" "$D/calib"
diff -rq "$O/logs" "$D/logs"
echo "copies identical"; ls -la "$D"
echo "3. delete $O"
rm -rf "$O"
du -sh --apparent-size "$U" "$D"
echo "done"
