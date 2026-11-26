#!/bin/bash
# Scan-mode randoms at CC-IN2P3: lambda(z) at fixed positions on the nested HEALPix grid
# (`rema grid-points`, `rema grid-lambda`).
#
#   scripts/randoms/grid_lambda.sh submit        prep -> ingest array -> grid-lambda array -> merge
#   scripts/randoms/grid_lambda.sh prep|ingest|run|merge   (the steps, run by the jobs)
#
# Settings (environment):
#   WORK     [$SPS/rema_gridlam]                 points, task outputs, merged grid, logs
#   CODE     [$WORK/code]                        rema code put first on PYTHONPATH
#   GALWORK  [$SPS/rema_scan]                    galaxies/ (ingested sweeps, kept) and randoms_index/
#   DOMAIN   [$SPS/rema_randoms/products/domain.fits]   run domain: the points and the sweeps
#   LEVELS   ["64 128"]                          grid levels measured in this run (NSIDE, 4096 grid)
#   NGAL [20] NTASK [30] MAX_ARRAY [30]          ingest tasks, grid-lambda tasks, concurrent tasks
#   ZSTEP [0.02]  BUFFER [1.2]                   redshift step, deg of galaxies around each sweep box
set -euo pipefail
SPS=${SPS:-/sps/lsst/users/$USER}
WORK=${WORK:-$SPS/rema_gridlam}
CODE=${CODE:-$WORK/code}
GALWORK=${GALWORK:-$SPS/rema_scan}
DOMAIN=${DOMAIN:-$SPS/rema_randoms/products/domain.fits}
LEVELS=${LEVELS:-"64 128"}
NGAL=${NGAL:-20}
NTASK=${NTASK:-30}
MAX_ARRAY=${MAX_ARRAY:-30}
ZSTEP=${ZSTEP:-0.02}
BUFFER=${BUFFER:-1.2}
DR11=${DR11:-/sps/lsst/datasets/desi/legacysurveys/dr11/south}
TAG=l$(echo $LEVELS | tr ' ' '_')
POINTS=$WORK/points_$TAG.fits
SELF=$(readlink -f "$0")

env_setup() {
  source $SPS/miniforge3/etc/profile.d/conda.sh
  conda activate $SPS/envs/rema
  export PYTHONPATH=$CODE${PYTHONPATH:+:$PYTHONPATH}
  export JAX_COMPILATION_CACHE_DIR=$GALWORK/jax_cache XLA_PYTHON_CLIENT_PREALLOCATE=false
  cd $WORK
}

case "${1:-}" in
prep)
  env_setup
  for n in $LEVELS; do
    [[ -f points_$n.fits ]] || python -m rema.cli grid-points --nside $n --domain $DOMAIN --out points_$n.fits
  done
  LEVELS="$LEVELS" POINTS=$POINTS DOMAIN=$DOMAIN DR11=$DR11 BUFFER=$BUFFER python - <<'P'
import os
import numpy as np
from rema.io.tables import read_table, table_hdu, write_fits
from rema.randoms.domain import RunDomain
from rema.sky.regions import Box, sky_union, sweeps_overlapping

parts = [read_table(f"points_{n}.fits", hdu="POINTS") for n in os.environ["LEVELS"].split()]
pts = {c: np.concatenate([np.asarray(p[c]) for p in parts]) for c in ("PIX", "RA", "DEC", "LEVEL")}
o = np.argsort(pts["PIX"], kind="stable")
pts = {c: v[o] for c, v in pts.items()}
write_fits(os.environ["POINTS"], [table_hdu(pts, {"LEVELS": os.environ["LEVELS"]}, "POINTS")])
dom = RunDomain.read(os.environ["DOMAIN"])
buf = float(os.environ["BUFFER"])
sky = sky_union([Box(*map(float, b)).buffered(buf) for b in dom.boxes])
sw = sorted(str(s) for s in sweeps_overlapping(sky, os.path.join(os.environ["DR11"], "sweep", "11.0"))
            if not s.name.endswith("-pz.fits"))
open("sweeps.txt", "w").write("\n".join(sw) + "\n")
print(f"{pts['PIX'].size} points, {len(sw)} sweeps")
P
  ;;
ingest)
  env_setup
  i=${SLURM_ARRAY_TASK_ID:-0}
  GALDIR=$GALWORK/galaxies I=$i N=$NGAL CAL=$DR11/rema/rema_dr11_v0.2.0_ra0-240/calib/calib.fits python - <<'P'
import os
from rema.calibration import Calibration
from rema.io.legacy import ingest_sweeps

i, n = int(os.environ["I"]), int(os.environ["N"])
sw = [s for s in open("sweeps.txt").read().split() if s][i::n]
cfg = Calibration.read(os.environ["CAL"]).config
ingest_sweeps(sw, os.environ["GALDIR"], cfg, require_pz=True)
print(f"task {i}: {len(sw)} sweeps")
P
  ;;
run)
  env_setup
  i=${SLURM_ARRAY_TASK_ID:-0}
  mkdir -p tasks_$TAG
  python -m rema.cli grid-lambda $POINTS --task $i --ntask $NTASK --zstep $ZSTEP --work $GALWORK \
    --index $GALWORK/randoms_index --out tasks_$TAG/grid_$(printf %03d $i).fits
  ;;
merge)
  env_setup
  TAG=$TAG NTASK=$NTASK python - <<'P'
import glob, os
import numpy as np
from astropy.io import fits
from rema.io.tables import read_table, table_hdu, write_fits

tag, n = os.environ["TAG"], int(os.environ["NTASK"])
files = sorted(glob.glob(f"tasks_{tag}/grid_*.fits"))
if len(files) != n:
    raise SystemExit(f"{len(files)} of {n} task outputs")
parts = [read_table(f, hdu="GRID") for f in files]
out = {c: np.concatenate([np.asarray(p[c]) for p in parts]) for c in parts[0]}
z = read_table(files[0], hdu="ZSTEPS")
t = sum(fits.getheader(f, "GRID")["TTOTAL"] for f in files)
hdr = {"MODE": "grid-lambda", "LEVELS": tag, "NPOS": out["PIX"].size, "NSCANNED": int(out["SCANNED"].sum()),
       "TTASKS": round(t, 1)}
write_fits(f"grid_{tag}.fits", [table_hdu(out, hdr, "GRID"), table_hdu({"Z": np.asarray(z["Z"])}, None, "ZSTEPS")])
print(f"grid_{tag}.fits: {out['PIX'].size} points, {int(out['SCANNED'].sum())} scanned, {t:.0f} task-seconds")
P
  ;;
submit)
  mkdir -p $WORK/logs
  common=(--parsable --licenses=sps --export=ALL)
  j0=$(sbatch "${common[@]}" --job-name=gridlam-prep --partition=htc --cpus-per-task=2 --mem=16G \
       --time=2:00:00 --output=$WORK/logs/prep-%j.log "$SELF" prep)
  j1=$(sbatch "${common[@]}" --job-name=gridlam-ingest --dependency=afterok:$j0 --partition=htc \
       --array=0-$((NGAL - 1)) --cpus-per-task=4 --mem=16G --time=12:00:00 \
       --output=$WORK/logs/ingest-%A_%a.log "$SELF" ingest)
  j2=$(sbatch "${common[@]}" --job-name=gridlam-run --dependency=afterok:$j1 --partition=gpu_v100 --gpus=1 \
       --array=0-$((NTASK - 1))%$MAX_ARRAY --cpus-per-task=5 --mem=45G --time=24:00:00 \
       --output=$WORK/logs/run-%A_%a.log "$SELF" run)
  j3=$(sbatch "${common[@]}" --job-name=gridlam-merge --dependency=afterok:$j2 --partition=htc \
       --cpus-per-task=2 --mem=32G --time=2:00:00 --output=$WORK/logs/merge-%j.log "$SELF" merge)
  echo "prep $j0 ingest $j1 run $j2 merge $j3"
  ;;
*)
  sed -n '2,16p' "$SELF"; exit 1 ;;
esac
