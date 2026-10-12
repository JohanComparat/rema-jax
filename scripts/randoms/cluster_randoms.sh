#!/bin/bash
# Cluster randoms of the DR11 south catalogue at CC-IN2P3 (docs: cluster_randoms).
#
#   scripts/randoms/cluster_randoms.sh config     # the catalogue's configuration, once
#   scripts/randoms/cluster_randoms.sh submit     # index, footprint, z_vlim, copies, place, samples
#   scripts/randoms/cluster_randoms.sh external   # the X-ray / SZ samples
#   scripts/randoms/cluster_randoms.sh status
#
# Stages, chained with afterok:
#   index  (htc array, one task per randoms file): rema randoms-index into $WORK/randoms_index
#   maps   (htc): full-sky footprint from the NRAND indexed files
#   copies (htc): domain, z_vlim map, copies of the parent sample ($SPEC)
#   place  (GPU): SCALEVAL, MASKFRAC of every copy
#   sample (htc): the samples of $SAMPLES (YAML files) and their QA
# Products in $WORK/products; nothing is deleted.
#
# Code: $CODE (default $WORK/code, a `git archive` of the branch) is put first on PYTHONPATH, so
# the conda env's rema is not used. Override any variable by exporting it first.
set -euo pipefail

SPS=/sps/lsst/users/$USER
DR11=${DR11:-/sps/lsst/datasets/desi/legacysurveys/dr11/south}
REMA_OUT=${REMA_OUT:-$DR11/rema}
WORK=${WORK:-$SPS/rema_randoms}
CODE=${CODE:-$WORK/code}
NRAND=${NRAND:-4}
CAT=${CAT:-$REMA_OUT/rema_dr11_v0.2.0/clusters_dr11.fits}
RUNS=${RUNS:-"$REMA_OUT/rema_dr11_v0.2.0_ra0-240 $REMA_OUT/rema_dr11_v0.2.0_ra240-360"}
SPEC=${SPEC:-$CODE/scripts/randoms/parent_lambda.yaml}
SAMPLES=${SAMPLES:-"$CODE/scripts/randoms/lambda_vlim.yaml $CODE/scripts/randoms/lambda_vlim_s.yaml"}
NSIGMA=${NSIGMA:-5}
LSTAR=${LSTAR:-0.2}
SEAM_BUFFER=${SEAM_BUFFER:-0}
PART_CPU=${PART_CPU:-htc}
PART_GPU=${PART_GPU:-gpu_v100}

IDX=${IDX:-$WORK/randoms_index}      # an existing full index (e.g. that of rema_scan) is reused
P=$WORK/products
LOGS=$WORK/logs
mkdir -p "$IDX" "$P" "$LOGS"

env_lines() {
  cat <<EOF
source $SPS/miniforge3/etc/profile.d/conda.sh
conda activate $SPS/envs/rema
export PYTHONPATH=$CODE\${PYTHONPATH:+:\$PYTHONPATH}
export XDG_CACHE_HOME=$SPS/.cache MPLCONFIGDIR=$SPS/.cache/matplotlib
export JAX_COMPILATION_CACHE_DIR=$WORK/jax_cache XLA_PYTHON_CLIENT_PREALLOCATE=false
set -euo pipefail
EOF
}

submit() {   # name, deps, sbatch options..., -- command
  local name=$1 dep=$2; shift 2
  local opts=()
  while [[ $1 != -- ]]; do opts+=("$1"); shift; done
  shift
  local d=()
  [[ -n $dep ]] && d=(--dependency=afterok:$dep)
  sbatch --parsable --job-name="crand-$name" --licenses=sps --output="$LOGS/%x-%A_%a.log" \
    "${d[@]}" "${opts[@]}" --wrap="$(env_lines)
echo \"\$(date) \$SLURM_JOB_ID on \$(hostname)\"
$*
echo \"\$(date) done\""
}

case ${1:-} in
  submit)
    [[ -d $CODE/rema ]] || { echo "no code in $CODE"; exit 1; }
    [[ -f $P/catalog_config.yaml ]] || { echo "run $0 config first"; exit 1; }
    files=""
    for k in $(seq 0 $((NRAND - 1))); do files="$files $DR11/randoms/randoms-south-1-$k.fits"; done
    j_idx=""
    if [[ ! -s $IDX/randoms-south-1-$((NRAND - 1)).json ]]; then
      j_idx=$(submit index "" --partition=$PART_CPU --array=0-$((NRAND - 1)) --cpus-per-task=2 --mem=40G --time=6:00:00 -- \
        "f=($files); python -m rema.cli randoms-index \${f[\$SLURM_ARRAY_TASK_ID]} --outdir $IDX --config $P/catalog_config.yaml")
    fi
    j_map=$(submit maps "$j_idx" --partition=$PART_CPU --cpus-per-task=4 --mem=140G --time=8:00:00 -- \
      "python -m rema.cli maps --index $IDX --nrand $NRAND --config $P/catalog_config.yaml --out $P/footprint_dr11.fits")
    j_cp=$(submit copies "$j_map" --partition=$PART_CPU --cpus-per-task=8 --mem=100G --time=12:00:00 -- \
      "python -m rema.cli cluster-randoms domain --runs $RUNS --seam-buffer $SEAM_BUFFER --out $P/domain.fits
python -m rema.cli cluster-randoms vlim-map --footprint $P/footprint_dr11.fits --domain $P/domain.fits --catalog $CAT --lstar $LSTAR --nsigma $NSIGMA --index $IDX --nrand $NRAND --out $P/zvlim_dr11.fits
python -m rema.cli cluster-randoms copies --clusters $CAT --spec $SPEC --zvlim $P/zvlim_dr11.fits --domain $P/domain.fits --index $IDX --nrand $NRAND --with-id --out $P/copies_parent.fits")
    j_pl=$(submit place "$j_cp" --partition=$PART_GPU --gpus=1 --cpus-per-task=5 --mem=45G --time=12:00:00 -- \
      "export JAX_PLATFORMS=cuda
python -m rema.cli cluster-randoms place --copies $P/copies_parent.fits --footprint $P/footprint_dr11.fits --batch 65536")
    cmds=""
    for s in $SAMPLES; do
      n=$(basename "$s" .yaml)
      cmds="$cmds
python -m rema.cli cluster-randoms sample --clusters $CAT --spec $s --zvlim $P/zvlim_dr11.fits --domain $P/domain.fits --copies $P/copies_parent.fits --keep MEM_MATCH_ID --keep Z_LAMBDA_E --keep R_LAMBDA --keep GLAT --out $P/$n.fits
python -m rema.cli cluster-randoms qa --sample $P/$n.fits --zvlim $P/zvlim_dr11.fits --domain $P/domain.fits --out $P/${n}_qa.json"
    done
    j_s=$(submit sample "$j_pl" --partition=$PART_CPU --cpus-per-task=4 --mem=80G --time=6:00:00 -- "$cmds")
    echo "index $j_idx, maps $j_map, copies $j_cp, place $j_pl, samples $j_s" | tee -a "$LOGS/jobs.txt"
    ;;
  external)   # X-ray / SZ samples (after submit has made the footprint, domain and z_vlim map)
    # $P/selfn holds the selection inputs (act_dr6_rms.fits, spt_2500d_fields.fits, PSZ2 mask,
    # erass1_model.yaml, the exposure and HI maps) and $P/<cat>_joined.fits the joined catalogues
    # (cluster-randoms external-join, erass1-model).
    [[ -f $P/zvlim_dr11.fits ]] || { echo "run submit first"; exit 1; }
    mkdir -p "$P/selfn"
    cp "$CODE/scripts/randoms/spt_sz_2500d_fields.csv" "$P/selfn/"
    cmds=""
    for c in ${EXTERNAL:-act_dr6 spt_2500d psz2 erass1}; do
      spec=$P/selfn/$c.yaml
      sed "s|SELFN_DIR|$P/selfn|g" "$CODE/scripts/randoms/$c.yaml" > "$spec"
      cmds="$cmds
python -m rema.cli cluster-randoms copies --clusters $P/${c}_joined.fits --config $P/catalog_config.yaml --spec $spec --zvlim $P/zvlim_dr11.fits --domain $P/domain.fits --index $IDX --nrand $NRAND --out $P/copies_$c.fits
python -m rema.cli cluster-randoms place --copies $P/copies_$c.fits --footprint $P/footprint_dr11.fits --config $P/catalog_config.yaml
python -m rema.cli cluster-randoms sample --clusters $P/${c}_joined.fits --config $P/catalog_config.yaml --spec $spec --zvlim $P/zvlim_dr11.fits --domain $P/domain.fits --copies $P/copies_$c.fits --keep RA_OPT --keep DEC_OPT --keep INPUT_NAME --out $P/$c.fits
python -m rema.cli cluster-randoms qa --sample $P/$c.fits --zvlim $P/zvlim_dr11.fits --domain $P/domain.fits --out $P/${c}_qa.json"
    done
    j=$(submit external "" --partition=$PART_CPU --cpus-per-task=8 --mem=100G --time=24:00:00 -- "$cmds")
    echo "external $j" | tee -a "$LOGS/jobs.txt"
    ;;
  config)   # the catalogue's configuration, for the randoms cuts (run once before submit)
    export PYTHONPATH=$CODE${PYTHONPATH:+:$PYTHONPATH}
    python -m rema.cli cluster-randoms dump-config --catalog "$CAT" --out "$P/catalog_config.yaml"
    ;;
  status)
    squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %R" | grep crand- || true
    tail -n 3 "$LOGS"/*.log 2>/dev/null | tail -n 40
    du -sh "$WORK" 2>/dev/null
    ;;
  *)
    echo "usage: $0 config|submit|external|status"; exit 1 ;;
esac
