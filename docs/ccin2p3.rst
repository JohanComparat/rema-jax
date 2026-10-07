The DR11 south catalogue at CC-IN2P3
====================================

The blind run of rema 0.2.0 over Legacy Surveys DR11 south is done. Its catalogue is on the
`CC-IN2P3 <https://doc.cc.in2p3.fr>`_ data system, next to the DR11 data, in
``/sps/lsst/datasets/desi/legacysurveys/dr11/south/rema/`` (readable by the ``lsst`` group).
:doc:`redmapper_dr11` shows it in the figures of the redMaPPer papers. This page describes the
products, how they were made, and how to run the pipeline again.

The catalogue
-------------

- **3,481,608 clusters** with λ ≥ 3, over 18,500 deg²: 585,991 with λ ≥ 10, 145,133 with λ ≥ 20,
  11,763 with λ ≥ 50 and 899 with λ ≥ 100. **87.7 M members** (P ≥ 0.01).
- **Footprint:** E(B−V) < 0.2 (cut on the galaxies and the randoms alike) and \|b\| ≥ 15°. Near
  the Galactic plane, stellar contamination makes most seeds survive the first pass and the
  percolation runs out of memory.
- **Calibration:** one red-sequence calibration in g, r, i, z (z-band reference), fitted on RA
  160–180° and 190–210°, Dec −10° to 10° (1.29 M spectroscopic redshifts, 49,524 clusters).
- **Spectroscopy:** the ``Z_SPEC`` of the DR11 photo-z sweeps; ``SPEC_Z_BOOT``, ``VDISP_BOOT``,
  ``CG_SPEC_Z`` and ``BEST_Z`` come from the Clerc et al. (2016) clipping (:doc:`dr11`, step 7).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Directory
     - Content
   * - ``rema_dr11_v0.2.0/``
     - **The catalogue**: ``clusters_dr11.fits`` (2.7 GB), ``clusters_dr11_members.fits``
       (12 GB) and ``clusters_dr11_qa.json``. The two parts below, combined, with ``PART``,
       ``GLON``, ``GLAT``, ``SEAM_DIST`` and ``FLAG_SEAM``; ``MEM_MATCH_ID = PART << 40 | region
       << 32 | rank`` is unique, in the clusters and the members.
   * - ``rema_dr11_v0.2.0_ra0-240/``
     - Part 0: RA 0–240°, Dec −85° to 40°. 191 of 198 regions, 2,453,256 clusters. The merged
       catalogues, ``clusters_dr11_regions.fits`` (per-region statistics), the plan
       ``regions.fits``, ``logs/``, and the calibration ``calib/calib.fits`` with its plots and
       training galaxies (``calib/plots/rs_specz.fits``).
   * - ``rema_dr11_v0.2.0_ra240-360/``
     - Part 1: RA 240–360° and the polar cap below Dec −85°. 79 of 94 regions, 1,028,352
       clusters. The same files, with the same calibration.
   * - ``external/``
     - The public cluster catalogues and the DR10 red-sequence model of Kluge et al. (2024)
       used by :doc:`redmapper_dr11`, with a ``README.txt`` of their sources.
   * - ``notebooks/``
     - What the :ref:`notebooks <notebooks>` wrote when they were executed at CC-IN2P3.

**Seams.** The two parts were run separately. Within 2.33° of their common boundaries (RA 0° and
240° above Dec −85°, and Dec −85° between RA 0° and 240°) the data of a cluster stop at the
boundary, so its λ may be low: ``FLAG_SEAM`` marks these 123,160 clusters (``SEAM_DIST`` in
degrees). In the files of the parts, ``MEM_MATCH_ID`` is unique within a part only.

**Missing regions.** 7 regions of part 0 and 15 of part 1 are missing, all of them reaching
below \|b\| = 15°, where the runs failed and were not repeated. The QA files list them.

Using it
--------

.. code-block:: python

   import numpy as np
   from rema.io.tables import read_table

   R = "/sps/lsst/datasets/desi/legacysurveys/dr11/south/rema/rema_dr11_v0.2.0/"
   cat = read_table(R + "clusters_dr11.fits", hdu="CLUSTERS",
                    columns=["MEM_MATCH_ID", "RA", "DEC", "LAMBDA", "Z_LAMBDA", "BEST_Z", "FLAG_SEAM"])
   rich = (cat["LAMBDA"] >= 20) & ~cat["FLAG_SEAM"]
   mem = read_table(R + "clusters_dr11_members.fits", hdu="MEMBERS", columns=["MEM_MATCH_ID", "P"])

The column names follow redMaPPer (:doc:`reference`). ``read_table`` reads only the columns it
is given, so the 12 GB of members need not be loaded whole. At z_λ > 0.75, keep λ ≥ 10: down to
λ = 5 the catalogue favours completeness (:doc:`design`, section 14). The notebooks run at
CC-IN2P3 on the Jupyter platform: :ref:`cc-notebooks`.

How it was made
---------------

The production ran on 2026-10-05 to 2026-10-07 with ``scripts/slurm/rema_dr11_blind.sh``
(:doc:`hpc`), on the V100 nodes:

1. **Ingest** of the 1,600 sweeps and their photo-z files, and the index of the randoms files 0
   to 3.
2. **Calibration**, one CPU job on the two boxes above (``CALIB_BOX``).
3. **Two partial runs**, because the download of RA 240–360° was not finished when the first one
   started: ``AREA_BOX="0 240 -85 40"`` and ``AREA_BOX="240 360 -90 40;0 240 -90 -85"``, with
   ``GLAT_MIN=15`` (regions entirely at \|b\| < 15° are not run, clusters at \|b\| < 15° are cut
   at the merge) and regions of 100 deg² with 2° buffers.
4. **Failed regions** lying entirely at \|b\| ≥ 15° ran again with more memory, from their
   checkpoints: regions 43 and 53 of part 0 with 90 GB, and region 50 of part 1, at the edge of
   the bulge with 1.06 M candidates, with 150 GB (it peaked at 84 GB).
5. **Merges** with ``ALLOW_MISSING=1``; ``scripts/dr11/finalize_rema.sh`` accepted missing
   regions only if they reach below \|b\| = 15°, copied the products to ``dr11/south/rema/``,
   checked the copies byte for byte and deleted the work directory.
6. **Combination:** ``scripts/dr11/combine_dr11.py`` wrote ``rema_dr11_v0.2.0/`` and checked
   that the IDs are unique and that every member has its cluster.

A region took a median of 4 h and 18 GB of memory (90% of them below 37 GB); the 270 regions
took 990 task-hours. The region tasks batched under the queue limit (``MAX_ARRAY``) ran JAX on
the CPU of their node, because of a bug of the ``batch`` stage (fixed in rema 0.3.1); the
regions run on the GPU took about 1 h. The catalogue does not depend on the device: CPU and GPU
agree to 8·10⁻⁴ in λ (:doc:`install`).

Running it again
----------------

**Once:** the environment on ``/sps``, as in :ref:`cc-notebooks` (step 1, with its check on a
GPU node): the same environment serves the notebooks and the SLURM jobs.

**Each login shell.** The jobs inherit the submitting shell's environment (``rema-env``, from
:ref:`cc-notebooks`, does the first two lines):

.. code-block:: bash

   source /sps/lsst/users/$USER/miniforge3/etc/profile.d/conda.sh
   conda activate /sps/lsst/users/$USER/envs/rema
   export OUTDIR=/sps/lsst/users/$USER/rema_dr11_v0.3.1    # one directory per calibration and version
   export GLAT_MIN=15                                      # leave out the Galactic plane
   source $HOME/software/rema-jax/scripts/slurm/ccin2p3.env
   D=$REMA/scripts/slurm

``ccin2p3.env`` keeps every variable already exported and sets the rest: ``LEGACYSURVEY_DIR`` and
``DR11``, ``NRAND=4`` (randoms files 0 to 3 are on ``/sps``), the partitions, one GPU per region
task with 5 CPUs and 45 GB, ``--licenses=sps`` on every job, ``MAX_ARRAY=90``, the caches on
``/sps``, and ``JAX_PLATFORMS=cpu`` for the commands run on the login node.

**The run:**

.. code-block:: bash

   $D/rema_dr11_blind.sh prepare        # ingest and randoms index (CPU arrays on htc)
   rema regions --galaxies $OUTDIR/galaxies --calib-suggest 400 --config $D/dr11_south.yaml
   CALIB_BOX="160 180 -10 10;190 210 -10 10" $D/rema_dr11_blind.sh prepare   # calibration job
   # check $OUTDIR/calib/plots and the header of $OUTDIR/calib/calib.fits, or reuse a calibration:
   export CALIB=$OUTDIR/calib/calib.fits
   $D/rema_dr11_blind.sh run            # plan, priming region, region array, merge
   $D/rema_dr11_blind.sh status
   $D/rema_dr11_blind.sh clean          # after the merge: galaxy tables, randoms index, JAX cache

- ``prepare`` and ``run`` submit only what is missing or stale; after failures or timeouts, call
  them again in a shell set up as above. They refuse while their earlier jobs are queued or
  running. A new ``CALIB`` makes every region stale.
- Rerun regions with more memory: ``ARRAY=43,53 REGION_MEM=90G $D/rema_dr11_blind.sh run``. Merge
  with failed regions: ``ALLOW_MISSING=1``.
- Without ``CLUSTERS_DIR`` and ``MEMBERS_DIR`` the products go next to the sweeps, in
  ``sweep/11.0-rm/`` and ``sweep/11.0-rm-mem/``, with fixed names that a later run overwrites.
  ``AREA_BOX`` restricts a run to part of the sky (:doc:`hpc`).
- The DR11 data are complete on ``/sps`` (1,600 sweeps and photo-z files, randoms files 0 to 3).
  ``scripts/get_dr11_south_sweeps.sh`` and ``scripts/get_dr11_south_randoms.sh`` download them
  elsewhere (:doc:`dr11`, step 1).

**Cluster rules:**

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Rule
     - Setting (``ccin2p3.env``)
   * - GPUs are requested with ``--gpus``; the job filter rejects ``--gres``
     - ``GPUS=1``
   * - At most 5 CPUs per V100 and 12 per H100. A V100 node has 4 GPUs, 24 CPUs and 191 GB.
     - ``REGION_CPUS=5``, ``REGION_MEM=45G``. H100: ``PART_GPU=gpu_h100 REGION_CPUS=12
       REGION_MEM=96G``
   * - Jobs that read or write ``/sps`` declare it
     - ``--licenses=sps`` on every job
   * - ``htc``: at most 150 GB and 7 days per job
     - calibration and merge: 128 GB
   * - GPU partitions: at most 100 queued jobs per user
     - ``MAX_ARRAY=90``: each array task runs several regions in turn (``$OUTDIR/jobs/batches``)

**Disk space.** ``spsquotalist /sps/lsst/users/$USER`` prints the use of your directory. A full
run needs about 80 GB of galaxy tables, 8 GB of randoms index and 1.4 GB of JAX cache, all deleted
by ``clean``; each region keeps 53 MB (``footprint.fits`` and ``clusters.fits``), and the merged
catalogues take 14 GB. Galaxy tables and a randoms index made with the same configuration can be
moved into a new ``OUTDIR`` before ``clean`` and reused.

Validation at CC-IN2P3
----------------------

- On 2026-10-05, the 3-sweep development area (RA 0–5°, Dec −15–0°) run at CC-IN2P3 on a V100
  matched the laptop: identical galaxies and footprint, 448 against 449 clusters with λ ≥ 20
  (median \|Δλ/λ\| 2×10⁻⁶), the same centres at the 25 ACT clusters, 12,859 against 12,857
  clusters in the three-region run. The test suite passes on a CPU node, and the GPU test on a
  V100.
- On 2026-10-07, the notebooks executed at CC-IN2P3 with the calibration of the production run:
  inside their three sweeps the blind catalogue matches the production one (all 369 clusters with
  λ ≥ 20 have the same centre, median \|Δλ/λ\| 3×10⁻⁶), and the three-region run matches the
  one-region run exactly.
