Production on an HPC
====================

The full DR11 south blind run (1,600 sweeps, Dec −90 to +40) cuts the sky into regions of
about 100 deg² and runs one SLURM array task per region. `scripts/slurm/rema_dr11_blind.sh`
drives it in two phases, with a check of the calibration in between. Each task runs one stage
of `scripts/slurm/rema_task.sh`, which also runs without SLURM. The pipeline notebook
(:doc:`notebooks/pipeline_dr11`) runs the same stages on a few sweeps.

.. list-table::
   :header-rows: 1
   :widths: 14 14 72

   * - Stage
     - Tasks
     - What it does
   * - ingest
     - about 80
     - ``rema ingest SWEEPS --outdir galaxies/ --require-pz``: one compact table per sweep, so
       the 1.8 TB of sweeps are read once. Each table header records NGAL, NZSPEC, the mean
       E(B−V) and SURVHASH, the hash of the selection.
   * - randoms
     - 20
     - ``rema randoms-index``: each 23 GB randoms file is read once, and the needed columns are
       written sorted by HEALPix pixel. A region's footprint then reads only the rows of its
       data box.
   * - calib
     - 1 (CPU)
     - ``rema maps --index`` and ``rema calibrate`` on 300–500 deg² rich in spectroscopy
       (repeatable ``--box``).
   * - (check)
     - —
     - A person checks ``calib/plots/`` and the calibration header, then sets ``CALIB``.
   * - plan
     - inline
     - ``rema regions``: the region plan, from the galaxy counts.
   * - region
     - 1 + about 450
     - ``rema maps --index --regions … --region-id i`` builds the region's footprint, then
       ``rema blind --galaxies galaxies/ --regions … --region-id i --specpost``. One region runs
       first to fill the shared JAX compilation cache.
   * - merge
     - 1
     - ``rema merge``: one catalogue, members in a separate file, and the QA.

Quick start
-----------

.. code-block:: bash

   export LEGACYSURVEY_DIR=/path/to/legacysurvey     # CC-IN2P3: /sps/lsst/datasets/desi/legacysurveys
                                                     # NERSC: /global/cfs/cdirs/cosmo/data/legacysurvey
   # DR11 defaults to $LEGACYSURVEY_DIR/dr11/south; set it to use another copy
   export OUTDIR=$SCRATCH/rema_dr11_v1               # one directory per calibration and rema version
   export DEVICE=gpu PART_GPU=gpu PART_CPU=cpu ACCOUNT=myproject

   scripts/slurm/rema_dr11_blind.sh prepare          # ingest and randoms-index arrays
   rema regions --galaxies $OUTDIR/galaxies --calib-suggest 400 --config scripts/slurm/dr11_south.yaml
   CALIB_BOX="150 170 -5 15" scripts/slurm/rema_dr11_blind.sh prepare    # the calibration job
   # check $OUTDIR/calib/plots and the header of $OUTDIR/calib/calib.fits, then:
   export CALIB=$OUTDIR/calib/calib.fits
   scripts/slurm/rema_dr11_blind.sh run              # plan, priming region, region array, merge
   scripts/slurm/rema_dr11_blind.sh status

The calibration box above is only an example: use one of the boxes that ``--calib-suggest``
proposes.

- ``prepare`` and ``run`` submit only what is missing, or stale: made with another plan,
  calibration or rema version (``rema status``). After failures or timeouts, call them again.
  They refuse to submit while jobs they submitted before are still queued or running.
- To rerun particular regions, for example with more memory:
  ``ARRAY=17,42 REGION_MEM=192G scripts/slurm/rema_dr11_blind.sh run``.
- The products are written to ``CLUSTERS_DIR`` (default ``$OUTDIR``) and ``MEMBERS_DIR``
  (default ``CLUSTERS_DIR``):

  - ``$CLUSTERS_DIR/clusters_dr11.fits``, the clusters;
  - ``$MEMBERS_DIR/clusters_dr11_members.fits``, the members;
  - ``$CLUSTERS_DIR/clusters_dr11_regions.fits``, the per-region statistics;
  - ``$CLUSTERS_DIR/clusters_dr11_qa.json``, the QA.

  ``run`` checks that both directories are writable before it submits anything.

CC-IN2P3
--------

``scripts/slurm/ccin2p3.env`` holds the settings for the `CC-IN2P3 <https://doc.cc.in2p3.fr>`_
cluster. With rema cloned in ``$HOME/software/rema-jax``:

.. code-block:: bash

   mamba activate rema                               # jobs inherit this environment
   source $HOME/software/rema-jax/scripts/slurm/ccin2p3.env
   $REMA/scripts/slurm/rema_dr11_blind.sh prepare

It sets:

- the inputs: ``DR11=/sps/lsst/datasets/desi/legacysurveys/dr11/south`` and ``NRAND=4``
  (randoms files 0 to 3 are there);
- the work directory ``OUTDIR=/sps/lsst/users/$USER/rema_dr11``;
- the products next to ``sweep/11.0`` and ``sweep/11.0-photo-z``: clusters in
  ``sweep/11.0-rm/`` and members in ``sweep/11.0-rm-mem/``;
- ``htc`` for the CPU stages and ``gpu_v100`` for the regions, one GPU per task requested with
  ``--gpus`` (``GPUS=1``; the cluster rejects ``--gres``), with 5 CPUs and 45 GB (``REGION_CPUS``,
  ``REGION_MEM``), so that four tasks share a node of 4 GPUs, 24 CPUs and 191 GB. The cluster
  allows at most 5 CPUs per V100 and 12 per H100: ``PART_GPU=gpu_h100 REGION_CPUS=12
  REGION_MEM=96G`` for the H100 nodes;
- ``--licenses=sps`` on every job, which the cluster requires for jobs that use ``/sps``.

Variables exported before sourcing the file are kept. ``ACCOUNT`` defaults to your main group.
It also moves the pip and matplotlib caches out of ``$HOME``, which is small. The README gives
the install on ``/sps`` and the commands of one full run.

Parallel jobs and disk space
----------------------------

- **No shared files.** Every task writes its own outputs: one galaxy table per sweep, one index
  per randoms file, one directory per region, one log per task. Writes go to a temporary file
  renamed into place.
- **JAX cache.** JAX writes compilation-cache entries in place, without locking. The priming
  region writes the shared cache (``$OUTDIR/jax_cache``) alone; every later region task copies it
  to its job's ``TMPDIR`` and compiles there, so concurrent tasks never read a partial entry.
- **Per region.** A region task deletes its ``checkpoint/`` once ``clusters.fits`` is written; it
  keeps ``footprint.fits`` and ``clusters.fits``.
- **After the merge.** ``scripts/slurm/rema_dr11_blind.sh clean`` deletes the per-sweep galaxy
  tables, the randoms index and the JAX cache. It refuses unless the merged products exist,
  every region is done, and nothing changed after the merge. Rerunning a region afterwards
  needs ``prepare`` again.

The configuration
-----------------

``scripts/slurm/dr11_south.yaml`` holds the defaults plus ``survey.ebv_max: 0.2``. Galaxies and
randoms with E(B−V) ≥ 0.2 are cut alike, so the masked area and MASKFRAC stay consistent with
the galaxies.

- **Where it applies:** ingest, the randoms index, the footprints and the calibration use it.
  The region runs inherit it from the calibration.
- **SURVHASH:** a galaxy table made with another selection is refused, because its SURVHASH
  differs from the run's.

Regions and boundaries
----------------------

**The planner.** ``rema regions`` builds the plan from the per-sweep galaxy counts:

- **Bands:** Dec bands of 10° follow the sweep rows. The polar cap (\|Dec\| > 85°) is one
  full-RA ring.
- **Own boxes:** in each band, own boxes are runs of whole sweeps, starting after the band's
  largest RA gap so that RA 0/360 needs no special case. Each run holds about ``TARGET_AREA``
  deg² (100 by default): 10° × 10° near the equator, wider in RA towards the pole.
- **Caps:** regions above ``MAX_GAL`` galaxies, or ``MAX_PAIRS`` estimated percolation pairs,
  in their data box are split.
- **Order:** regions are numbered by decreasing cost.
- **Size:** for DR11 south this gives about 320–450 regions. The data box of a median region
  holds about 5 M galaxies, and the densest up to about 16 M.

**Clusters across sweeps and regions.**

- Sweeps are only I/O units. A region reads every per-sweep table that overlaps its *data box*
  (its own box grown by ``BUFFER`` = 2°) and cuts it to the box. A cluster spanning two sweeps is
  therefore seen whole.
- A cluster is kept only by the region whose *own box* contains its final centre, so the merged
  catalogue has no duplicates. ``MEM_MATCH_ID = region << 32 | rank`` is unique, and
  ``MEM_MATCH_ID >> 32`` gives the region.
- Members are concatenated as they are. A galaxy can be a member of clusters owned by two
  regions, as in a run over the whole sky.

**What the buffer covers.** A cluster changes directly through its aperture, and through the
claims of higher-ranked neighbours: their mask radius plus their seed offset. At z = 0.05 that
reach is 1.4°, 1.6° and 1.9° for λ = 30, 100 and 300, so the 2° buffer covers it
(:func:`rema.pipeline.required_buffer`).

**What it does not cover.** Percolation's full dependency radius is 1.9°, 2.3° and 2.9°. Chains
of claims that start beyond the buffer can still change a cluster near a boundary, at second
order, as in redMaPPer's tiling. ``rema merge`` reports the cluster density against the
distance to internal edges, and the cross-region close pairs.

**Measured** (pipeline notebook): the 74 deg² strip RA 0–5°, Dec −15–0° was run as three
25 deg² regions and as one region.

- The merged catalogue holds the same 12,857 clusters as the one-region catalogue.
- Every cluster with λ ≥ 5 has the same central galaxy at every distance from the two internal
  boundaries, including within 0.25°.
- λ agrees to a median relative difference of 3·10⁻⁶.
- 15 clusters (0.12%) land on another z_λ solution, with a different λ:

  - all 15 had reached the 5-iteration cap of z_λ (Z_LAMBDA_NITER = 5), as had about 20% of
    all clusters;
  - a different batching of the float32 kernels changes their result;
  - they lie up to 4.8° from any boundary, beyond the buffer's reach, so they are not a
    boundary effect.

With 25 deg² own boxes the regions' blind runs took twice the one-region time. Production regions of
100 deg² read relatively less buffer.

The randoms
-----------

``NRAND`` (default 20) is the number of randoms files whose index the footprints use. The
CC-IN2P3 copy (``/sps/lsst/datasets/desi/legacysurveys``) holds files 0 to 3 only: set
``NRAND=4`` there, or download more files with ``scripts/get_dr11_south_randoms.sh`` into a
copy you can write to and point ``DR11`` at it.

- One file gives about 8 randoms per nside-1024 pixel. The aperture of a λ = 20 cluster covers
  4 pixels at z = 0.3 and 1.4 pixels at z = 0.8.
- The index is built once (about 3 GB per file on disk), so more files cost little time.
- Use all 20 in production, unless scratch space is short.

Devices and resources
---------------------

``DEVICE=gpu``
    Region tasks request ``GPU_GRES`` (``gpu:1``), or ``GPUS`` GPUs with ``--gpus`` when it is
    set, ``PART_GPU`` and ``GPU_CONSTRAINT`` (pin the GPU model), with 8 CPUs, 96 GB and 8 h,
    and run JAX on the GPU. Blind mode is 4–5 times faster on a laptop GPU than on 16 CPU
    threads.
``DEVICE=cpu``
    Region tasks use ``PART_CPU``, with 16 CPUs, 64 GB and 24 h.

Override any of these with ``REGION_CPUS``, ``REGION_MEM`` and ``REGION_TIME``.

A region process peaked at 4–5 GB of memory for 1.4–2.1 M galaxies in its data box on the GPU,
about 2.4 kB per galaxy. Check the pilot's ``PEAKRSS`` before relying on that for the densest
regions.

- **Other stages:** ingest, randoms, calib and merge always run on CPUs.
- **Throttles:** ``GPAR``, ``RPAR`` and ``BPAR`` limit the concurrent tasks of the arrays.
- **Extra options:** ``EXTRA_SBATCH`` adds sbatch options to every job.
- **JAX cache:** each device kind has its own ``$OUTDIR/jax_cache/$DEVICE``. The priming task
  fills it before the array starts, because JAX writes cache entries without a lock.

Before the full run
-------------------

Run a pilot of about 10 regions with ``ARRAY=…``. Include:

- the most expensive (region 0);
- the polar ring;
- a region crossing RA 0;
- a median one;
- one on a survey edge.

From their headers (NGAL, NCAND, NDEP, TTOTAL, PEAKRSS), check time and memory against the
galaxy and pair counts. Set ``MAX_GAL`` or ``MAX_PAIRS`` if needed. Then delete the plan and
run. The plan is fixed once the run starts.

To run only part of the sky, for a test or while the download is incomplete, export
``AREA_BOX="RA0 RA1 DEC0 DEC1[;...]"`` before ``prepare`` and keep it for ``run``. Every stage,
and the driver's bookkeeping, then use only the sweeps overlapping that area. Use a separate
``OUTDIR`` and ``CLUSTERS_DIR``, because a plan made for one area is not extended to another.

Cost
----

Rough figures for DR11 south:

.. list-table::
   :header-rows: 1
   :widths: 14 12 44 30

   * - Stage
     - Jobs
     - Compute
     - Read / write
   * - ingest
     - about 80
     - about 80 core-h
     - 2.2 TB / 120 GB
   * - randoms
     - 20
     - about 10 core-h, 16–24 GB each
     - 0.46 TB / 60 GB
   * - calib
     - 1
     - 2–4 h on 32 cores, 64–128 GB
     - —
   * - region
     - 320–450
     - 150–250 GPU-h (laptop RTX 3060 class); about 1,000 task-h on 16 CPU cores
     - about 1 TB / 60 GB
   * - merge
     - 1
     - about 1 h
     - — / about 15 GB (about 4 M clusters, 90 M member rows)

The GPU and CPU figures are extrapolated from the 75 deg² development area, with the buffers
doubling the processed area.
