DR11 blind run at CC-IN2P3
==========================

Step-by-step commands for the DR11 south blind run on the `CC-IN2P3 <https://doc.cc.in2p3.fr>`_
SLURM cluster: install, data, a run on the part of the sky already downloaded, and the full
sky. The driver and its stages are described in :doc:`hpc`; this page adds the cluster's
paths, rules and checks.

- **Inputs:** ``/sps/lsst/datasets/desi/legacysurveys/dr11/south``: ``sweep/11.0`` (sweeps),
  ``sweep/11.0-photo-z`` (row-matched photo-z files, with ``Z_SPEC``) and ``randoms/``
  (files 0 to 3).
- **Work directory:** ``/sps/lsst/users/$USER/...``, one per calibration and rema version.
  ``$HOME`` is small: the Python environment and the caches also live on ``/sps``.
- **Products:** ``clusters_dr11.fits``, ``clusters_dr11_members.fits``,
  ``clusters_dr11_regions.fits`` and ``clusters_dr11_qa.json``. The full-sky run writes them next
  to the sweeps, in ``sweep/11.0-rm/`` and ``sweep/11.0-rm-mem/``; a partial run keeps them in its
  work directory.

Products of the DR11 south run
------------------------------

The DR11 south run of rema 0.2.0 is on the data system, next to the sweeps, in
``/sps/lsst/datasets/desi/legacysurveys/dr11/south/rema/``:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Directory
     - Content
   * - ``rema_dr11_v0.2.0_ra0-240/``
     - RA 0–240°, Dec −85° to 40° (1026 sweeps, 198 regions): ``clusters_dr11.fits``,
       ``clusters_dr11_members.fits``, ``clusters_dr11_regions.fits``, ``clusters_dr11_qa.json``,
       the plan ``regions.fits``, ``logs/`` and the calibration ``calib/calib.fits`` with its
       plots
   * - ``rema_dr11_v0.2.0_ra240-360/``
     - RA 240–360° and the polar cap below Dec −85° (574 sweeps, 94 regions), the same files,
       with the same calibration
   * - ``notebooks/``
     - what the DR11 notebooks write when run at CC-IN2P3

The calibration was fitted on RA 160–180° and 190–210°, Dec −10° to 10° (1.29 M
spectroscopic redshifts, 49,524 clusters). The two parts were run separately: within 2.33° of
their common boundaries (RA 0° and 240° above Dec −85°, and Dec −85° between RA 0° and 240°)
a cluster's data stop at the boundary, so its λ may be low. ``MEM_MATCH_ID`` is unique within a
part only.

The catalogues cover E(B−V) < 0.2 (cut on the galaxies and the randoms) and \|b\| ≥ 15°: the merge
keeps the clusters at \|b\| ≥ 15° (header ``GLATMIN``, QA ``n_clusters_low_glat``), regions lying
entirely at \|b\| < 15° were not run, and regions reaching below 15° that failed were not rerun
(the QA lists them as missing). Near the Galactic plane stellar contamination makes most
seeds survive the first pass, and percolation runs out of memory.

The :ref:`notebooks <notebooks>` read these files by default at CC-IN2P3: the calibration of
the run, and the merged catalogues to compare with. :doc:`redmapper_dr11` shows the catalogue in
the figures of the redMaPPer papers.

Install, once
-------------

.. code-block:: bash

   ssh cca.in2p3.fr
   SPS=/sps/lsst/users/$USER
   git clone https://github.com/JohanComparat/rema-jax.git $HOME/software/rema-jax
   # Miniforge and the environment on /sps (6 GB with the CUDA libraries)
   curl -L -o /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash /tmp/mf.sh -b -p $SPS/miniforge3 && rm /tmp/mf.sh
   source $SPS/miniforge3/etc/profile.d/conda.sh
   conda create -y -p $SPS/envs/rema python=3.12 pip
   conda activate $SPS/envs/rema
   pip install --no-cache-dir -e "$HOME/software/rema-jax[cuda,plots]"   # JAX from the PyPI wheels

- An existing Miniforge (for example in ``$HOME``) works as well; only the environment needs
  to be on ``/sps``.
- ``[cuda]`` installs the CUDA 12 JAX wheels. CUDA 13 does not support the V100 GPUs.
- To update: ``git -C $HOME/software/rema-jax pull`` (the install is editable). Check the
  version with ``python -c "import rema; print(rema.__version__)"``.

Check that a GPU node sees its GPU (one short job):

.. code-block:: bash

   srun -p gpu_v100 --gpus 1 -t 0-00:10 -c 4 --mem 8G -L sps \
        python -c "import jax; print(jax.devices())"          # [CudaDevice(id=0)]

.. _cc-notebooks:

Jupyter notebooks (notebook.cc.in2p3.fr)
----------------------------------------

The `notebook platform <https://notebook.cc.in2p3.fr>`_ of CC-IN2P3 runs a JupyterLab session as
an interactive SLURM job, on a CPU or a GPU node (`platform documentation
<https://doc.cc.in2p3.fr/fr/Computing/jnps/jn-platform.html>`_). Log in with your CC-IN2P3
account or eduGAIN. A session sees ``$HOME``, ``/pbs/throng``, ``/pbs/software`` and the ``/sps``
spaces of your groups; the DR11 data and the products of this page are in ``/sps/lsst``, readable
by the ``lsst`` group. The :ref:`notebooks <notebooks>` of this documentation run there in a
``rema`` kernel: the environment of `Install, once`_, registered once as a Jupyter kernel.

**Once: the kernel.** In a terminal (``ssh cca.in2p3.fr``, or File → New → Terminal in a session),
after `Install, once`_:

.. code-block:: bash

   SPS=/sps/lsst/users/$USER
   source $SPS/miniforge3/etc/profile.d/conda.sh
   conda activate $SPS/envs/rema
   pip install --no-cache-dir ipykernel pyzmq colossus pandas
   mkdir -p $SPS/rema_notebooks
   python -m ipykernel install --user --name rema --display-name rema \
       --env LEGACYSURVEY_DIR /sps/lsst/datasets/desi/legacysurveys \
       --env REMA_WORK $SPS/rema_notebooks \
       --env JAX_COMPILATION_CACHE_DIR $SPS/.cache/rema/jax \
       --env XLA_PYTHON_CLIENT_PREALLOCATE false
   ln -s $SPS $HOME/sps                     # optional: /sps in the file browser of the platform

- The platform starts a kernel through ``ipykernel`` and ``pyzmq``; ``colossus`` and ``pandas``
  are used only by :doc:`notebooks/redmapper_dr11`.
- ``ipykernel install --user`` writes ``~/.local/share/jupyter/kernels/rema/kernel.json``, with
  the Python of the environment and the ``--env`` variables, so the kernel needs no login script.
  ``REMA_WORK`` is where the notebooks write their products: the default, next to the production
  run, is writable by its owner only. ``JAX_COMPILATION_CACHE_DIR`` keeps the compiled programs on
  ``/sps`` between sessions (``$HOME`` is small).
- To change a variable, run the ``ipykernel install`` line again (it replaces the kernel).
  ``jupyter kernelspec list`` lists the kernels; ``jupyter kernelspec remove rema`` removes this one.
- Without access to ``/sps/lsst``: install the environment in a ``/sps`` space of your group, copy
  the products you need there, and give their location with ``--env LEGACYSURVEY_DIR`` (the DR11
  tree) or ``--env REMA_PRODUCTS`` (the production runs only).
- Keep the ``[cuda]`` extra of `Install, once`_ (CUDA 12 JAX wheels): the platform documentation
  installs ``jax[cuda13]``, which does not support the V100 GPUs.
- The rema of the PyPI release may be older than the notebooks of the repository: install the
  clone in editable mode, as above, and update both with ``git -C $HOME/software/rema-jax pull``,
  then restart the kernel.

**Each session.** On https://notebook.cc.in2p3.fr, choose the computing group (``lsst``), the
partition and the resources, then open the notebook from
``software/rema-jax/docs/notebooks/`` and select the ``rema`` kernel (Kernel → Change Kernel):

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - Session
   * - ``blind_dr11``, ``pipeline_dr11``, ``scan_dr11``
     - GPU partition (V100 or H100), 1 GPU, 5 CPUs, 32 GB. The three run in under 2 hours on a
       V100 and need about 23 GB of memory.
   * - ``redmapper_dr11``
     - CPU partition, 4 CPUs, 32 GB. No GPU. The first run reduces the catalogues and one 23 GB
       randoms file (about 15 minutes per part of the run); later runs reuse them and take a few
       minutes.

The notebooks find the sweeps, the randoms, the calibration and the production catalogues on
``/sps`` by themselves. In a GPU session, ``import jax; jax.devices()`` returns
``[CudaDevice(id=0)]``. The public cluster catalogues of the comparison figures of
:doc:`redmapper_dr11` are next to the products, in ``$REMA_PRODUCTS/external`` (with a
``README.txt`` of their sources), where :doc:`notebooks/redmapper_dr11` finds them, and the
spectroscopic training galaxies of the calibration are in ``calib/plots/rs_specz.fits`` of the run.
The DR10 red-sequence model of Kluge et al. (2024) is not public: its curves are left out of the
red-sequence figures.

If something goes wrong:

- **The kernel does not start:** ``ipykernel`` or ``pyzmq`` is missing from the environment, or
  ``kernel.json`` points to another Python. The session log is
  ``~/.jupyterhub/notebook_server_<job id>.log``.
- **The session stops in the middle of a cell:** it went over its memory or time; start a new
  session with more memory.
- **JAX sees only the CPU:** the session was started on a CPU partition, or ``JAX_PLATFORMS=cpu``
  is set; ``nvidia-smi`` in a terminal of the session shows the GPU.
- **A write fails** (``cannot write to ...``): ``REMA_WORK`` is not set in the kernel.
- End a session with File → Log Out: closing the tab leaves the job running until its time limit.

The data
--------

DR11 south is 1600 sweeps (1.8 TB) and 1600 photo-z files (0.35 TB). The randoms files 0 to 3
are on ``/sps``. Two jobs fill a copy you can write to, the sweeps and the photo-z files in
parallel. Both resume and check sha256 sums, so after a timeout submit the same job again:

.. code-block:: bash

   L=/sps/lsst/users/$USER/logs; mkdir -p $L
   G=$HOME/software/rema-jax/scripts/get_dr11_south_sweeps.sh
   DEST=/sps/lsst/datasets/desi/legacysurveys/dr11/south/sweep
   SUBDIRS=11.0 sbatch --export=ALL -p htc -t 7-00:00:00 -c 4 --mem 4G -L sps -J dr11_sweeps \
       -o $L/dr11_sweeps_%j.log --wrap "$G $DEST 4"
   SUBDIRS=11.0-photo-z sbatch --export=ALL -p htc -t 7-00:00:00 -c 4 --mem 4G -L sps -J dr11_pz \
       -o $L/dr11_pz_%j.log --wrap "$G $DEST 4"

The ingest needs both files of a sweep. This lists, per 30° of RA, how many sweeps are ready:

.. code-block:: bash

   LS=/sps/lsst/datasets/desi/legacysurveys/dr11/south/sweep
   ra30() { awk -F- '{n[int(substr($2,1,3)/30)*30]++} END {for (r in n) print r, n[r]}' | sort -n; }
   join -a1 -e0 -o 0,1.2,2.2 \
     <(awk '$2 ~ /fits$/ {print $2}' $LS/11.0/legacysurvey_dr11_south_sweep_11.0.sha256sum | ra30) \
     <(comm -12 <(sort $LS/11.0/.verified) <(sed 's/-pz\.fits$/.fits/' $LS/11.0-photo-z/.verified | sort) | ra30) \
     | awk '{printf "RA %3d-%3d: %3d of %3d sweeps ready%s\n", $1, $1+30, $3, $2, ($3==$2 ? "" : "  (incomplete)")}'

.. code-block:: text

   RA   0- 30: 153 of 153 sweeps ready
   ...
   RA 210-240: 133 of 133 sweeps ready
   RA 240-270:  29 of 126 sweeps ready  (incomplete)
   RA 270-300:   0 of 113 sweeps ready  (incomplete)

Each login shell
----------------

The jobs inherit the submitting shell's environment, so set it up the same way before every
``prepare``, ``run``, ``status`` or ``clean``:

.. code-block:: bash

   source /sps/lsst/users/$USER/miniforge3/etc/profile.d/conda.sh
   conda activate /sps/lsst/users/$USER/envs/rema
   export OUTDIR=/sps/lsst/users/$USER/rema_dr11_v0.2.0_ra0-240
   export AREA_BOX="0 240 -85 40"                   # a partial run; omit for the full sky
   export CLUSTERS_DIR=$OUTDIR MEMBERS_DIR=$OUTDIR  # a partial run; omit for the full sky
   source $HOME/software/rema-jax/scripts/slurm/ccin2p3.env
   D=$REMA/scripts/slurm

``ccin2p3.env`` keeps every variable already exported and sets the rest: ``LEGACYSURVEY_DIR``
and ``DR11``, ``NRAND=4``, the partitions, the GPU request, ``--licenses=sps`` on every job,
the caches on ``/sps``, and ``JAX_PLATFORMS=cpu`` for commands run on the login node.

A run on part of the sky
------------------------

``AREA_BOX`` restricts every stage, and the driver's bookkeeping, to the sweeps that overlap it.
Choose it among the RA ranges that are ready. The example leaves out the polar cap
(Dec < −85°), where the planner makes one ring over all RA.

**1. Ingest and randoms index** (CPU arrays on ``htc``):

.. code-block:: bash

   $D/rema_dr11_blind.sh prepare
   $D/rema_dr11_blind.sh status

.. code-block:: text

   prepare: galaxy tables 1010/1026 sweeps, randoms indexes 4/4; calibration: none (rerun 'prepare' with CALIB_BOX)
   no region plan yet (/sps/lsst/users/<user>/rema_dr11_v0.2.0_ra0-240/regions.fits): 'run' makes it
   60724625_52 rema-ingest RUNNING 3:05
   60724625_51 rema-ingest RUNNING 3:09

The area holds 1026 sweeps: the 1074 of RA 0–240° less the 48 of the polar cap.

**2. Calibration.** When the ingest is done, list areas of about 400 deg² rich in
spectroscopy, then calibrate on one (one CPU job, a few hours):

.. code-block:: bash

   rema regions --galaxies $OUTDIR/galaxies --calib-suggest 400 --config $D/dr11_south.yaml
   CALIB_BOX="RA0 RA1 DEC0 DEC1" $D/rema_dr11_blind.sh prepare

Check ``$OUTDIR/calib/plots/`` and the header of ``$OUTDIR/calib/calib.fits`` (:doc:`dr11`,
step 4). To go on without waiting, step 3 can be submitted while the calibration job is still
queued or running: with ``CALIB=$OUTDIR/calib/calib.fits`` the regions then start once it
succeeds (and never if it fails), with an unchecked calibration.

**3. Regions and merge:**

.. code-block:: bash

   export CALIB=$OUTDIR/calib/calib.fits
   $D/rema_dr11_blind.sh run
   $D/rema_dr11_blind.sh status

The first ``run`` writes the region plan (RA 0–240°: 198 regions of about 100 deg²). One region
runs alone to fill the JAX cache; the others follow on the V100s, two per array task to stay
under the queue limit, then the merge. Check region 0's ``PEAKRSS`` and
``TTOTAL`` (its catalogue header) before the others run (:doc:`hpc`, "Before the full run").

**4. Clean** once the merge is done. It deletes the galaxy tables, the randoms index and the
JAX cache, and keeps the plan, the calibration, the logs and the region catalogues:

.. code-block:: bash

   $D/rema_dr11_blind.sh clean

The full sky
------------

Once both downloads report 1600 verified files:

.. code-block:: bash

   wc -l $DR11/sweep/11.0/.verified $DR11/sweep/11.0-photo-z/.verified

set up the shell without ``AREA_BOX``, ``CLUSTERS_DIR`` and ``MEMBERS_DIR``, in a new
``OUTDIR`` (for example ``rema_dr11_v0.2.0``), and run the same four steps. A calibration made
on a partial run can be reused with ``export CALIB=...``. The products then go to
``sweep/11.0-rm/`` and ``sweep/11.0-rm-mem/``; their names are fixed, so a later run overwrites
them.

Cluster rules
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Rule
     - Setting
   * - GPUs are requested with ``--gpus``; the job filter rejects ``--gres``
     - ``GPUS=1`` (``ccin2p3.env``)
   * - At most 5 CPUs per V100 and 12 per H100. A V100 node has 4 GPUs, 24 CPUs and 191 GB.
     - ``REGION_CPUS=5``, ``REGION_MEM=45G``, so four region tasks share a node. For the
       H100 nodes: ``PART_GPU=gpu_h100 REGION_CPUS=12 REGION_MEM=96G``.
   * - Jobs that read or write ``/sps`` declare it
     - ``--licenses=sps`` on every job (``EXTRA_SBATCH``)
   * - ``htc``: at most 150 GB and 7 days per job
     - The calibration asks for 128 GB and the merge for 128 GB
   * - GPU partitions: at most 100 queued jobs per user, an array task counting as one
     - ``MAX_ARRAY=90``: with more regions, each task of the region array runs several in turn
       (``$OUTDIR/jobs/batches``, logs ``batch_<task>.log``), with a time limit scaled to match;
       ``BPAR=50`` tasks at a time

A region of the 75 deg² test area peaked at 4–4.4 GB on a V100 (about 2.4 kB per galaxy of its
data box); 45 GB leaves room for the densest regions.

Disk space
----------

``/sps`` quotas are per group, but each user directory has its own quota:
``spsquotalist /sps/lsst/users/$USER`` prints its use and limit (updated every 30 minutes;
``du`` is exact but slow). The work directory of the RA 0–240° run (1026 sweeps, 198 regions)
measured:

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Part
     - Size
     - Kept by ``clean``
   * - ``galaxies/``, one table per sweep
     - 50 GB (49 MB per sweep)
     - no
   * - ``randoms_index/``, one per randoms file
     - 8.1 GB (2 GB per file)
     - no
   * - ``jax_cache/``
     - 1.4 GB
     - no
   * - ``regions/NNNN/checkpoint``, while a region runs
     - about 100 MB each
     - deleted when the region is written
   * - ``regions/NNNN/{clusters,footprint}.fits``
     - 53 MB per region (10.5 GB)
     - yes
   * - merged catalogues
     - about 10–15 GB
     - yes

The whole sky takes about 80 GB of galaxy tables. A run of another area or the full sky with
the same configuration can reuse the galaxy tables and the randoms index: move ``galaxies/``
and ``randoms_index/`` into the new ``OUTDIR`` before ``clean`` deletes them, and ``prepare``
then ingests only the missing sweeps (tables made with another survey configuration are
detected by their ``SURVHASH`` and ingested again).

Monitoring and reruns
---------------------

- ``status`` prints the ingest, randoms and calibration progress before the region plan exists,
  the regions done, missing and stale afterwards, and the driver's queued jobs (also
  ``squeue -u $USER``). Logs are in ``$OUTDIR/logs``.
- ``prepare`` and ``run`` submit only what is missing or stale. After failures or timeouts,
  call them again in a shell set up as above, with the same ``OUTDIR``, ``AREA_BOX`` and
  ``CALIB``. They refuse to submit while their earlier jobs are still queued or running.
- A new ``CALIB`` makes every region stale: ``run`` then runs them all again.
- To rerun particular regions: ``ARRAY=17,42 $D/rema_dr11_blind.sh run``. To merge while some
  regions failed: ``ALLOW_MISSING=1``.
- A CUDA error printed by a command run on the login node means ``JAX_PLATFORMS`` is not
  ``cpu``: source ``ccin2p3.env``.

Validation on CC-IN2P3
----------------------

On 2026-10-05 the 3-sweep development area (RA 0–5°, Dec −15–0°) was run at CC-IN2P3 with
rema 0.1.0 on a V100, and compared with the same runs on a laptop GPU (pre-release code):

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Step
     - Laptop
     - CC-IN2P3
   * - Ingest
     - 3,476,500 galaxies
     - identical (REFMAG to the last float32 bit)
   * - Footprint
     - 22,905 pixels, 70.9 deg²
     - identical
   * - Calibration
     - 2,763 clusters, z_λ NMAD 0.0100
     - 2,755 clusters, NMAD 0.0101
   * - Blind mode, λ ≥ 20
     - 449 clusters
     - 448; median \|Δλ/λ\| 2×10⁻⁶
   * - Scan at 25 ACT clusters
     - 25 clusters, 2,331 members
     - same centres; max \|Δλ/λ\| 2×10⁻³
   * - Three regions and merge
     - 12,857 clusters
     - 12,859; the same QA

The test suite passes on a CC CPU node, and the GPU test on a V100.
