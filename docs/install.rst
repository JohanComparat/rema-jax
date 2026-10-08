Installation
============

From PyPI (the distribution is ``rema-jax``; the package is imported as ``rema`` and installs the
``rema`` command):

.. code-block:: bash

   pip install rema-jax                 # CPU
   pip install "rema-jax[cuda]"         # NVIDIA GPU, CUDA 12 JAX wheels ([cuda13] for CUDA 13)
   pip install "rema-jax[plots]"        # matplotlib, for the calibration plots

rema needs Python 3.11 or later. The SLURM scripts of :doc:`hpc` and the notebooks are in the
repository, not in the package.

From the repository, for development:

.. note::

   **Maintainer setup.** On the development laptop, use the shared ``dev`` environment defined in ``~/software/dev_env`` (``conda activate dev``); this package is already installed there in editable mode. Do not create a separate environment for it: add missing dependencies to ``~/software/dev_env`` and rebuild.

.. code-block:: bash

   git clone https://github.com/JohanComparat/rema-jax.git
   cd rema-jax
   mamba env create -f environment.yml        # Python 3.12, jax[cuda12], optax, ggah_mod
   mamba activate rema
   pip install -e ".[dev]"
   python -m ipykernel install --user --name rema   # Jupyter kernel used by the notebooks

.. warning::

   Install JAX from PyPI (``pip``), as ``environment.yml`` does. The conda-forge build
   ``jaxlib 0.10.2`` for the CPU (``cpu_py312h02fec33_1``) computes wrong richnesses, different
   at every call; the PyPI wheels of the same version are correct. ``pytest
   tests/test_jax_build.py`` checks the installed build in a few seconds.

On a machine without a GPU, replace ``jax[cuda12]`` by ``jax`` in ``environment.yml``. There is
nothing to compile, and no Spark, Java, GSL, esutil or healsparse.

.. _cc-notebooks:

The notebooks at CC-IN2P3 (notebook.cc.in2p3.fr)
------------------------------------------------

The :ref:`notebooks <notebooks>` run on the `notebook platform <https://notebook.cc.in2p3.fr>`_
of CC-IN2P3, next to the DR11 data and the products of the DR11 south run (:doc:`ccin2p3`). The
platform runs a JupyterLab session as an interactive SLURM job, on a CPU or a GPU node
(`platform documentation <https://doc.cc.in2p3.fr/fr/Computing/jnps/jn-platform.html>`_). Log in
with your CC-IN2P3 account or eduGAIN. A session sees ``$HOME``, ``/pbs/throng``,
``/pbs/software`` and the ``/sps`` spaces of your groups; the DR11 data and products are in
``/sps/lsst``, readable by the ``lsst`` group.

**1. Once: the environment and the kernel.** In a terminal (``ssh cca.in2p3.fr``, or File → New →
Terminal in a session). ``$HOME`` is small (20 GB), so Miniforge, its package cache, the
environment (6 GB with the CUDA libraries) and pip's temporary files all go to ``/sps``:

.. code-block:: bash

   export SPS=/sps/lsst/users/$USER
   git clone https://github.com/JohanComparat/rema-jax.git $HOME/software/rema-jax
   # Miniforge on /sps (0.6 GB); its package cache is then on /sps too
   curl -sSL -o $SPS/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash $SPS/Miniforge3.sh -b -p $SPS/miniforge3 && rm $SPS/Miniforge3.sh
   source $SPS/miniforge3/etc/profile.d/conda.sh     # defines the conda command in this shell
   conda create -y -p $SPS/envs/rema python=3.12 pip
   conda activate $SPS/envs/rema
   mkdir -p $SPS/tmp
   TMPDIR=$SPS/tmp pip install --no-cache-dir -e "$HOME/software/rema-jax[cuda,plots,dev,docs]" \
       ipykernel pyzmq colossus pandas
   rmdir $SPS/tmp
   python -m ipykernel install --user --name rema --display-name rema \
       --env JAX_COMPILATION_CACHE_DIR $SPS/.cache/rema/jax \
       --env XLA_PYTHON_CLIENT_PREALLOCATE false
   ln -s $SPS $HOME/sps                     # optional: /sps in the file browser of the platform

``[cuda]`` installs the CUDA 12 JAX wheels (CUDA 13 does not support the V100 GPUs); ``dev`` and
``docs`` add pytest and nbclient, to run the tests and execute the notebooks.

A login shell at CC-IN2P3 (``ssh``) reads ``~/.profile``, not ``~/.bashrc``, so a ``conda init``
in ``~/.bashrc`` does not define ``conda`` there. Add these two lines to both files; ``rema-env``
then activates the environment in any shell:

.. code-block:: bash

   export SPS=/sps/lsst/users/$USER
   alias rema-env='source $SPS/miniforge3/etc/profile.d/conda.sh && conda activate $SPS/envs/rema'

Check the installation with a short job on a V100: JAX must list a CUDA device, and
``tests/test_jax_build.py`` detects a faulty JAX build. ``sbatch --wait`` returns when the job
ends (on 2026-10-07 interactive ``srun`` steps failed to launch on the GPU nodes, while batch
jobs ran):

.. code-block:: bash

   rema-env
   python -c "import rema, jax; print(rema.__version__, jax.__version__)"
   sbatch --wait -p gpu_v100 --gpus 1 -t 0-00:10 -c 4 --mem 8G -L sps -o $SPS/env_check.log --wrap \
       'python -c "import jax; print(jax.devices())"; cd $HOME/software/rema-jax && JAX_PLATFORMS=cpu python -m pytest -q tests/test_jax_build.py'
   cat $SPS/env_check.log          # [CudaDevice(id=0)] ... 1 passed

**When it fails:**

- ``conda: command not found``: run ``source $SPS/miniforge3/etc/profile.d/conda.sh`` (or
  ``rema-env``) in that shell; ``$SPS`` must be set.
- ``CondaValueError: prefix already exists``: an older environment is there; remove it with
  ``rm -rf $SPS/envs/rema`` before ``conda create``.
- ``Disk quota exceeded``: a cache went to ``$HOME``; keep Miniforge on ``/sps`` and pip's
  ``--no-cache-dir`` and ``TMPDIR`` as above.

**2. Each session.** On https://notebook.cc.in2p3.fr, choose the computing group (``lsst``), the
partition and the resources below, open a notebook from ``software/rema-jax/docs/notebooks/``
and select the ``rema`` kernel (Kernel → Change Kernel):

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - Session
   * - ``blind_dr11``, ``pipeline_dr11``, ``scan_dr11``
     - GPU partition (V100 or H100), 1 GPU, 5 CPUs, 32 GB. The three run in under 2 hours on a
       V100 and need about 23 GB of memory.
   * - ``redmapper_dr11``
     - CPU partition, 4 CPUs, 32 GB, no GPU. The first run reduces the two parts of the
       production catalogue and one 23 GB randoms file (about 30 minutes); later runs reuse them
       and take a few minutes.

The notebooks use two directories, given by two environment variables, and need neither at
CC-IN2P3:

- ``REMA_DATA``, shared and read only: the DR11 south directory, with the sweeps, the randoms
  and, in ``rema/``, the production catalogues, their calibration and the external catalogues of
  :doc:`redmapper_dr11`. Default: ``/sps/lsst/datasets/desi/legacysurveys/dr11/south`` when it
  exists, else ``~/data/legacysurvey/dr11/south``.
- ``REMA_WORK``, yours: everything the notebooks write. Default: ``~/rema_work``. The DR11
  notebooks write about 0.5 GB there, :doc:`notebooks/redmapper_dr11` 7.5 GB; when ``$HOME``
  lacks the room (20 GB at CC-IN2P3), add ``--env REMA_WORK $SPS/rema_work`` to the
  ``ipykernel install`` line.

In a GPU session, ``import jax; jax.devices()`` returns ``[CudaDevice(id=0)]``.

**3. Updates.** ``git -C $HOME/software/rema-jax pull`` updates the code and the notebooks
together (the install is editable); then restart the kernel.

What the kernel settings do:

- ``ipykernel install --user`` writes ``~/.local/share/jupyter/kernels/rema/kernel.json``, with
  the Python of the environment and the ``--env`` variables, so the kernel needs no login script.
  The platform starts kernels through ``ipykernel`` and ``pyzmq``; ``colossus`` and ``pandas``
  are used only by :doc:`notebooks/redmapper_dr11`.
- ``JAX_COMPILATION_CACHE_DIR`` keeps the compiled programs on ``/sps`` between sessions.
- To change a variable, run the ``ipykernel install`` line again (it replaces the kernel).
  ``jupyter kernelspec list`` lists the kernels; ``jupyter kernelspec remove rema`` removes this one.
- Keep the ``[cuda]`` extra (CUDA 12 JAX wheels): the platform documentation installs
  ``jax[cuda13]``, which does not support the V100 GPUs.
- Install the clone, not the PyPI release: the notebooks of the repository may need a newer
  rema than the last release.
- Without access to ``/sps/lsst``: install the environment in a ``/sps`` space of your group, copy
  the parts of the DR11 south directory you need there, with the same layout, and give their
  location with ``--env REMA_DATA``.

If something goes wrong:

- **The kernel does not start:** ``ipykernel`` or ``pyzmq`` is missing from the environment, or
  ``kernel.json`` points to another Python. The session log is
  ``~/.jupyterhub/notebook_server_<job id>.log``.
- **The session stops in the middle of a cell:** it went over its memory or time; start a new
  session with more memory.
- **JAX sees only the CPU:** the session was started on a CPU partition, or ``JAX_PLATFORMS=cpu``
  is set; ``nvidia-smi`` in a terminal of the session shows the GPU.
- **A write fails** (``cannot write to REMA_WORK``, or ``Permission denied``): ``REMA_WORK``
  points to a directory that is not yours; ``REMA_DATA`` is never written to.
- End a session with File → Log Out: closing the tab leaves the job running until its time limit.

The batch production at CC-IN2P3 (SLURM arrays) uses the same environment: :doc:`ccin2p3`.

Devices and JAX settings
------------------------

``JAX_PLATFORMS``
    ``cuda`` requires the GPU and ``cpu`` forces the CPU. When it is unset, JAX uses the GPU if
    the CUDA wheels find one.

``XLA_PYTHON_CLIENT_PREALLOCATE=false``
    The ``rema`` command sets it, so that JAX does not reserve 75% of the GPU memory at start.
    Set it yourself in Python, before importing JAX.

``JAX_COMPILATION_CACHE_DIR``
    The ``rema`` command keeps compiled programs here (default ``~/.cache/rema/jax``). A run
    compiles a few dozen programs, each in a few seconds (z_λ takes 7–8 s per shape on a GPU).
    With a warm cache, a run starts computing immediately.

In Python, do the same before the first JAX computation:

.. code-block:: python

   import os
   os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
   import jax
   jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/rema/jax"))
   jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

The cluster kernels run in float32. CPU and GPU catalogues agree: on the 75 deg² development
area, all 330 clusters with λ ≥ 20 have the same central galaxy on both, and their λ differ by
at most 8·10⁻⁴ (relative).

Which device for what, measured on a laptop (i9-11900H with 16 threads, RTX 3060 Laptop GPU with
6 GB):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Stage
     - Device
   * - ``rema ingest``, ``rema maps``
     - I/O bound. Three sweeps take 10 s; one 23 GB randoms file takes about 1 min.
   * - ``rema calibrate``
     - CPU. The red-sequence fit runs in float64, which consumer GPUs run slowly. 75 deg² take
       17 min.
   * - ``rema blind``
     - GPU. 75 deg² with the footprint and wcen centring take 15 min, plus 1.5 min of
       spectroscopic post-processing. The CPU is 4–5 times slower.
   * - ``rema scan``
     - Either. 25 positions take under a minute on the CPU. A GPU does 4,000 positions in 72 s.

Tests
-----

.. code-block:: bash

   pytest -m "not slow"                 # unit tests and doctests, on the CPU
   pytest                               # adds the end-to-end runs on mock clusters
   pytest -m data                       # needs the local DR11 files (REMA_DR11_DIR or LEGACYSURVEY_DIR)
   JAX_PLATFORMS=cuda pytest -m gpu     # GPU compile-time check

Documentation
-------------

.. code-block:: bash

   pip install -e ".[docs]"
   make -C docs html                    # then open docs/_build/html/index.html

The notebooks are stored with their outputs, so the build does not need the DR11 data.

On GitHub, the ``docs`` workflow (``.github/workflows/docs.yml``) builds the documentation, with
warnings as errors, whenever a push touches the package or the docs. It keeps the HTML as the
``docs-html`` artifact of the run. The published documentation is built by Read the Docs
(``.readthedocs.yaml``) at https://rema-jax.readthedocs.io/en/latest/. To publish on GitHub Pages as well,
set the repository variable ``DOCS_DEPLOY`` to ``true`` and choose "GitHub Actions" as the Pages
source; pushes to ``main`` then deploy the site.
