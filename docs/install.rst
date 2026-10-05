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
