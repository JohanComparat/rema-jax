"""Sphinx configuration for the rema documentation (``make -C docs html``)."""

import os
import sys

# autodoc imports rema; no GPU is needed for that.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(".."))

import rema  # noqa: E402

project = "rema"
author = "Johan Comparat"
copyright = "2026, Johan Comparat"
release = rema.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",
]

# The notebooks need the DR11 data: they are stored with their outputs and never run here.
nb_execution_mode = "off"
myst_heading_anchors = 3

napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": False}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
}

html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": 3}
html_title = f"rema {release}"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/.ipynb_checkpoints", "examples"]


def _escape_bars(app, what, name, obj, options, lines):
    """Docstrings are plain text: |x| is an absolute value, not a reST substitution."""
    lines[:] = [line.replace("|", "\\|") for line in lines]


def setup(app):
    app.connect("autodoc-process-docstring", _escape_bars)
