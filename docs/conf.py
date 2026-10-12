"""Sphinx configuration for the rema documentation (``make -C docs html``)."""

import dataclasses
import os
import sys

import pybtex.plugin
import sphinxcontrib.bibtex.plugin
from pybtex.style.formatting.plain import Style as PlainStyle
from pybtex.style.template import join, node, sentence
from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year import AuthorYearReferenceStyle

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
    "sphinxcontrib.bibtex",
    "myst_nb",
]

# Citations: :cite:t:`key` gives "Rykoff et al. (2014)", :cite:p:`key{Fig. 3}` gives
# "(Rykoff et al., 2014, Fig. 3)"; every cited paper is listed on the Literature page.
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year_round"
bibtex_default_style = "rema"

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


@dataclasses.dataclass
class _RoundAuthorYear(AuthorYearReferenceStyle):
    """Author-year citations in round brackets, as in the astronomy journals."""

    bracket_parenthetical: BracketStyle = dataclasses.field(
        default_factory=lambda: BracketStyle("(", ")", sep="; ", sep2="; ", last_sep="; "))
    bracket_textual: BracketStyle = dataclasses.field(default_factory=lambda: BracketStyle("(", ")"))
    bracket_author: BracketStyle = dataclasses.field(default_factory=lambda: BracketStyle("(", ")"))
    bracket_label: BracketStyle = dataclasses.field(default_factory=lambda: BracketStyle("(", ")"))
    bracket_year: BracketStyle = dataclasses.field(default_factory=lambda: BracketStyle("(", ")"))


@node
def _names_et_al(children, context, role, **kwargs):
    """The first three names of a role, then "et al." when there are more."""
    persons = context["entry"].persons[role]
    style = context["style"]
    shown = [p for p in persons if str(p) != "others"]
    more = len(shown) > 3 or len(shown) < len(persons)
    names = [style.format_name(p, style.abbreviate_names) for p in shown[:3]]
    if more:
        return join(sep=", ")[names + ["et al."]].format_data(context)
    return join(sep=", ", sep2=" and ", last_sep=", and ")[names].format_data(context)


class _RemaStyle(PlainStyle):
    """pybtex's plain style with at most three authors per entry."""

    def format_names(self, role, as_sentence=True):
        formatted = _names_et_al(role)
        return sentence[formatted] if as_sentence else formatted


sphinxcontrib.bibtex.plugin.register_plugin(
    "sphinxcontrib.bibtex.style.referencing", "author_year_round", _RoundAuthorYear)
pybtex.plugin.register_plugin("pybtex.style.formatting", "rema", _RemaStyle)


def _escape_bars(app, what, name, obj, options, lines):
    """Docstrings are plain text: |x| is an absolute value, not a reST substitution."""
    lines[:] = [line.replace("|", "\\|") for line in lines]


def setup(app):
    app.connect("autodoc-process-docstring", _escape_bars)
