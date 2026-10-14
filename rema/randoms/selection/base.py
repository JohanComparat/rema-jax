"""Interface of an external selection function.

A sample selected by an external survey (an X-ray or SZ detection list, rescanned by rema) is
selected twice: by that survey, then by rema's optical confirmation on DR11. The randoms carry
the second through the footprint (:mod:`rema.randoms.aperture`) and the first through an
:class:`ExternalSelection`:

- ``in_footprint(ra, dec)``: the survey's area (its masks); random positions are drawn there only;
- ``own_cut(cat)``: the survey's detection cut applied to the data (e.g. S/N >= 5);
- ``p_detect(src, ra, dec)``: the probability that the survey detects, at (ra, dec), a cluster
  with the observables of ``src`` (columns of the source cluster of each copy);
- ``approximate``: True when the model is not the survey's own (eRASS1 here);
- ``provenance()``: what the products record about the model.

The detection probability is that of the measured observables (S/N, count rate) moved to the
new position; with ``eddington`` set, a plugin may average it over the true observable given the
measured one (:func:`eddington_average`).
"""

from __future__ import annotations

import numpy as np

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator: make a selection available to :func:`get_selection` under ``name``."""
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def get_selection(spec) -> "ExternalSelection":
    """Selection from a name or a dict ``{"name": ..., **parameters}`` (None: no selection).

    >>> get_selection(None).name, get_selection({"name": "none"}).approximate
    ('none', False)
    """
    if spec is None:
        spec = {"name": "none"}
    if isinstance(spec, str):
        spec = {"name": spec}
    spec = dict(spec)
    name = spec.pop("name")
    if name not in _REGISTRY:
        raise KeyError(f"unknown selection {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**spec)


class ExternalSelection:
    """Base class: everything detected everywhere (override what the survey restricts)."""

    name = "base"
    approximate = False
    columns: tuple[str, ...] = ()      # source columns p_detect and own_cut read

    def in_footprint(self, ra, dec) -> np.ndarray:
        return np.ones(np.shape(ra), bool)

    def own_cut(self, cat: dict) -> np.ndarray:
        return np.ones(len(next(iter(cat.values()))), bool)

    def p_detect(self, src: dict, ra, dec) -> np.ndarray:
        return np.ones(np.shape(ra), np.float64)

    def provenance(self) -> dict:
        return {"SELNAME": self.name, "APPROX": bool(self.approximate)}


@register("none")
class NoSelection(ExternalSelection):
    """A sample selected by rema alone (richness)."""


def eddington_average(p_of_true, y_hat, sigma, alpha: float, n: int = 16) -> np.ndarray:
    """Average of ``p_of_true(y)`` over the true observable y given a measurement ``y_hat`` with
    Gaussian error ``sigma``, under a power-law population n(y) ~ y^-alpha:

        p = int P(y) N(y_hat; y, sigma) y^-alpha dy / int N(y_hat; y, sigma) y^-alpha dy,

    by Gauss-Hermite quadrature around y_hat (nodes with y <= 0 get no weight).

    >>> float(np.round(eddington_average(lambda y: np.ones_like(y), np.array([10.0]), 1.0, 2.0)[0], 6))
    1.0
    """
    x, w = np.polynomial.hermite_e.hermegauss(n)
    y_hat, sigma = np.atleast_1d(np.asarray(y_hat, np.float64)), np.asarray(sigma, np.float64)
    y = y_hat[:, None] + np.atleast_1d(sigma)[..., None] * x[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        prior = np.where(y > 0, y ** (-alpha), 0.0)
    ww = w[None, :] * prior
    p = np.asarray(p_of_true(y), np.float64)
    return np.sum(ww * p, axis=1) / np.maximum(np.sum(ww, axis=1), 1e-300)
