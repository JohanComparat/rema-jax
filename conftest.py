"""Repository-level pytest configuration.

Docstring examples are collected with ``--doctest-modules``.  NumPy 2 prints scalars as
``np.float64(0.5)``; the examples are written for the plain form, so the legacy scalar
representation is selected whenever doctests are collected.
"""
import numpy as np


def pytest_configure(config):
    if config.getoption("doctestmodules", default=False):
        np.set_printoptions(legacy="1.25")
