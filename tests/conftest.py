"""Shared fixtures and locations of the optional local data.

Tests marked ``data`` read the local DR11 sweeps, DR11 randoms or redMaPPer reference files and
are skipped when those are absent.  Locations can be overridden with environment variables:
REMA_DR11_DIR (the dr11/south directory), else LEGACYSURVEY_DIR (the Legacy Surveys root).
"""

import os
from pathlib import Path

import numpy as np
import pytest

_LS_DIR = os.environ.get("LEGACYSURVEY_DIR", "/home/comparat/data/legacysurvey")
DR11_DIR = Path(os.environ.get("REMA_DR11_DIR", f"{_LS_DIR}/dr11/south"))
REDMAPPER_DIR = Path(os.environ.get("REMA_REDMAPPER_DIR", "/home/comparat/data/redmapper"))

SWEEP = DR11_DIR / "sweep" / "11.0" / "sweep-000m005-005p000.fits"
SWEEP_PZ = DR11_DIR / "sweep" / "11.0-photo-z" / "sweep-000m005-005p000-pz.fits"
RANDOMS = DR11_DIR / "randoms" / "randoms-south-1-0.fits"
DR10_PARS = (REDMAPPER_DIR / "data" / "cal" / "legacy_dr10_grz_z_v0.3"
             / "legacy_dr10_south_v0.3_grz_z_cal_iter1_pars.fit")


def require(*paths):
    """Skip the calling test unless every path exists (and is not empty)."""
    for p in paths:
        p = Path(p)
        if not p.exists() or p.stat().st_size == 0:
            pytest.skip(f"local data not available: {p}")


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20261001)
