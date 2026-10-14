"""External selection functions (X-ray, SZ) for cluster randoms; see :mod:`.base`."""

from . import act, erass1, psz2, spt  # noqa: F401  (register the plugins)
from .base import ExternalSelection, NoSelection, get_selection, register  # noqa: F401
