"""Settings of the cluster randoms.

They are kept out of :class:`rema.config.RemaConfig`, whose YAML is stored with every
calibration and catalogue and hashed in the blind-mode checkpoints: the randoms are a
downstream product of a finished catalogue, configured on their own and recorded in the
headers of what they write.

>>> c = RandomsConfig()
>>> c.vlim.lstar_frac, c.vlim.nsigma, c.generate.n_samp
(0.2, 5.0, 100)
>>> RandomsConfig.from_dict(c.to_dict()) == c
True
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import _build, _merge, _to_plain


@dataclass(frozen=True)
class VlimConfig:
    """The z_vlim map: where a galaxy of ``lstar_frac`` L* is detected at ``nsigma``."""

    lstar_frac: float = 0.2             # luminosity limit in units of L* (redMaPPer vlim_lstar)
    nsigma: float = 5.0                 # depth significance (rema's members: S/N >= 5)
    band: str | None = None             # depth band (None: the reference band)
    nside: int = 1024                   # HEALPix NSIDE of the map (NESTED)
    glat_min: float = 15.0              # |b| cut of the run (DR11: 15 deg)


@dataclass(frozen=True)
class GenerateConfig:
    """Copies of the clusters at random positions, and the cuts they must pass."""

    n_samp: int = 100                   # copies per cluster
    # Catalogue cuts LAMBDA / SCALEVAL >= lambda_det_min and MASKFRAC < max_maskfrac
    # (None: the catalogue's richness.minlambda and mask.max_maskfrac).
    lambda_det_min: float | None = None
    max_maskfrac: float | None = None
    pool_files: int = 4                 # DR11 randoms files of the position pool
    seed: int = 1
    batch: int = 16384                  # positions per aperture-completeness call


@dataclass(frozen=True)
class RandomsConfig:
    vlim: VlimConfig = field(default_factory=VlimConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "RandomsConfig":
        return _build(cls, d or {})

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RandomsConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def detection_cuts(self, cfg) -> tuple[float, float]:
        """(lambda_det_min, max_maskfrac), from the catalogue configuration ``cfg`` when unset.

        >>> from rema.config import RemaConfig
        >>> RandomsConfig().detection_cuts(RemaConfig())
        (3.0, 0.2)
        """
        g = self.generate
        lam = cfg.richness.minlambda if g.lambda_det_min is None else g.lambda_det_min
        mf = cfg.mask.max_maskfrac if g.max_maskfrac is None else g.max_maskfrac
        return float(lam), float(mf)

    def updated(self, d: dict[str, Any]) -> "RandomsConfig":
        """Copy with a nested dict of updates merged in (unknown keys rejected).

        >>> RandomsConfig().updated({"vlim": {"nsigma": 10}}).vlim.nsigma
        10.0
        """
        base = self.to_dict()
        _merge(base, d, "randoms configuration")
        return RandomsConfig.from_dict(base)
