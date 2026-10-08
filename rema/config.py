"""Run configuration.

A :class:`RemaConfig` is a tree of frozen dataclasses, one per stage, written to and read from
YAML. Defaults follow the redMaPPer v0.7.7 configuration of the DR10 run (``run_zred_iter1.yml``); the
redMaPPer key a field corresponds to is given in its comment when the name differs.

>>> cfg = RemaConfig()
>>> cfg.survey.bands
('g', 'r', 'i', 'z')
>>> cfg.richness.percolation.r0, cfg.richness.percolation.beta
(1.0, 0.2)
>>> RemaConfig.from_dict(cfg.to_dict()) == cfg
True
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Legacy Surveys MASKBITS (DR11 bitmask page): bit number -> name.
MASKBITS = {
    0: "NPRIMARY", 1: "BRIGHT", 2: "SATUR_G", 3: "SATUR_R", 4: "SATUR_Z", 5: "ALLMASK_G",
    6: "ALLMASK_R", 7: "ALLMASK_Z", 8: "WISEM1", 9: "WISEM2", 10: "BAILOUT", 11: "MEDIUM",
    12: "GALAXY", 13: "CLUSTER", 14: "SATUR_I", 15: "ALLMASK_I", 16: "SUB_BLOB",
    17: "RESOLVED", 18: "MCLOUDS",
}

# DECam A/E(B-V) used for the Legacy Surveys MW_TRANSMISSION columns (DR11 catalogs page).
EXTINCTION_DECAM = {"u": 3.995, "g": 3.214, "r": 2.165, "i": 1.592, "z": 1.211, "Y": 1.064}


@dataclass(frozen=True)
class SurveyConfig:
    """Photometric catalogue definition and cleaning."""

    bands: tuple[str, ...] = ("g", "r", "i", "z")
    ref_band: str = "z"
    zeropoint: float = 22.5
    # Reject objects whose MASKBITS has any of these bits (unless LARGEGALAXY is kept).
    # Bright and medium stars, saturation and all-mask in the photometric bands, bailouts,
    # large galaxies, globular clusters, resolved dwarfs, Magellanic clouds.
    maskbits_reject: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 17, 18)
    keep_largegalaxy: bool = True       # keep objects with FITBITS LARGEGALAXY (bit 9), e.g. BCGs
    reject_psf: bool = True             # star/galaxy separation: drop TYPE == 'PSF'
    nobs_min: int = 1                   # in every band of `bands`
    ref_snr_min: float = 5.0            # selection: reference-band S/N
    mag_max: float | None = None        # selection: reference magnitude; None -> m*(zmax) + 2.5
    ebv_max: float | None = None        # selection: E(B-V) < ebv_max, galaxies and randoms (None: no cut)
    flux_floor: float = 0.015           # fractional flux-error floor added in quadrature
    zspec_range: tuple[float, float] = (0.001, 3.0)
    zspec_exclude_surveys: tuple[str, ...] = ("COSMOS2015",)   # photometric "reference" redshifts


@dataclass(frozen=True)
class CosmologyConfig:
    """Flat w0waCDM with massive neutrinos (ggah_mod); lengths are in h^-1 Mpc as in redMaPPer.

    ``Omega_m`` includes the neutrinos. The finder only sees the expansion history: D_A(z) in
    h^-1 Mpc sets every aperture and E(z) the zred volume factor, so ``Omega_b`` has no effect and
    ``h`` enters only through the radiation and neutrino densities. ``Omega_b`` and ``sum_mnu``
    are ggah_mod's defaults, the values used before they were configurable.

    >>> c = CosmologyConfig(Omega_m=0.25)
    >>> c.label()
    'Omega_m=0.25'
    >>> c.header()["OMEGAM"], CosmologyConfig().label()
    (0.25, 'fiducial')
    """

    Omega_m: float = 0.3
    h: float = 0.7
    Omega_b: float = 0.0493
    sum_mnu: float = 0.06               # eV, normal hierarchy
    w0: float = -1.0
    wa: float = 0.0

    def to_ggah(self):
        """The ggah_mod :class:`~ggah_mod.cosmology.Cosmology` (validated)."""
        from ggah_mod.cosmology import Cosmology

        return Cosmology.create(**dataclasses.asdict(self))

    def header(self) -> dict[str, float]:
        """FITS keywords of the cosmology."""
        return {key: float(getattr(self, name)) for name, key in _COSMO_KEYS.items()}

    def label(self) -> str:
        """The parameters that differ from the defaults, ``'fiducial'`` when none do."""
        ref = CosmologyConfig()
        diff = [f"{f.name}={getattr(self, f.name):g}" for f in dataclasses.fields(self)
                if getattr(self, f.name) != getattr(ref, f.name)]
        return ",".join(diff) or "fiducial"


_COSMO_KEYS = {"Omega_m": "OMEGAM", "h": "HUBBLE", "Omega_b": "OMEGAB", "sum_mnu": "MNU",
               "w0": "W0", "wa": "WA"}


def parse_cosmology_overrides(items) -> dict[str, float]:
    """``["Omega_m=0.25", "w0=-0.8,wa=0.1"]`` -> ``{"Omega_m": 0.25, "w0": -0.8, "wa": 0.1}``.

    >>> parse_cosmology_overrides(["Omega_m=0.25", "w0=-0.8,wa=0.1"])
    {'Omega_m': 0.25, 'w0': -0.8, 'wa': 0.1}
    """
    known = [f.name for f in dataclasses.fields(CosmologyConfig)]
    out: dict[str, float] = {}
    for item in items or ():
        for pair in str(item).split(","):
            if not pair.strip():
                continue
            key, sep, value = pair.partition("=")
            key = key.strip()
            if not sep or key not in known:
                raise ValueError(f"cosmology override {pair!r}: expected KEY=VALUE with KEY in {known}")
            out[key] = float(value)
    return out


@dataclass(frozen=True)
class ModelConfig:
    """Red-sequence filter definition."""

    zrange: tuple[float, float] = (0.05, 0.90)
    mstar: str = "des_z03"              # packaged table name, or a path to a FITS table (Z, MSTAR)
    template: str = "bc03_legacy_grizw1"  # colours seeding the calibration: packaged name or FITS table
    alpha: float = -1.0                 # calib_lumfunc_alpha
    lval_reference: float = 0.2         # luminosity cut of the richness, in L*
    chisq_max: float = 20.0
    chisq_mode: str = "lupt"            # "lupt" asinh magnitudes (default) or "mag" (redMaPPer)
    zbin_coarse: float = 0.005          # zredc_binsize_coarse: z grid of zred
    zbin_fine: float = 0.001            # z grid of the tabulated model and background
    nfw_rs: float = 0.15
    nfw_rcore: float = 0.1
    rsig: float = 0.05                  # softness of the radial cut
    filter: str = "redsequence"         # membership filter: "redsequence" (colour chi^2) or
                                        # "photoz" (DR11 photo-z, any colour; see PhotozConfig)


@dataclass(frozen=True)
class BackgroundConfig:
    zbinsize: float = 0.02              # bkg_zbinsize
    chisqbinsize: float = 0.5           # bkg_chisqbinsize
    refmagbinsize: float = 0.2          # bkg_refmagbinsize
    zredbinsize: float = 0.01           # bkg_zredbinsize
    lmax_faint: float = 0.1             # galaxies counted down to m*(z) - 2.5 log10(lmax_faint)
    min_counts: float = 50.0            # adaptive merging of sparse magnitude bins
    smooth_z: float = 0.02
    smooth_m: float = 0.2


@dataclass(frozen=True)
class PhotozConfig:
    """Galaxy photo-z of the photo-z filter (``model.filter: photoz``) and of ``rema pscd``.

    The photo-z of a galaxy is a Gaussian of mean ZPHOT and width

        s = max(err_scale(m) ZPHOT_STD, err_floor (1 + ZPHOT)),

    with ``err_scale`` piecewise linear in the reference magnitude m through the nodes
    ``err_scale_mag`` (and flat beyond them); without nodes ``err_scale`` holds one value.
    """

    err_scale: tuple[float, ...] = (1.0,)
    err_scale_mag: tuple[float, ...] = ()
    err_floor: float = 0.01
    nsig_max: float = 4.0               # members within nsig_max s of the cluster redshift
    mag_max: float | None = None        # extra reference-magnitude limit of the members (and
                                        # of the completeness); None: the catalogue's
    zbinsize: float = 0.005             # z grid of the stacked photo-z background
    zrange_bkg: tuple[float, float] = (0.0, 1.6)
    min_counts: float = 50.0            # adaptive merging of sparse magnitude bins


@dataclass(frozen=True)
class PscdConfig:
    """``rema pscd``: Photo-z Space Cluster Detection, the AMICO matched filter (Bellagamba et al.
    2018; Maturi et al. 2019) in (position, magnitude, photo-z). The galaxy photo-z, their
    magnitude limit (``photoz.mag_max``) and the noise are those of :class:`PhotozConfig`.

    The cluster template is the KiDS one (Maturi et al. 2019): an NFW profile of concentration
    ``concentration`` and radius ``r200``, truncated at ``rtrunc`` r200 in projection, times a
    Schechter function of slope ``alpha`` around the configuration's m*(z), with ``n200``
    galaxies brighter than m* + ``dmag_n200`` inside r200: amplitude A = 1 is that template. The
    profile is flat inside ``rcore``.
    """

    pixel: float = 0.01                 # deg, of the (gnomonic) amplitude grid
    dz: float = 0.01                    # redshift step of the grid
    zrange: tuple[float, float] | None = None   # None: model.zrange
    r200: float = 0.7                   # h^-1 Mpc (1 Mpc for h = 0.7: M200 ~ 1e14 Msun/h)
    concentration: float = 3.59
    rtrunc: float = 1.35                # profile truncation, in r200
    rcore: float = 0.1                  # h^-1 Mpc, flat core of the projected NFW (as rema's
                                        # radial filter; the KiDS template has none). A cusp
                                        # narrower than the pixels biases A low at high z.
    alpha: float = -1.06
    n200: float = 22.9
    dmag_n200: float = 2.0
    snr_min: float = 3.0                # detections down to this S/N (thresholds come later)
    snr_kind: str = "amico"             # S/N of the extraction: "amico", A/sigma_A with the
                                        # cluster's shot noise (Bellagamba et al. 2018, Eq. 11),
                                        # or "background", A sqrt(alpha) (background noise only)
    max_detections: int = 1000000
    pmin: float = 0.01                  # members kept in the output: P(i in j) >= pmin
    lambda_star_dmag: float = 1.5       # LAMBDA_STAR: members brighter than m* + 1.5, r < r200
    max_maskfrac: float = 0.2
    min_coverage: float = 0.2           # cells whose profile is less covered by the footprint
                                        # are not searched (edges of the region)
    tile: int = 32                      # pixels per side of the tiles of the maximum search


@dataclass(frozen=True)
class NullConfig:
    """Null tests: shuffled galaxy properties, which keep positions and magnitudes.

    "photoz" permutes the (ZPHOT, ZPHOT_STD) pairs among galaxies of the same reference magnitude
    (bins of ``magbin``): the angular clustering stays, the redshift coherence of real structures
    is lost. "colour" permutes the fluxes, rescaled to the galaxy's own reference flux: the red
    sequence of real clusters is lost.
    """

    shuffle: str = "none"               # "none", "photoz" or "colour"
    seed: int = 1
    magbin: float = 0.1


@dataclass(frozen=True)
class ZredConfig:
    use_lndet: bool = False             # add -1/2 ln det S to the zred likelihood
    zred_e_min: float = 0.005
    nrefine: int = 2                    # neighbours on each side for the parabola refinement


@dataclass(frozen=True)
class StageRadius:
    r0: float = 1.0
    beta: float = 0.2


@dataclass(frozen=True)
class RichnessConfig:
    firstpass: StageRadius = field(default_factory=lambda: StageRadius(0.5, 0.0))
    likelihoods: StageRadius = field(default_factory=lambda: StageRadius(1.0, 0.2))
    percolation: StageRadius = field(default_factory=lambda: StageRadius(1.0, 0.2))
    zscan: StageRadius = field(default_factory=lambda: StageRadius(0.5, 0.0))
    lambda_bounds: tuple[float, float] = (0.5, 2000.0)
    n_grid: int = 32                    # log-spaced lambda values bracketing the root
    n_newton: int = 4
    firstpass_niter: int = 1            # lambda + z_lambda iterations of the first pass (DR10: 1)
    minlambda: float = 3.0
    min_lnlamlike: float | None = None  # candidates also need LNLAMLIKE >= this after the first
                                        # pass and the likelihood pass (None: no cut, as
                                        # redMaPPer; the photo-z filter needs one)
    maxrad_factor: float = 1.2          # neighbour radius = factor * r0 * 3**beta (h^-1 Mpc)


@dataclass(frozen=True)
class MaskConfig:
    nside_coverage: int = 128           # survey (brick) coverage, from all randoms
    nside_fine: int = 1024              # good fraction and depth, from good randoms
    randoms_density: float = 2500.0     # per deg^2 and per randoms file
    n_r: int = 24                       # Gauss-Legendre radial nodes of the aperture quadrature
    n_phi: int = 16                     # azimuths
    n_mag: int = 24                     # magnitude nodes of the selection integral
    max_maskfrac: float = 0.2


@dataclass(frozen=True)
class ZlambdaConfig:
    topfrac: float = 0.7
    pcol_soft: float = 0.04
    maxiter: int = 5
    tol: float = 2e-4
    ngrid: int = 21
    half_width: float = 0.03            # half width of the local z grid
    npzbins: int = 21
    pivot: float = 30.0


@dataclass(frozen=True)
class CenteringConfig:
    method: str = "auto"                # "bcg", "wcen", or "auto": wcen when the calibration has it
    maxcen: int = 5                     # percolation_maxcen
    pbcg_cut: float = 0.5               # percolation_pbcg_cut
    scan_maxrad: float = 0.4            # percolation_maxrad in scan mode (h^-1 Mpc)
    wcen_rsoft: float = 0.05
    wcen_pivot: float = 30.0
    wcen_maxlambda: float = 100.0
    wcen_zred_chisq_max: float = 100.0
    wcen_uselum: bool = True            # weight the connectivity w by luminosity
    wcen_ncand: int = 64                # candidates per cluster in the pairwise w (static cap)
    wcen_minlambda: float = 10.0        # calibration sample: wcen_minlambda < lambda/S < wcen_maxlambda
    wcen_cal_zrange: tuple[float, float] = (0.05, 0.60)


@dataclass(frozen=True)
class PercolationConfig:
    rmask_0: float = 1.5
    rmask_beta: float = 0.2
    rmask_gamma: float = 0.0
    rmask_zpivot: float = 0.3
    lmask: float = 0.1
    niter: int = 2
    member_pmin: float = 0.01


@dataclass(frozen=True)
class SeedConfig:
    chisq_max: float = 20.0
    dmag_max: float = 1.75              # seeds brighter than m*(zred) + dmag_max
    zphot_err_max: float = 0.1          # photo-z filter: seeds with s / (1 + ZPHOT) below this


@dataclass(frozen=True)
class ScanConfig:
    zrange: tuple[float, float] = (0.05, 1.0)
    zstep: float = 0.005


@dataclass(frozen=True)
class SpecConfig:
    min_members: int = 3
    vmax_init: float = 5000.0           # km/s window around the first biweight location
    nsigma_clip: float = 3.0
    niter: int = 20
    gapper_nmax: int = 15               # gapper below this number of members, biweight above
    c_location: float = 6.0
    c_scale: float = 9.0
    nboot: int = 64
    seed: int = 12345


@dataclass(frozen=True)
class CalibConfig:
    niter: int = 3
    pcut: float = 0.3
    color_pcut: float = 0.7
    minlambda: float = 5.0
    nsig_trunc: float = 1.5             # calib_color_nsig
    color_nodesize: float = 0.05
    slope_nodesize: float = 0.1
    scatter_nodesize: float = 0.1
    pivot_nodesize: float = 0.1
    corr_nodesize: float = 0.15
    smooth_prior: float = 1.0           # strength of the second-difference prior on the nodes
    wcen: bool = True                   # fit the wcen centring model after the red sequence
    wcen_niter: int = 1                 # 1: fit on BCG-centred clusters (as in DR10); 2: refit
                                        # on wcen-centred clusters
    seed: int = 12345                   # random centres and satellites of the wcen calibration


@dataclass(frozen=True)
class RemaConfig:
    survey: SurveyConfig = field(default_factory=SurveyConfig)
    cosmology: CosmologyConfig = field(default_factory=CosmologyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    zred: ZredConfig = field(default_factory=ZredConfig)
    richness: RichnessConfig = field(default_factory=RichnessConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    zlambda: ZlambdaConfig = field(default_factory=ZlambdaConfig)
    centering: CenteringConfig = field(default_factory=CenteringConfig)
    percolation: PercolationConfig = field(default_factory=PercolationConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    spec: SpecConfig = field(default_factory=SpecConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
    photoz: PhotozConfig = field(default_factory=PhotozConfig)
    pscd: PscdConfig = field(default_factory=PscdConfig)
    null: NullConfig = field(default_factory=NullConfig)

    # ------------------------------------------------------------------ helpers
    @property
    def ref_index(self) -> int:
        return self.survey.bands.index(self.survey.ref_band)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "RemaConfig":
        return _build(cls, d or {})

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RemaConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def replace(self, **sections: Any) -> "RemaConfig":
        """Copy with whole sections, or ``section={"key": value}`` updates, replaced.

        >>> RemaConfig().replace(model={"chisq_mode": "mag"}).model.chisq_mode
        'mag'
        """
        updates = {}
        for name, value in sections.items():
            current = getattr(self, name)
            if isinstance(value, dict):
                value = _build(type(current), {**dataclasses.asdict(current), **value})
            updates[name] = value
        return dataclasses.replace(self, **updates)


def apply_overrides(cfg: RemaConfig, items) -> RemaConfig:
    """``cfg`` with ``SECTION.KEY[.SUB]=VALUE`` overrides (VALUE read as YAML).

    An item ``@FILE`` merges a partial YAML configuration (e.g. ``model: {filter: photoz}``).
    Nested sections are merged, so the other keys keep their values; unknown keys are rejected.

    >>> c = apply_overrides(RemaConfig(), ["model.filter=photoz", "richness.firstpass.r0=0.4",
    ...                                    "photoz.err_scale=[0.8, 0.6]", "photoz.mag_max=22"])
    >>> c.model.filter, c.richness.firstpass, c.photoz.err_scale, c.photoz.mag_max
    ('photoz', StageRadius(r0=0.4, beta=0.0), (0.8, 0.6), 22.0)
    """
    d = cfg.to_dict()
    for item in items or ():
        if str(item).startswith("@"):
            _merge(d, yaml.safe_load(Path(str(item)[1:]).read_text()) or {}, str(item))
            continue
        key, sep, value = str(item).partition("=")
        path = key.strip().split(".")
        if not sep or len(path) < 2:
            raise ValueError(f"override {item!r}: expected SECTION.KEY=VALUE")
        node = d
        for name in path[:-1]:
            if not isinstance(node.get(name), dict):
                raise KeyError(f"override {item!r}: {name!r} is not a configuration section")
            node = node[name]
        if path[-1] not in node:
            raise KeyError(f"override {item!r}: unknown key {path[-1]!r}; known: {sorted(node)}")
        node[path[-1]] = yaml.safe_load(value)
    return RemaConfig.from_dict(d)


def _merge(d: dict, upd: dict, where: str) -> None:
    """Merge the nested dict ``upd`` into ``d`` in place; unknown keys are rejected."""
    for key, value in upd.items():
        if key not in d:
            raise KeyError(f"{where}: unknown key {key!r}; known: {sorted(d)}")
        if isinstance(d[key], dict) and isinstance(value, dict):
            _merge(d[key], value, where)
        else:
            d[key] = value


def _to_plain(obj: Any) -> Any:
    """Tuples -> lists so that the YAML is plain; keeps nested dicts."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _build(cls: type, d: dict[str, Any]) -> Any:
    """Instantiate a (nested) frozen dataclass from a plain dict, rejecting unknown keys."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(d) - set(fields)
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown key(s) {sorted(unknown)}; known: {sorted(fields)}")
    kwargs = {}
    for name, value in d.items():
        ftype = _field_type(cls, name)
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            value = _build(ftype, value)
        elif isinstance(value, list):
            value = tuple(value)
        elif isinstance(value, (int, str)) and not isinstance(value, bool) and _is_float(ftype):
            value = float(value)            # "mag_max: 22" and "1e6" (a string in YAML 1.1)
        kwargs[name] = value
    return cls(**kwargs)


def _is_float(ftype: Any) -> bool:
    """A ``float`` or ``float | None`` annotation."""
    import types
    import typing

    if ftype is float:
        return True
    args = typing.get_args(ftype)
    return (typing.get_origin(ftype) in (typing.Union, types.UnionType) and float in args
            and int not in args)


def _field_type(cls: type, name: str) -> Any:
    """Resolve a field's annotation (string annotations because of `from __future__`)."""
    import typing

    hints = typing.get_type_hints(cls)
    return hints[name]
