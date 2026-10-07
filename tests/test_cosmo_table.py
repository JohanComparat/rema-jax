"""Cosmology configuration and the traced / differentiable distance table."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rema.config import CosmologyConfig, RemaConfig, parse_cosmology_overrides
from rema.model.cosmo import PARAMS, CosmoTable, _float64


def test_default_table_is_the_old_one():
    """The default configuration reproduces the tables built before the cosmology was configurable."""
    from ggah_mod.cosmology import (Cosmology, angular_diameter_distance, comoving_distance,
                                    comoving_volume_element, hubble_e)

    tab = CosmoTable.from_config(CosmologyConfig())
    z = np.arange(0.0, 2.0 + 5e-4, 1e-3)
    with _float64():
        c = Cosmology.create(Omega_m=0.3, h=0.7)
        zz = jnp.asarray(z, jnp.float64)
        ref = [np.asarray(f(zz, c)) for f in (angular_diameter_distance, comoving_distance,
                                              hubble_e, comoving_volume_element)]
    for got, want in zip((tab.da_tab, tab.dc_tab, tab.ez_tab, tab.dvdz_tab), ref):
        np.testing.assert_allclose(np.asarray(got), want.astype(np.float32), rtol=1e-6)
    assert np.asarray(tab.da_tab).dtype == np.float32
    old = CosmoTable.create(Omega_m=0.3, h=0.7)
    assert jax.tree_util.tree_structure(old) == jax.tree_util.tree_structure(tab)
    np.testing.assert_array_equal(np.asarray(old.da_tab), np.asarray(tab.da_tab))
    assert float(tab.h) == pytest.approx(0.7)


def test_traced_table_matches_concrete():
    from ggah_mod.cosmology import Cosmology

    c = CosmologyConfig(Omega_m=0.27, w0=-0.9)
    concrete = CosmoTable.from_config(c, zmax=1.0, dz=0.01)
    leaves = {f.name: getattr(c, f.name) for f in dataclasses.fields(c)}

    @jax.jit
    def build(om):
        return CosmoTable.from_cosmology(Cosmology(**{**leaves, "Omega_m": om}),
                                         jnp.linspace(0.0, 1.0, 101))

    with _float64():
        traced = build(jnp.asarray(0.27, jnp.float64))
        np.testing.assert_allclose(np.asarray(traced.da_tab), np.asarray(concrete.da_tab), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(traced.ez_tab), np.asarray(concrete.ez_tab), rtol=1e-6)


def test_distance_response_to_omega_m():
    """D_A [Mpc/h] falls by 0.4 to 2.6 % between z = 0.1 and 0.9 for Omega_m 0.30 -> 0.35."""
    z = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    fid, hi = CosmoTable.create(), CosmoTable.create(Omega_m=0.35)
    ratio = np.asarray(hi.da(z)) / np.asarray(fid.da(z)) - 1
    np.testing.assert_allclose(ratio, [-0.0037, -0.0107, -0.0168, -0.0219, -0.0262], atol=4e-4)


def test_jvp_matches_finite_differences():
    c = CosmologyConfig()
    tab, der = CosmoTable.jvp(c, ("Omega_m", "w0", "h", "Omega_b"), zmax=1.0, dz=0.01)
    np.testing.assert_allclose(np.asarray(tab.da_tab),
                               np.asarray(CosmoTable.from_config(c, zmax=1.0, dz=0.01).da_tab))
    for p, step in (("Omega_m", 1e-3), ("w0", 1e-3), ("h", 1e-3)):
        up = CosmoTable.from_config(dataclasses.replace(c, **{p: getattr(c, p) + step}), zmax=1.0, dz=0.01)
        dn = CosmoTable.from_config(dataclasses.replace(c, **{p: getattr(c, p) - step}), zmax=1.0, dz=0.01)
        for name in ("da_tab", "ez_tab"):
            fd = (np.asarray(getattr(up, name), np.float64) - np.asarray(getattr(dn, name), np.float64)) / (2 * step)
            ad = np.asarray(getattr(der[p], name))
            scale = np.abs(np.asarray(getattr(tab, name))).max()
            np.testing.assert_allclose(ad[1:], fd[1:], atol=2e-3 * scale)
        assert float(der[p].params[PARAMS.index(p)]) == 1.0
    # The baryons sit inside Omega_m: the expansion history does not see them.
    for name in ("da_tab", "dc_tab", "ez_tab", "dvdz_tab"):
        assert not np.any(np.asarray(getattr(der["Omega_b"], name)))
    # h enters only through radiation and neutrinos: a 1e-4-level effect on D_A [Mpc/h].
    rel = np.abs(np.asarray(der["h"].da_tab[1:]) / np.asarray(tab.da_tab[1:]))
    assert rel.max() < 2e-3
    # D_A falls with Omega_m and rises with w0 (less acceleration).
    assert np.all(np.asarray(der["Omega_m"].da_tab[10:]) < 0)
    assert np.all(np.asarray(der["w0"].da_tab[10:]) < 0)
    with pytest.raises(ValueError, match="unknown"):
        CosmoTable.jvp(c, ("sigma8",))


def test_new_cosmology_does_not_recompile():
    f = jax.jit(lambda t, z: t.mpc_per_deg(z))
    a, b = CosmoTable.create(), CosmoTable.create(Omega_m=0.25, w0=-0.8)
    assert float(f(a, 0.5)) > float(f(b, 0.5))
    assert f._cache_size() == 1


def test_config_round_trip_and_old_yaml():
    cfg = RemaConfig().replace(cosmology={"Omega_m": 0.25, "w0": -0.9})
    assert RemaConfig.from_dict(cfg.to_dict()) == cfg
    old = RemaConfig.from_dict({"cosmology": {"Omega_m": 0.3, "h": 0.7}})
    assert old.cosmology == CosmologyConfig()
    assert cfg.cosmology.label() == "Omega_m=0.25,w0=-0.9"
    assert cfg.cosmology.header() == {"OMEGAM": 0.25, "HUBBLE": 0.7, "OMEGAB": 0.0493, "MNU": 0.06,
                                      "W0": -0.9, "WA": 0.0}
    assert float(cfg.cosmology.to_ggah().w0) == -0.9


def test_parse_cosmology_overrides():
    assert parse_cosmology_overrides(None) == {}
    assert parse_cosmology_overrides(["h=0.65, sum_mnu=0.12", ""]) == {"h": 0.65, "sum_mnu": 0.12}
    for bad in (["sigma8=0.8"], ["Omega_m"]):
        with pytest.raises(ValueError, match="KEY=VALUE"):
            parse_cosmology_overrides(bad)
