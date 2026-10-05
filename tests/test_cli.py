"""Command line: the configuration a run uses."""

import logging

import numpy as np

from rema import cli
from rema.config import RemaConfig
from rema.io.tables import table_hdu, write_fits


def test_run_config_flag_then_stored_then_defaults(tmp_path, caplog):
    stored = RemaConfig().replace(model={"chisq_mode": "mag"}, spec={"nboot": 8})
    assert cli._run_cfg(None, stored, "calibration") == stored
    assert cli._run_cfg(None, None, "calibration") == RemaConfig()
    path = tmp_path / "run.yaml"
    RemaConfig().to_yaml(path)
    with caplog.at_level(logging.WARNING, logger="rema"):
        cfg = cli._run_cfg(str(path), stored, "calibration")
    assert cfg == RemaConfig()
    assert "model.chisq_mode, spec.nboot" in caplog.text


def test_config_diff():
    a = RemaConfig()
    assert cli._config_diff(a, a) == []
    b = a.replace(richness={"minlambda": 5.0}, scan={"zrange": (0.05, 1.1)})
    assert cli._config_diff(a, b) == ["richness.minlambda", "scan.zrange"]


def test_catalog_config(tmp_path):
    cfg = RemaConfig().replace(spec={"nboot": 8})
    path = tmp_path / "cat.fits"
    write_fits(path, [table_hdu({"MEM_MATCH_ID": np.arange(2)}, extname="CLUSTERS"),
                      table_hdu({"YAML": np.array([cfg.to_yaml().encode()])}, extname="CONFIG")])
    assert cli._catalog_cfg(path) == cfg
    plain = tmp_path / "plain.fits"
    write_fits(plain, [table_hdu({"MEM_MATCH_ID": np.arange(2)}, extname="CLUSTERS")])
    assert cli._catalog_cfg(plain) is None
