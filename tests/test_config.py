import pytest

from rema.config import RemaConfig, StageRadius


def test_roundtrip_yaml(tmp_path):
    cfg = RemaConfig().replace(model={"chisq_mode": "mag", "zrange": (0.1, 0.8)})
    path = tmp_path / "cfg.yml"
    cfg.to_yaml(path)
    back = RemaConfig.from_yaml(path)
    assert back == cfg
    assert back.model.zrange == (0.1, 0.8)
    assert isinstance(back.richness.percolation, StageRadius)


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        RemaConfig.from_dict({"model": {"not_a_key": 1}})


def test_ref_index():
    assert RemaConfig().ref_index == 3
